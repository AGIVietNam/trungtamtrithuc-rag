"""URL resolver — detect public link pattern + convert sang direct download URL.

Hỗ trợ các nguồn share phổ biến:
  - Google Drive: drive.google.com/file/d/{ID}/view → uc?export=download&id={ID}
  - OneDrive personal: 1drv.ms / onedrive.live.com → append ?download=1
  - Dropbox: dl=0 → dl=1
  - YouTube: pass-through, dispatch youtube pipeline
  - Generic: HTTP/HTTPS bất kỳ → tải thẳng

KHÔNG hỗ trợ:
  - Link private (cần OAuth) — BE phải resolve qua Microsoft Graph rồi đẩy
    sang `/from-urls` với headers Bearer token.
  - SharePoint nội bộ TDI — dùng tài khoản Microsoft, BE/FE auth riêng.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse, parse_qs

logger = logging.getLogger(__name__)

UrlKind = Literal["document", "video_file", "youtube", "audio_file"]


@dataclass(frozen=True)
class ResolvedUrl:
    """Kết quả resolve URL — đầu vào cho worker download + ingest."""
    download_url: str
    kind: UrlKind
    suggested_filename: str
    source: str   # "gdrive" | "onedrive" | "dropbox" | "youtube" | "generic"


# --- Document / video extension classification ---

DOC_EXTS = frozenset({".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md", ".xlsx"})
VIDEO_EXTS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"})
AUDIO_EXTS = frozenset({".mp3", ".m4a", ".wav", ".ogg", ".flac"})


def _kind_from_filename(filename: str) -> UrlKind:
    """Phỏng đoán kind từ đuôi file. Default: document."""
    ext = Path(filename).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video_file"
    if ext in AUDIO_EXTS:
        return "audio_file"
    return "document"


# --- Google Drive ---

# Google Drive folder — KHÔNG hỗ trợ (cần Drive API liệt kê items)
_GDRIVE_FOLDER_PATTERN = re.compile(r"drive\.google\.com/drive/(?:u/\d+/)?folders/")

# Google Drive native binary file (PDF, DOCX user upload lên Drive)
# Hỗ trợ multi-account: drive.google.com/file/u/0/d/{id}/...
_GDRIVE_FILE_PATTERNS = [
    re.compile(r"drive\.google\.com/file/(?:u/\d+/)?d/([A-Za-z0-9_-]+)"),
    re.compile(r"drive\.google\.com/(?:u/\d+/)?open\?id=([A-Za-z0-9_-]+)"),
    re.compile(r"drive\.google\.com/uc\?(?:[^&]+&)*id=([A-Za-z0-9_-]+)"),
]

# Google Docs/Sheets/Slides — file Google native, KHÔNG download trực tiếp
# qua uc?export=download. Phải dùng /export?format=...
# Hỗ trợ multi-account: docs.google.com/document/u/0/d/{id}/...
_GDOCS_NATIVE_PATTERNS = [
    # (regex, doc_kind, export_format, suggested_ext)
    (re.compile(r"docs\.google\.com/document/(?:u/\d+/)?d/([A-Za-z0-9_-]+)"),     "document",     "docx", ".docx"),
    (re.compile(r"docs\.google\.com/spreadsheets/(?:u/\d+/)?d/([A-Za-z0-9_-]+)"), "spreadsheets", "xlsx", ".xlsx"),
    (re.compile(r"docs\.google\.com/presentation/(?:u/\d+/)?d/([A-Za-z0-9_-]+)"), "presentation", "pptx", ".pptx"),
    (re.compile(r"docs\.google\.com/drawings/(?:u/\d+/)?d/([A-Za-z0-9_-]+)"),     "drawings",     "pdf",  ".pdf"),
]

# Google Forms — không export tài liệu được, reject
_GFORMS_PATTERN = re.compile(r"docs\.google\.com/forms/")


def _resolve_gdrive(url: str) -> ResolvedUrl | None:
    if _GDRIVE_FOLDER_PATTERN.search(url):
        raise ValueError("Link Google Drive folder không hỗ trợ — vui lòng share file đơn lẻ.")
    if _GFORMS_PATTERN.search(url):
        raise ValueError("Link Google Forms không export được dạng tài liệu.")

    # 1) Google Docs/Sheets/Slides — phải dùng export endpoint
    for pat, doc_kind, fmt, ext in _GDOCS_NATIVE_PATTERNS:
        m = pat.search(url)
        if m:
            file_id = m.group(1)
            dl = f"https://docs.google.com/{doc_kind}/d/{file_id}/export?format={fmt}"
            return ResolvedUrl(
                download_url=dl,
                kind="document",
                suggested_filename=f"gdocs_{file_id}{ext}",
                source="gdocs",
            )

    # 2) Google Drive file binary — uc?export=download
    for pat in _GDRIVE_FILE_PATTERNS:
        m = pat.search(url)
        if m:
            file_id = m.group(1)
            # Endpoint download trực tiếp; với file >100MB Google trả HTML
            # confirm token — handler trong job_runner phát hiện text/html
            # response → parse confirm token → retry với token (job_runner tự lo).
            dl = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
            return ResolvedUrl(
                download_url=dl,
                kind="document",      # default; sẽ correct sau khi tải về theo content-type
                suggested_filename=f"gdrive_{file_id}",
                source="gdrive",
            )
    return None


# --- OneDrive personal (consumer) ---

def _resolve_onedrive(url: str) -> ResolvedUrl | None:
    if not ("1drv.ms" in url or "onedrive.live.com" in url):
        return None
    # Append download=1 — OneDrive trả file binary thay vì HTML preview
    sep = "&" if "?" in url else "?"
    dl = f"{url}{sep}download=1"
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name or "onedrive_file"
    return ResolvedUrl(
        download_url=dl,
        kind=_kind_from_filename(name),
        suggested_filename=name,
        source="onedrive",
    )


# --- SharePoint / OneDrive Business ---

# Pattern: {tenant}.sharepoint.com hoặc {tenant}-my.sharepoint.com
_SHAREPOINT_PATTERN = re.compile(r"https?://([a-z0-9-]+)(?:-my)?\.sharepoint\.com", re.IGNORECASE)

# Anonymous share link (đôi khi public): /:b:/g/, /:w:/g/, /:x:/g/, /:p:/g/
_SHAREPOINT_ANON_PATTERN = re.compile(r"sharepoint\.com/:([bwxp]):/g/", re.IGNORECASE)
_SHAREPOINT_FOLDER_ANON = re.compile(r"sharepoint\.com/:f:/g/", re.IGNORECASE)


def _resolve_sharepoint(url: str) -> ResolvedUrl | None:
    if not _SHAREPOINT_PATTERN.search(url):
        return None
    if _SHAREPOINT_FOLDER_ANON.search(url):
        raise ValueError("Link SharePoint folder không hỗ trợ — vui lòng share file đơn lẻ.")

    # Anonymous public share: thêm download=1 để bypass viewer
    if _SHAREPOINT_ANON_PATTERN.search(url):
        sep = "&" if "?" in url else "?"
        dl = f"{url}{sep}download=1"
        parsed = urlparse(url)
        name = Path(unquote(parsed.path)).name or "sharepoint_file"
        return ResolvedUrl(
            download_url=dl,
            kind=_kind_from_filename(name),
            suggested_filename=name,
            source="sharepoint_anon",
        )

    # Mọi link SharePoint khác (stream.aspx, /personal/, /sites/...) → cần
    # Microsoft Graph + access_token user. AI không tự xử lý được.
    raise ValueError(
        "Link SharePoint nội bộ cần đăng nhập Microsoft. Vui lòng dùng "
        "tính năng \"Đăng nhập Microsoft\" trên giao diện thay vì paste link, "
        "hoặc BE phải resolve qua Microsoft Graph trước khi gửi sang AI."
    )


# --- Dropbox ---

_DROPBOX_FOLDER_PATTERN = re.compile(r"dropbox\.com/sh/", re.IGNORECASE)


def _resolve_dropbox(url: str) -> ResolvedUrl | None:
    if "dropbox.com" not in url and "dropboxusercontent.com" not in url:
        return None
    if _DROPBOX_FOLDER_PATTERN.search(url):
        raise ValueError("Link Dropbox folder không hỗ trợ — vui lòng share file đơn lẻ.")

    if "?dl=0" in url:
        dl = url.replace("?dl=0", "?dl=1")
    elif "&dl=0" in url:
        dl = url.replace("&dl=0", "&dl=1")
    elif "dl=" not in url:
        sep = "&" if "?" in url else "?"
        dl = f"{url}{sep}dl=1"
    else:
        dl = url
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name or "dropbox_file"
    return ResolvedUrl(
        download_url=dl,
        kind=_kind_from_filename(name),
        suggested_filename=name,
        source="dropbox",
    )


# --- YouTube + các stream platform yt-dlp hỗ trợ ---

# yt-dlp hỗ trợ ~1500 site. Match theo HOST + path prefix (không substring),
# tránh trường hợp `x.com/` match nhầm `dropbox.com/`.
_STREAM_HOSTS = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be",
    "vimeo.com", "www.vimeo.com", "player.vimeo.com",
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com",
    "facebook.com", "www.facebook.com", "fb.watch",
    "instagram.com", "www.instagram.com",
    "twitter.com", "www.twitter.com", "x.com", "www.x.com",
    "twitch.tv", "www.twitch.tv",
    "dailymotion.com", "www.dailymotion.com",
    "bilibili.com", "www.bilibili.com",
})

_YT_HOSTS = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be",
})


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _is_youtube(url: str) -> bool:
    return _host(url) in _YT_HOSTS


def _is_stream(url: str) -> bool:
    return _host(url) in _STREAM_HOSTS


def _resolve_stream(url: str) -> ResolvedUrl | None:
    if not _is_stream(url):
        return None
    # Cả YouTube và stream khác đều route qua ingest_youtube pipeline (dùng
    # yt-dlp + transcript fallback Whisper). Pipeline tự fetch metadata
    # và transcript cho mọi site yt-dlp hỗ trợ.
    return ResolvedUrl(
        download_url=url,
        kind="youtube",
        suggested_filename="",
        source="youtube" if _is_youtube(url) else "stream",
    )


# --- Generic HTTP ---

def _resolve_generic(url: str) -> ResolvedUrl:
    """Fallback cho HTTP/HTTPS URL bất kỳ — đoán filename từ path."""
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    if not name or "." not in name:
        # URL không có extension rõ ràng — caller phải truyền filename trong metadata
        name = "download.bin"
    return ResolvedUrl(
        download_url=url,
        kind=_kind_from_filename(name),
        suggested_filename=name,
        source="generic",
    )


# --- Main entry ---

_RESOLVERS = (
    _resolve_stream,        # YouTube + Vimeo/TikTok/etc — không qua download flow
    _resolve_gdrive,        # Drive/Docs/Sheets/Slides — có thể raise cho folder/forms
    _resolve_onedrive,      # OneDrive personal
    _resolve_sharepoint,    # SharePoint anonymous OK; private raise rõ ràng
    _resolve_dropbox,       # có thể raise cho folder
)


def _strip_anchor(url: str) -> str:
    idx = url.find("#")
    return url[:idx] if idx >= 0 else url


def resolve(url: str) -> ResolvedUrl:
    """Resolve 1 URL public → ResolvedUrl với download_url + kind.

    Raise ValueError với message friendly cho:
    - URL không phải HTTP/HTTPS
    - Folder share (Drive/Dropbox/SharePoint)
    - Google Forms (không export được)
    - SharePoint private (cần Microsoft Graph)
    """
    url = url.strip()
    if not url:
        raise ValueError("URL trống")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"URL phải bắt đầu http:// hoặc https://: {url[:60]}")
    url = _strip_anchor(url)

    for fn in _RESOLVERS:
        result = fn(url)   # có thể raise ValueError với message rõ
        if result is not None:
            logger.info("resolve: %s → %s (%s)", url[:80], result.source, result.kind)
            return result

    return _resolve_generic(url)
