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

_GDRIVE_ID_PATTERNS = [
    re.compile(r"drive\.google\.com/file/d/([A-Za-z0-9_-]+)"),
    re.compile(r"drive\.google\.com/open\?id=([A-Za-z0-9_-]+)"),
    re.compile(r"docs\.google\.com/[^/]+/d/([A-Za-z0-9_-]+)"),
]


def _resolve_gdrive(url: str) -> ResolvedUrl | None:
    for pat in _GDRIVE_ID_PATTERNS:
        m = pat.search(url)
        if m:
            file_id = m.group(1)
            # Endpoint download trực tiếp; với file >100MB Google trả HTML
            # confirm token — caller tự handle (httpx detect content-type).
            dl = f"https://drive.google.com/uc?export=download&id={file_id}"
            return ResolvedUrl(
                download_url=dl,
                kind="document",      # default; sẽ correct sau khi tải về theo content-type
                suggested_filename=f"gdrive_{file_id}",
                source="gdrive",
            )
    return None


# --- OneDrive personal ---

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


# --- Dropbox ---

def _resolve_dropbox(url: str) -> ResolvedUrl | None:
    if "dropbox.com" not in url:
        return None
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


# --- YouTube ---

_YT_PATTERNS = (
    "youtube.com/watch", "youtube.com/playlist", "youtube.com/shorts",
    "youtu.be/", "music.youtube.com",
)


def _is_youtube(url: str) -> bool:
    return any(pat in url for pat in _YT_PATTERNS)


def _resolve_youtube(url: str) -> ResolvedUrl | None:
    if not _is_youtube(url):
        return None
    return ResolvedUrl(
        download_url=url,           # ingest_youtube tự fetch qua yt-dlp + transcript-api
        kind="youtube",
        suggested_filename="",      # video_pipeline tự lấy title
        source="youtube",
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
    _resolve_youtube,    # check trước vì YouTube không qua download flow
    _resolve_gdrive,
    _resolve_onedrive,
    _resolve_dropbox,
)


def resolve(url: str) -> ResolvedUrl:
    """Resolve 1 URL public → ResolvedUrl với download_url + kind.

    Raise ValueError nếu URL không phải HTTP/HTTPS.
    """
    url = url.strip()
    if not url:
        raise ValueError("URL trống")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"URL phải bắt đầu http:// hoặc https://: {url[:60]}")

    for fn in _RESOLVERS:
        result = fn(url)
        if result is not None:
            logger.info("resolve: %s → %s (%s)", url[:80], result.source, result.kind)
            return result

    return _resolve_generic(url)
