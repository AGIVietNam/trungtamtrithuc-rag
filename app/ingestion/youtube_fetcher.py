from __future__ import annotations

import json
import logging
import re
import subprocess
import time

from app import config

logger = logging.getLogger(__name__)


def fetch_youtube_via_whisper(url_or_id: str) -> dict:
    """Fallback: tải audio từ YouTube bằng yt-dlp rồi transcribe bằng Whisper.

    Dùng khi transcript API bị YouTube chặn hoặc video không có transcript sẵn.
    Trả về dict cùng shape với fetch_youtube_transcript().
    """
    import tempfile
    from pathlib import Path

    video_id = _parse_youtube_id(url_or_id)
    source_url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmp:
        out_tmpl = str(Path(tmp) / f"{video_id}.%(ext)s")
        # YouTube yêu cầu JS runtime để giải challenge; ưu tiên node (có sẵn trên hệ thống).
        import shutil as _shutil
        js_runtime = "node" if _shutil.which("node") else ("deno" if _shutil.which("deno") else None)
        js_flags = ["--js-runtimes", js_runtime] if js_runtime else []

        download_cmd = [
            "yt-dlp", *js_flags,
            "-f", "bestaudio/best",
            "-x", "--audio-format", "mp3", "--audio-quality", "5",
            "--no-warnings", "--no-playlist",
            "-o", out_tmpl,
            source_url,
        ]
        logger.info("yt-dlp: tải audio %s", video_id)
        result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr[:400]}")

        # lấy title riêng (lightweight, không cần JS runtime vì chỉ metadata)
        title = _fetch_title_oembed(source_url) or _fetch_title_ytdlp(video_id) or video_id

        # tìm file audio được tạo (.mp3 do --audio-format)
        audio_files = list(Path(tmp).glob(f"{video_id}.*"))
        audio = next((p for p in audio_files if p.suffix.lower() == ".mp3"), None)
        if audio is None:
            audio = audio_files[0] if audio_files else None
        if audio is None or not audio.exists():
            raise RuntimeError("yt-dlp didn't produce an audio file")

        from app.ingestion.video_transcriber import get_transcriber

        transcriber = get_transcriber()
        logger.info("Whisper: phiên âm %s (%.1f MB)", audio.name, audio.stat().st_size / 1e6)
        tr = transcriber.transcribe(audio)
        segments = tr.get("segments") or []

    return {
        "video_id": video_id,
        "title": title or video_id,
        "segments": segments,
        "source_url": source_url,
    }


def _parse_proxy_list() -> list[str]:
    """Tách YOUTUBE_PROXY_LIST thành list URL proxy, bỏ dòng trống."""
    raw = config.YOUTUBE_PROXY_LIST or ""
    items: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        u = chunk.strip()
        if u:
            items.append(u)
    return items


def _build_proxy_config_for(proxy_url: str | None):
    """Tạo GenericProxyConfig cho 1 URL cụ thể (dùng khi xoay vòng trong list)."""
    if not proxy_url:
        return None
    try:
        from youtube_transcript_api.proxies import GenericProxyConfig  # type: ignore
    except ImportError:
        return None
    return GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)


def _build_proxy_config():
    """Trả về ProxyConfig cho youtube_transcript_api dựa trên env, không dùng list xoay vòng.

    Dùng khi không có YOUTUBE_PROXY_LIST: ưu tiên Webshare rotating endpoint,
    fallback về proxy đơn GenericProxyConfig.
    """
    try:
        from youtube_transcript_api.proxies import (  # type: ignore
            GenericProxyConfig, WebshareProxyConfig,
        )
    except ImportError:
        return None

    if config.WEBSHARE_PROXY_USERNAME and config.WEBSHARE_PROXY_PASSWORD:
        return WebshareProxyConfig(
            proxy_username=config.WEBSHARE_PROXY_USERNAME,
            proxy_password=config.WEBSHARE_PROXY_PASSWORD,
        )
    if config.YOUTUBE_PROXY_HTTP or config.YOUTUBE_PROXY_HTTPS:
        return GenericProxyConfig(
            http_url=config.YOUTUBE_PROXY_HTTP or None,
            https_url=config.YOUTUBE_PROXY_HTTPS or config.YOUTUBE_PROXY_HTTP or None,
        )
    return None


# trạng thái xoay vòng proxy (process-wide)
import itertools  # noqa: E402
_proxy_cycle: "itertools.cycle[str] | None" = None


def _next_proxy_from_list() -> str | None:
    """Trả về proxy kế tiếp trong YOUTUBE_PROXY_LIST theo vòng tròn."""
    global _proxy_cycle
    proxies = _parse_proxy_list()
    if not proxies:
        return None
    if _proxy_cycle is None:
        _proxy_cycle = itertools.cycle(proxies)
    return next(_proxy_cycle)


def _parse_youtube_id(url_or_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    for pattern in [r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})"]:
        m = re.search(pattern, url_or_id)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot parse YouTube video id from: {url_or_id}")


def _fetch_title_oembed(url: str) -> str:
    import requests  # type: ignore

    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=10,
        )
        if resp.ok:
            return resp.json().get("title", "")
    except Exception:
        pass
    return ""


def _fetch_title_ytdlp(video_id: str) -> str:
    try:
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "--print", "title",
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def fetch_youtube_metadata(url_or_id: str) -> dict:
    """Lấy đầy đủ metadata video YouTube qua yt-dlp --dump-single-json.

    KHÔNG tải video, không phụ thuộc transcript. Miễn phí, ~0.5-1s.

    Trả về:
      {
        video_id, title, description, thumbnail, channel,
        duration_sec, categories, yt_tags, source_url
      }
    """
    video_id = _parse_youtube_id(url_or_id)
    source_url = f"https://www.youtube.com/watch?v={video_id}"

    result = subprocess.run(
        ["yt-dlp", "--skip-download", "--dump-single-json",
         "--no-warnings", "--no-playlist", source_url],
        capture_output=True, text=True, timeout=45,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {result.stderr[:300]}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"yt-dlp output không phải JSON hợp lệ: {exc}")

    return {
        "video_id": video_id,
        "title": (data.get("title") or "").strip(),
        "description": (data.get("description") or "").strip(),
        "thumbnail": data.get("thumbnail") or "",
        "channel": (data.get("channel") or data.get("uploader") or "").strip(),
        "duration_sec": int(data.get("duration") or 0),
        "categories": data.get("categories") or [],
        "yt_tags": data.get("tags") or [],
        "source_url": source_url,
    }


def fetch_playlist_video_ids(playlist_url: str) -> list[dict]:
    """Extract all video IDs and titles from a YouTube playlist using yt-dlp."""
    result = subprocess.run(
        [
            "yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s",
            "--no-warnings", playlist_url,
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp playlist error: {result.stderr[:500]}")

    videos = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        video_id = parts[0].strip()
        title = parts[1].strip() if len(parts) > 1 else video_id
        if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            videos.append({"video_id": video_id, "title": title})
    return videos


def fetch_playlist_info(playlist_url: str) -> dict:
    """Lấy metadata playlist + danh sách videos từ yt-dlp trong 1 call.

    Dùng `--flat-playlist -J` — không tải video, không cần LLM, ~1-2s.

    Return:
      {
        "playlist_id", "playlist_title", "playlist_description",
        "playlist_uploader", "playlist_thumbnail", "playlist_url",
        "video_count", "videos": [{"video_id", "title"}, ...]
      }
    """
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "-J", "--no-warnings", playlist_url],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp playlist error: {result.stderr[:500]}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"yt-dlp playlist JSON invalid: {exc}")

    videos: list[dict] = []
    for entry in data.get("entries") or []:
        vid = (entry.get("id") or "").strip()
        if vid and re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
            videos.append({
                "video_id": vid,
                "title": (entry.get("title") or vid).strip(),
            })

    thumbs = data.get("thumbnails") or []
    playlist_thumbnail = ""
    for t in reversed(thumbs):
        if t.get("url"):
            playlist_thumbnail = t["url"]
            break

    return {
        "playlist_id": (data.get("id") or "").strip(),
        "playlist_title": (data.get("title") or "").strip(),
        "playlist_description": (data.get("description") or "").strip(),
        "playlist_uploader": (data.get("uploader") or data.get("channel") or "").strip(),
        "playlist_thumbnail": playlist_thumbnail,
        "playlist_url": playlist_url,
        "video_count": len(videos),
        "videos": videos,
    }


def _is_ip_block_error(exc: BaseException) -> bool:
    """Nhận diện lỗi bị YouTube chặn IP (nhiều tên class tuỳ phiên bản)."""
    name = type(exc).__name__
    if name in {"IpBlocked", "RequestBlocked", "YouTubeRequestFailed"}:
        return True
    msg = str(exc).lower()
    return "blocking requests from your ip" in msg or "ipblocked" in msg


def fetch_youtube_transcript(
    url_or_id: str,
    langs: list[str] | None = None,
) -> dict:
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

    langs = langs or ["vi", "en"]
    video_id = _parse_youtube_id(url_or_id)
    source_url = f"https://www.youtube.com/watch?v={video_id}"

    proxy_list = _parse_proxy_list()
    max_retries = max(1, config.YOUTUBE_TRANSCRIPT_MAX_RETRIES)
    delay = config.YOUTUBE_TRANSCRIPT_RETRY_DELAY

    # Giới hạn thời gian cho 1 request qua proxy (socket-level) — nếu proxy chết
    # thì fail nhanh thay vì treo. Chỉ áp dụng khi dùng proxy.
    import socket
    original_timeout = socket.getdefaulttimeout()
    per_request_timeout = 15.0 if (proxy_list or _build_proxy_config()) else None

    raw = None
    last_err: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        # Mỗi lần thử dùng 1 proxy khác nhau nếu có YOUTUBE_PROXY_LIST,
        # ngược lại dùng cấu hình Webshare / proxy cố định, hoặc không proxy.
        if proxy_list:
            current_url = _next_proxy_from_list()
            proxy_config = _build_proxy_config_for(current_url)
            attempt_label = f"proxy={current_url}"
        else:
            proxy_config = _build_proxy_config()
            attempt_label = "no-rotation"

        try:
            if per_request_timeout:
                socket.setdefaulttimeout(per_request_timeout)
            try:
                api = YouTubeTranscriptApi(proxy_config=proxy_config) if proxy_config else YouTubeTranscriptApi()
                fetched = api.fetch(video_id, languages=langs)
                raw = fetched.to_raw_data()
            finally:
                socket.setdefaulttimeout(original_timeout)
            break
        except AttributeError:
            socket.setdefaulttimeout(original_timeout)
            raw = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            break
        except Exception as exc:  # bao gồm IpBlocked, RequestBlocked, mạng, timeout, proxy chết
            socket.setdefaulttimeout(original_timeout)
            last_err = exc
            # Với proxy list: mọi lỗi đều rotate (không chỉ IP block) vì có thể proxy chết.
            # Không có proxy list: chỉ retry khi IP bị chặn, còn lại raise ngay.
            if not proxy_list and not _is_ip_block_error(exc):
                raise
            if attempt >= max_retries:
                raise
            if not proxy_list and proxy_config is None:
                logger.warning(
                    "YouTube IP blocked (chưa cấu hình proxy). Set YOUTUBE_PROXY_LIST hoặc WEBSHARE_PROXY_* để xoay IP. Thử lại %d/%d sau %.1fs...",
                    attempt, max_retries, delay,
                )
            else:
                logger.info(
                    "IP bị chặn lần thử %d/%d (%s), đổi IP và retry sau %.1fs...",
                    attempt, max_retries, attempt_label, delay,
                )
            time.sleep(delay)
            delay = min(delay * 1.3, 8.0)
    if raw is None:
        raise last_err if last_err else RuntimeError("fetch transcript failed")

    segments = []
    for item in raw:
        start = float(item["start"])
        duration = float(item.get("duration", 0.0))
        segments.append({
            "start": round(start, 3),
            "end": round(start + duration, 3),
            "text": item["text"].strip(),
        })

    title = _fetch_title_oembed(source_url) or _fetch_title_ytdlp(video_id) or video_id

    return {
        "video_id": video_id,
        "title": title,
        "segments": segments,
        "source_url": source_url,
    }
