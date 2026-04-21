from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas import IngestResponse
from app.ingestion.doc_pipeline import ingest_document, ensure_collections
from app.ingestion.metadata_generator import generate_document_metadata

logger = logging.getLogger(__name__)

router = APIRouter()

# Ensure Qdrant collections exist on first import
try:
    ensure_collections()
except Exception as exc:
    logger.warning("Could not ensure collections at startup: %s", exc)

try:
    from app.ingestion.video_pipeline import ensure_collections as ensure_video_collections
    ensure_video_collections()
except Exception as exc:
    logger.warning("Could not ensure video collections at startup: %s", exc)


# ---------------------------------------------------------------------------
# Ingest — tài liệu (PDF, DOCX, TXT, MD)
# ---------------------------------------------------------------------------

@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    collection: str = Form(default="ttt_documents"),
    title: str = Form(default=""),
    domain: str = Form(default=""),
    description: str = Form(default=""),
    tags: str = Form(default=""),
    url: str = Form(default=""),
) -> IngestResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx"):
        return IngestResponse(
            status="error",
            chunks_added=0,
            message=f"Định dạng '{suffix}' không được hỗ trợ. Chỉ chấp nhận: PDF, DOCX, TXT, MD, XLSX.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Build metadata from form fields
    meta = {}
    if title.strip():
        meta["title"] = title.strip()
    if domain.strip():
        meta["domain"] = domain.strip()
    if description.strip():
        meta["description"] = description.strip()
    if tags.strip():
        meta["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if url.strip():
        meta["url"] = url.strip()

    try:
        result = ingest_document(
            file_path=tmp_path,
            original_name=title.strip() or file.filename,
            metadata=meta if meta else None,
        )
        return IngestResponse(
            status="ok",
            chunks_added=result.num_chunks,
            message=f"Nạp thành công '{file.filename}': {result.num_chunks} đoạn từ {result.num_pages} trang.",
        )
    except Exception as exc:
        logger.exception("Ingest error for %s: %s", file.filename, exc)
        return IngestResponse(
            status="error",
            chunks_added=0,
            message=f"Lỗi khi nạp '{file.filename}': {exc}",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Preview — AI auto-generate metadata (title/description/domain/tags).
# Chỉ parse file và gọi Haiku, KHÔNG upsert Qdrant. FE dùng để prefill form.
# ---------------------------------------------------------------------------

_DOC_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx"}


@router.post("/file/preview")
async def preview_file_metadata(file: UploadFile = File(...)) -> dict:
    """Parse file + AI gen metadata. Trả JSON cho FE prefill form.

    KHÔNG lưu vào Qdrant. User có thể review/sửa trước khi submit thực sự.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _DOC_SUFFIXES:
        return {
            "status": "error",
            "message": f"Định dạng '{suffix}' không được hỗ trợ.",
            "metadata": None,
        }

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from app.ingestion.doc_parser import parse

        parsed = parse(Path(tmp_path))
        raw_content = parsed.get("content", "")

        # Flatten 5 trang đầu thành text sample (đủ ngữ cảnh cho metadata)
        if isinstance(raw_content, list):
            pages = [p.get("text", "") for p in raw_content[:5] if p.get("text", "").strip()]
            text_sample = "\n\n".join(pages)
        else:
            text_sample = str(raw_content)

        meta = generate_document_metadata(
            text_sample=text_sample,
            filename=file.filename,
        )

        if meta is None:
            return {
                "status": "partial",
                "message": "Không gen được metadata (file quá ngắn hoặc LLM lỗi). Vui lòng nhập thủ công.",
                "metadata": None,
            }

        return {
            "status": "ok",
            "message": "Metadata đã gen thành công.",
            "metadata": meta.model_dump(),
        }
    except Exception as exc:
        logger.exception("Preview metadata error for %s: %s", file.filename, exc)
        return {
            "status": "error",
            "message": f"Lỗi phân tích file: {exc}",
            "metadata": None,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Ingest — video file upload (MP4, MKV, AVI, MOV)
# ---------------------------------------------------------------------------

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


def _build_metadata_dict(
    title: str,
    domain: str,
    description: str,
    tags: str,
    url: str,
) -> dict:
    """Tạo metadata dict từ Form fields, bỏ field trống."""
    meta: dict = {}
    if title.strip():
        meta["title"] = title.strip()
    if domain.strip():
        meta["domain"] = domain.strip()
    if description.strip():
        meta["description"] = description.strip()
    if tags.strip():
        meta["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if url.strip():
        meta["url"] = url.strip()
    return meta


@router.post("/video/file", response_model=IngestResponse)
async def ingest_video_file(
    file: UploadFile = File(...),
    collection: str = Form(default="ttt_videos"),
    title: str = Form(default=""),
    domain: str = Form(default=""),
    description: str = Form(default=""),
    tags: str = Form(default=""),
    url: str = Form(default=""),
) -> IngestResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        return IngestResponse(
            status="error",
            chunks_added=0,
            message=f"Định dạng '{suffix}' không được hỗ trợ. Chỉ chấp nhận: MP4, MKV, AVI, MOV.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    meta = _build_metadata_dict(title, domain, description, tags, url)

    try:
        from app.ingestion.video_pipeline import ingest_video_file as _ingest_video
        result = _ingest_video(
            local_path=tmp_path,
            original_name=file.filename,
            metadata=meta if meta else None,
        )
        return IngestResponse(
            status="ok",
            chunks_added=result.num_chunks,
            message=f"Phiên âm thành công '{file.filename}': {result.num_chunks} đoạn.",
        )
    except ImportError:
        logger.exception("Whisper not installed")
        return IngestResponse(
            status="error",
            chunks_added=0,
            message="openai-whisper chưa được cài đặt. Chạy: pip install openai-whisper",
        )
    except Exception as exc:
        logger.exception("Video ingest error for %s: %s", file.filename, exc)
        return IngestResponse(
            status="error",
            chunks_added=0,
            message=f"Lỗi khi phiên âm '{file.filename}': {exc}",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# --- Preview: video file → Whisper transcribe → AI gen metadata ---

@router.post("/video/file/preview")
async def preview_video_metadata(file: UploadFile = File(...)) -> dict:
    """Phiên âm video + AI gen metadata. Trả JSON cho FE prefill form.

    KHÔNG upsert Qdrant. Có thể mất 30s-2 phút do Whisper.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        return {
            "status": "error",
            "message": f"Định dạng '{suffix}' không được hỗ trợ.",
            "metadata": None,
        }

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from app.ingestion.video_transcriber import get_transcriber

        transcriber = get_transcriber()
        result = transcriber.transcribe(Path(tmp_path))
        segments = result.get("segments", [])

        # Trích text thuần từ segments (bỏ timestamp), cap ~6000 chars
        parts = [s.get("text", "").strip() for s in segments if s.get("text", "").strip()]
        text_sample = " ".join(parts)

        meta = generate_document_metadata(
            text_sample=text_sample,
            filename=file.filename,
        )

        if meta is None:
            return {
                "status": "partial",
                "message": "Không gen được metadata (transcript quá ngắn hoặc LLM lỗi).",
                "metadata": None,
            }

        return {
            "status": "ok",
            "message": "Metadata đã gen từ transcript.",
            "metadata": meta.model_dump(),
        }
    except Exception as exc:
        logger.exception("Preview video metadata error for %s: %s", file.filename, exc)
        return {
            "status": "error",
            "message": f"Lỗi: {exc}",
            "metadata": None,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Ingest — YouTube URL
# ---------------------------------------------------------------------------

def _is_playlist_url(url: str) -> bool:
    """Detect if URL is a YouTube playlist (not a single video with list param)."""
    import re
    # Pure playlist URL: youtube.com/playlist?list=...
    if re.search(r"youtube\.com/playlist\?", url):
        return True
    return False


@router.post("/youtube", response_model=IngestResponse)
async def ingest_youtube(
    url: str,
    collection: str = "ttt_videos",
    title: str = Form(default=""),
    domain: str = Form(default=""),
    description: str = Form(default=""),
    tags: str = Form(default=""),
) -> IngestResponse:
    if not url or not url.strip():
        return IngestResponse(
            status="error",
            chunks_added=0,
            message="Vui lòng nhập URL YouTube.",
        )

    clean_url = url.strip()
    meta = _build_metadata_dict(title, domain, description, tags, url=clean_url)

    # Auto-detect playlist URLs and route to playlist handler
    if _is_playlist_url(clean_url):
        try:
            from app.ingestion.video_pipeline import ingest_youtube_playlist as _ingest_pl
            # Playlist: chỉ truyền domain (tags/title/description per-video khác nhau)
            pl_meta = {"domain": meta["domain"]} if meta.get("domain") else None
            data = _ingest_pl(playlist_url=clean_url, metadata=pl_meta)
            results = data["results"]
            playlist_info = data.get("playlist_info") or {}
            total_ok = sum(1 for r in results if r["status"] == "ok")
            total_chunks = sum(r["chunks_added"] for r in results)
            failed = [r for r in results if r["status"] == "error"]
            pl_title = playlist_info.get("playlist_title") or "playlist"
            msg = (
                f"Playlist '{pl_title}': {total_ok}/{len(results)} video thành công, "
                f"tổng {total_chunks} đoạn."
            )
            if failed:
                msg += f" ({len(failed)} video lỗi)"
            return IngestResponse(
                status="ok" if total_ok > 0 else "error",
                chunks_added=total_chunks,
                message=msg,
            )
        except Exception as exc:
            logger.exception("Playlist ingest error for %s: %s", clean_url, exc)
            return IngestResponse(
                status="error",
                chunks_added=0,
                message=f"Lỗi khi nạp playlist: {exc}",
            )

    try:
        from app.ingestion.video_pipeline import ingest_youtube as _ingest_yt
        result = _ingest_yt(url=clean_url, metadata=meta if meta else None)
        return IngestResponse(
            status="ok",
            chunks_added=result.num_chunks,
            message=f"Nạp thành công '{result.source_name}': {result.num_chunks} đoạn.",
        )
    except ValueError as exc:
        return IngestResponse(
            status="error",
            chunks_added=0,
            message=f"URL không hợp lệ: {exc}",
        )
    except Exception as exc:
        logger.exception("YouTube ingest error for %s: %s", clean_url, exc)
        return IngestResponse(
            status="error",
            chunks_added=0,
            message=f"Lỗi khi nạp YouTube: {exc}",
        )


# --- Preview: YouTube URL → yt-dlp metadata + transcript → AI classify domain/tags ---

@router.post("/youtube/preview")
async def preview_youtube_metadata(url: str) -> dict:
    """Lấy metadata YouTube có sẵn (title/description/thumbnail) qua yt-dlp +
    AI gen domain/tags từ transcript. KHÔNG upsert Qdrant.

    Playlist URL → trả status=skip (playlist không preview được từng video).
    """
    if not url or not url.strip():
        return {"status": "error", "message": "Vui lòng nhập URL.", "metadata": None}

    clean_url = url.strip()
    if _is_playlist_url(clean_url):
        try:
            from app.ingestion.youtube_fetcher import fetch_playlist_info
            info = fetch_playlist_info(clean_url)
        except Exception as exc:
            logger.warning("Playlist preview fail cho %s: %s", clean_url, exc)
            return {
                "status": "error",
                "message": f"Không lấy được metadata playlist: {exc}",
                "metadata": None,
            }
        return {
            "status": "ok",
            "message": (
                f"Playlist '{info.get('playlist_title') or '?'}' — "
                f"{info.get('video_count', 0)} video. Metadata playlist sẽ gắn vào mọi chunk con."
            ),
            "metadata": {
                "is_playlist": True,
                "playlist_id": info.get("playlist_id", ""),
                "playlist_title": info.get("playlist_title", ""),
                "playlist_description": info.get("playlist_description", ""),
                "playlist_uploader": info.get("playlist_uploader", ""),
                "playlist_thumbnail": info.get("playlist_thumbnail", ""),
                "video_count": info.get("video_count", 0),
                "url": info.get("playlist_url", clean_url),
            },
        }

    try:
        from app.ingestion.youtube_fetcher import (
            fetch_youtube_metadata, fetch_youtube_transcript,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Lỗi import: {exc}", "metadata": None}

    # 1) yt-dlp metadata (miễn phí, nhanh)
    try:
        yt = fetch_youtube_metadata(clean_url)
    except Exception as exc:
        logger.warning("yt-dlp metadata fail cho %s: %s", clean_url, exc)
        return {
            "status": "error",
            "message": f"Không lấy được metadata YouTube: {exc}",
            "metadata": None,
        }

    # 2) Transcript (best-effort — có thể fail do IP block / video không có transcript)
    transcript_text = ""
    try:
        data = fetch_youtube_transcript(clean_url)
        parts = [s.get("text", "").strip() for s in data.get("segments", []) if s.get("text")]
        transcript_text = " ".join(parts)
    except Exception as exc:
        logger.info("Không lấy được transcript cho AI classify: %s", type(exc).__name__)

    # 3) AI classify domain + tags (dùng title + description + transcript làm input)
    ai_parts: list[str] = []
    if yt.get("title"):
        ai_parts.append(f"Tiêu đề YouTube: {yt['title']}")
    if yt.get("description"):
        ai_parts.append(f"Mô tả YouTube: {yt['description'][:800]}")
    if transcript_text:
        ai_parts.append(f"Nội dung transcript: {transcript_text[:3000]}")
    ai_input = "\n\n".join(ai_parts)

    ai_meta = generate_document_metadata(
        text_sample=ai_input,
        filename=yt.get("title") or "youtube-video",
    )

    # 4) Gộp: title/description từ YouTube, domain/tags từ AI
    out = {
        "title": yt.get("title", ""),
        "description": (yt.get("description") or "")[:800].strip(),
        "url": yt.get("source_url", clean_url),
        "thumbnail": yt.get("thumbnail", ""),
        "channel": yt.get("channel", ""),
        "duration_sec": yt.get("duration_sec", 0),
        "domain": ai_meta.domain if ai_meta else "mặc định",
        "tags": ai_meta.tags if ai_meta else [],
        "sources": {
            "title": "youtube",
            "description": "youtube",
            "domain": "ai" if ai_meta else "fallback",
            "tags": "ai" if ai_meta else "empty",
        },
    }
    return {
        "status": "ok",
        "message": (
            "Lấy metadata YouTube + AI classify thành công."
            if ai_meta else
            "Đã lấy metadata YouTube, AI classify thất bại (transcript không khả dụng)."
        ),
        "metadata": out,
    }


# ---------------------------------------------------------------------------
# Ingest — YouTube Playlist
# ---------------------------------------------------------------------------

@router.post("/youtube-playlist")
async def ingest_youtube_playlist(url: str) -> dict:
    if not url or not url.strip():
        return {"status": "error", "message": "Vui lòng nhập URL playlist.", "results": []}

    try:
        from app.ingestion.video_pipeline import ingest_youtube_playlist as _ingest_pl
        data = _ingest_pl(playlist_url=url.strip())
        results = data["results"]
        playlist_info = data.get("playlist_info") or {}
        total_ok = sum(1 for r in results if r["status"] == "ok")
        total_chunks = sum(r["chunks_added"] for r in results)
        pl_title = playlist_info.get("playlist_title") or "playlist"
        return {
            "status": "ok",
            "message": (
                f"Playlist '{pl_title}': {total_ok}/{len(results)} video thành công, "
                f"tổng {total_chunks} đoạn."
            ),
            "playlist_info": playlist_info,
            "total_videos": len(results),
            "success_count": total_ok,
            "total_chunks": total_chunks,
            "results": results,
        }
    except RuntimeError as exc:
        return {"status": "error", "message": f"Lỗi playlist: {exc}", "results": []}
    except Exception as exc:
        logger.exception("Playlist ingest error for %s: %s", url, exc)
        return {"status": "error", "message": f"Lỗi khi nạp playlist: {exc}", "results": []}
