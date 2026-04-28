from __future__ import annotations

import hashlib
import logging
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from app.schemas import (
    BatchSubmitResponse,
    BatchStatusResponse,
    FromUrlsRequest,
    IngestResponse,
    JobStatusResponse,
    JobSubmitResponse,
)
from app.ingestion.doc_pipeline import ingest_document
from app.ingestion.metadata_generator import generate_document_metadata
from app.core.qdrant_store import DOMAINS, PERSONA_TO_DOMAIN
from app.core.job_runner import get_runner
from app.core.job_store import JobType, get_store
from app.config import (
    INGEST_MAX_BATCH_SIZE,
    MAX_UPLOAD_MB,
    PREVIEW_MAX_PAGES,
    PREVIEW_VIDEO_CLIP_SEC,
    UPLOAD_DIR,
    UPLOAD_STREAM_CHUNK,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Streaming upload helper — tránh OOM với file lớn
# ---------------------------------------------------------------------------

_MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


async def _stream_to_disk(upload: UploadFile, dest: Path) -> tuple[int, str]:
    """Đọc UploadFile theo chunk, ghi vào dest, đồng thời tính sha256.

    Reject sớm nếu vượt MAX_UPLOAD_MB.
    Trả về (size_bytes, sha256_hex_8chars) — sha256 dùng làm cache key giống
    flow cũ (`_content_hash`) để giữ đặt tên file persistent.
    """
    h = hashlib.sha256()
    written = 0
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(UPLOAD_STREAM_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > _MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise ValueError(
                    f"File quá lớn (>{MAX_UPLOAD_MB}MB). Tăng MAX_UPLOAD_MB nếu cần."
                )
            h.update(chunk)
            f.write(chunk)
    return written, h.hexdigest()[:8]


def _safe_filename(name: str) -> str:
    """Sanitize filename: strip path components, replace unsafe chars."""
    name = Path(name).name
    name = re.sub(r'[^\w.\-]', '_', name, flags=re.UNICODE)
    name = re.sub(r'_+', '_', name).strip('._')
    return name or "document"


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


# Lifespan (main.py:_lifespan) gọi registry.ensure_all() tạo đủ 20 collection —
# không ensure ở module import nữa để tránh ghi đè thứ tự startup.


# Primary = slug (khớp NestJS). Legacy = persona VN map về slug.
VALID_SLUGS: frozenset[str] = frozenset(DOMAINS)
VALID_PERSONAS: frozenset[str] = frozenset(PERSONA_TO_DOMAIN.keys())


def _validate_domain(domain: str) -> str | None:
    """Trả message lỗi nếu domain không hợp lệ, None nếu OK.

    Nhận slug ("thiet_ke") hoặc persona VN ("thiết kế") — cả hai đều valid.
    """
    d = (domain or "").strip()
    if not d:
        return "Lĩnh vực (domain) là bắt buộc — vui lòng chọn lĩnh vực trước khi upload."
    if d not in VALID_SLUGS and d not in VALID_PERSONAS:
        return (
            f"Lĩnh vực '{d}' không hợp lệ. "
            f"Chọn 1 trong: {sorted(VALID_SLUGS)}"
        )
    return None


# ---------------------------------------------------------------------------
# Ingest — tài liệu (PDF, DOCX, TXT, MD)
# ---------------------------------------------------------------------------

@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    domain: str = Form(default=""),
    description: str = Form(default=""),
    tags: str = Form(default=""),
    url: str = Form(default=""),
) -> IngestResponse:
    t0 = time.perf_counter()
    domain_err = _validate_domain(domain)
    if domain_err:
        return IngestResponse(status="error", chunks_added=0, message=domain_err)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md", ".xlsx"):
        return IngestResponse(
            status="error",
            chunks_added=0,
            message=f"Định dạng '{suffix}' không được hỗ trợ. Chỉ chấp nhận: PDF, DOCX, PPTX, TXT, MD, XLSX.",
        )

    # Stream-to-disk: tránh load cả file vào RAM. Hash tính cùng lúc với write.
    # Dùng tmp name (chưa biết hash) → rename sau khi có hash.
    tmp_path = UPLOAD_DIR / f".uploading_{uuid.uuid4().hex}{suffix}"
    try:
        _, file_hash = await _stream_to_disk(file, tmp_path)
    except ValueError as exc:
        return IngestResponse(status="error", chunks_added=0, message=str(exc))

    original_name = title.strip() or file.filename
    safe_name = f"{file_hash}_{_safe_filename(original_name)}"
    if not safe_name.lower().endswith(suffix):
        safe_name += suffix
    persistent_path = UPLOAD_DIR / safe_name
    tmp_path.replace(persistent_path)

    # Build metadata from form fields
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
    else:
        # No external URL — serve file locally so page links work
        meta["source_url"] = f"/files/{safe_name}"

    try:
        t_upload = time.perf_counter()

        result = ingest_document(
            file_path=str(persistent_path),
            original_name=original_name,
            metadata=meta if meta else None,
        )
        t_ingest = time.perf_counter()
        logger.info(
            "POST /api/ingest/file steps (%s): upload=%.3fs ingest=%.3fs total=%.3fs (chunks=%d, pages=%d)",
            file.filename,
            t_upload - t0,
            t_ingest - t_upload,
            t_ingest - t0,
            result.num_chunks, result.num_pages,
        )
        return IngestResponse(
            status="ok",
            chunks_added=result.num_chunks,
            message=f"Nạp thành công '{file.filename}': {result.num_chunks} đoạn từ {result.num_pages} trang.",
        )
    except Exception as exc:
        persistent_path.unlink(missing_ok=True)
        logger.exception("Ingest error for %s: %s", file.filename, exc)
        logger.info(
            "POST /api/ingest/file FAILED (%s): upload=%.3fs ingest=%.3fs total=%.3fs",
            file.filename,
            t_upload - t0,
            time.perf_counter() - t_upload,
            time.perf_counter() - t0,
        )
        return IngestResponse(
            status="error",
            chunks_added=0,
            message=f"Lỗi khi nạp '{file.filename}': {exc}",
        )


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
    t0 = time.perf_counter()
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _DOC_SUFFIXES:
        return {
            "status": "error",
            "message": f"Định dạng '{suffix}' không được hỗ trợ.",
            "metadata": None,
        }

    tmp_path = str(Path(tempfile.mktemp(suffix=suffix)))
    try:
        await _stream_to_disk(file, Path(tmp_path))
    except ValueError as exc:
        Path(tmp_path).unlink(missing_ok=True)
        return {"status": "error", "message": str(exc), "metadata": None}
    t_upload = time.perf_counter()

    try:
        from app.ingestion.doc_parser import parse

        # Preview: parse N trang đầu thay vì cả file. Cho file 500 trang giảm
        # Docling time từ ~10 phút xuống ~5 giây.
        parsed = parse(Path(tmp_path), max_pages=PREVIEW_MAX_PAGES)
        raw_content = parsed.get("content", "")

        if isinstance(raw_content, list):
            pages = [p.get("text", "") for p in raw_content if p.get("text", "").strip()]
            text_sample = "\n\n".join(pages)
        else:
            text_sample = str(raw_content)
        t_parse = time.perf_counter()

        meta = generate_document_metadata(
            text_sample=text_sample,
            filename=file.filename,
        )
        t_ai = time.perf_counter()

        logger.info(
            "POST /api/ingest/file/preview steps (%s): upload=%.3fs parse=%.3fs ai_meta=%.3fs total=%.3fs (sample=%d chars, ai_ok=%s)",
            file.filename,
            t_upload - t0, t_parse - t_upload, t_ai - t_parse, t_ai - t0,
            len(text_sample), meta is not None,
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
        logger.info(
            "POST /api/ingest/file/preview FAILED (%s): upload=%.3fs total=%.3fs",
            file.filename, t_upload - t0, time.perf_counter() - t0,
        )
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
    title: str = Form(default=""),
    domain: str = Form(default=""),
    description: str = Form(default=""),
    tags: str = Form(default=""),
    url: str = Form(default=""),
) -> IngestResponse:
    t0 = time.perf_counter()
    domain_err = _validate_domain(domain)
    if domain_err:
        return IngestResponse(status="error", chunks_added=0, message=domain_err)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        return IngestResponse(
            status="error",
            chunks_added=0,
            message=f"Định dạng '{suffix}' không được hỗ trợ. Chỉ chấp nhận: MP4, MKV, AVI, MOV.",
        )

    tmp_path = str(Path(tempfile.mktemp(suffix=suffix)))
    try:
        await _stream_to_disk(file, Path(tmp_path))
    except ValueError as exc:
        Path(tmp_path).unlink(missing_ok=True)
        return IngestResponse(status="error", chunks_added=0, message=str(exc))
    t_upload = time.perf_counter()

    meta = _build_metadata_dict(title, domain, description, tags, url)


    try:
        from app.ingestion.video_pipeline import ingest_video_file as _ingest_video
        result = _ingest_video(
            local_path=tmp_path,
            original_name=file.filename,
            metadata=meta if meta else None,
        )
        t_ingest = time.perf_counter()
        logger.info(
            "POST /api/ingest/video/file steps (%s): upload=%.3fs transcribe+embed=%.3fs total=%.3fs (chunks=%d)",
            file.filename,
            t_upload - t0,
            t_ingest - t_upload,
            t_ingest - t0,
            result.num_chunks,
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
        logger.info(
            "POST /api/ingest/video/file FAILED (%s): upload=%.3fs transcribe=%.3fs total=%.3fs",
            file.filename,
            t_upload - t0,
            time.perf_counter() - t_upload,
            time.perf_counter() - t0,
        )
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
    t0 = time.perf_counter()
    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        return {
            "status": "error",
            "message": f"Định dạng '{suffix}' không được hỗ trợ.",
            "metadata": None,
        }

    tmp_path = str(Path(tempfile.mktemp(suffix=suffix)))
    try:
        await _stream_to_disk(file, Path(tmp_path))
    except ValueError as exc:
        Path(tmp_path).unlink(missing_ok=True)
        return {"status": "error", "message": str(exc), "metadata": None}
    t_upload = time.perf_counter()

    try:
        from app.ingestion.video_transcriber import get_transcriber

        transcriber = get_transcriber()
        # Preview: chỉ transcribe N giây đầu (mặc định 180s = 3 phút). Cắt clip
        # bằng ffmpeg stream-copy nên rất nhanh dù video gốc 2 tiếng.
        result = transcriber.transcribe(
            Path(tmp_path),
            clip_duration_sec=PREVIEW_VIDEO_CLIP_SEC,
        )
        segments = result.get("segments", [])

        # Trích text thuần từ segments (bỏ timestamp), cap ~6000 chars
        parts = [s.get("text", "").strip() for s in segments if s.get("text", "").strip()]
        text_sample = " ".join(parts)
        t_transcribe = time.perf_counter()

        meta = generate_document_metadata(
            text_sample=text_sample,
            filename=file.filename,
        )
        t_ai = time.perf_counter()

        logger.info(
            "POST /api/ingest/video/file/preview steps (%s): upload=%.3fs transcribe=%.3fs ai_meta=%.3fs total=%.3fs (segments=%d, sample=%d chars)",
            file.filename,
            t_upload - t0, t_transcribe - t_upload,
            t_ai - t_transcribe, t_ai - t0,
            len(segments), len(text_sample),
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
        logger.info(
            "POST /api/ingest/video/file/preview FAILED (%s): upload=%.3fs total=%.3fs",
            file.filename, t_upload - t0, time.perf_counter() - t0,
        )
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
    title: str = Form(default=""),
    domain: str = Form(default=""),
    description: str = Form(default=""),
    tags: str = Form(default=""),
) -> IngestResponse:
    t0 = time.perf_counter()
    if not url or not url.strip():
        return IngestResponse(
            status="error",
            chunks_added=0,
            message="Vui lòng nhập URL YouTube.",
        )
    domain_err = _validate_domain(domain)
    if domain_err:
        return IngestResponse(status="error", chunks_added=0, message=domain_err)

    clean_url = url.strip()
    meta = _build_metadata_dict(title, domain, description, tags, url=clean_url)

    # Auto-detect playlist URLs and route to playlist handler
    if _is_playlist_url(clean_url):
        try:
            from app.ingestion.video_pipeline import ingest_youtube_playlist as _ingest_pl
            # Playlist: chỉ truyền domain (tags/title/description per-video khác nhau)
            pl_meta = {"domain": meta["domain"]} if meta.get("domain") else None
            data = _ingest_pl(playlist_url=clean_url, metadata=pl_meta)
            t_done = time.perf_counter()
            results = data["results"]
            playlist_info = data.get("playlist_info") or {}
            total_ok = sum(1 for r in results if r["status"] == "ok")
            total_chunks = sum(r["chunks_added"] for r in results)
            failed = [r for r in results if r["status"] == "error"]
            pl_title = playlist_info.get("playlist_title") or "playlist"
            logger.info(
                "POST /api/ingest/youtube (playlist) steps (%s): fetch+ingest_all=%.3fs total=%.3fs (videos=%d, ok=%d, chunks=%d)",
                clean_url, t_done - t0, t_done - t0,
                len(results), total_ok, total_chunks,
            )
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
            logger.info(
                "POST /api/ingest/youtube (playlist) FAILED (%s): total=%.3fs",
                clean_url, time.perf_counter() - t0,
            )
            return IngestResponse(
                status="error",
                chunks_added=0,
                message=f"Lỗi khi nạp playlist: {exc}",
            )

    try:
        from app.ingestion.video_pipeline import ingest_youtube as _ingest_yt
        result = _ingest_yt(url=clean_url, metadata=meta if meta else None)
        t_done = time.perf_counter()
        logger.info(
            "POST /api/ingest/youtube steps (%s): fetch+transcribe+embed=%.3fs total=%.3fs (chunks=%d)",
            clean_url, t_done - t0, t_done - t0, result.num_chunks,
        )
        return IngestResponse(
            status="ok",
            chunks_added=result.num_chunks,
            message=f"Nạp thành công '{result.source_name}': {result.num_chunks} đoạn.",
        )
    except ValueError as exc:
        logger.info(
            "POST /api/ingest/youtube invalid URL (%s): total=%.3fs",
            clean_url, time.perf_counter() - t0,
        )
        return IngestResponse(
            status="error",
            chunks_added=0,
            message=f"URL không hợp lệ: {exc}",
        )
    except Exception as exc:
        logger.exception("YouTube ingest error for %s: %s", clean_url, exc)
        logger.info(
            "POST /api/ingest/youtube FAILED (%s): total=%.3fs",
            clean_url, time.perf_counter() - t0,
        )
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
    t0 = time.perf_counter()
    if not url or not url.strip():
        return {"status": "error", "message": "Vui lòng nhập URL.", "metadata": None}

    clean_url = url.strip()
    if _is_playlist_url(clean_url):
        try:
            from app.ingestion.youtube_fetcher import fetch_playlist_info
            info = fetch_playlist_info(clean_url)
        except Exception as exc:
            logger.warning("Playlist preview fail cho %s: %s", clean_url, exc)
            logger.info(
                "POST /api/ingest/youtube/preview (playlist) FAILED (%s): total=%.3fs",
                clean_url, time.perf_counter() - t0,
            )
            return {
                "status": "error",
                "message": f"Không lấy được metadata playlist: {exc}",
                "metadata": None,
            }
        logger.info(
            "POST /api/ingest/youtube/preview (playlist) steps (%s): fetch_info=%.3fs total=%.3fs",
            clean_url, time.perf_counter() - t0, time.perf_counter() - t0,
        )
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
        logger.info(
            "POST /api/ingest/youtube/preview FAILED (%s) at ytdlp: total=%.3fs",
            clean_url, time.perf_counter() - t0,
        )
        return {
            "status": "error",
            "message": f"Không lấy được metadata YouTube: {exc}",
            "metadata": None,
        }
    t_ytdlp = time.perf_counter()

    # 2) Transcript (best-effort — có thể fail do IP block / video không có transcript)
    transcript_text = ""
    transcript_ok = False
    try:
        data = fetch_youtube_transcript(clean_url)
        parts = [s.get("text", "").strip() for s in data.get("segments", []) if s.get("text")]
        transcript_text = " ".join(parts)
        transcript_ok = bool(transcript_text)
    except Exception as exc:
        logger.info("Không lấy được transcript cho AI classify: %s", type(exc).__name__)
    t_transcript = time.perf_counter()

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
    t_ai = time.perf_counter()
    logger.info(
        "POST /api/ingest/youtube/preview steps (%s): ytdlp=%.3fs transcript=%.3fs%s ai_classify=%.3fs total=%.3fs (ai_ok=%s)",
        clean_url,
        t_ytdlp - t0,
        t_transcript - t_ytdlp, "" if transcript_ok else " [no-transcript]",
        t_ai - t_transcript, t_ai - t0,
        ai_meta is not None,
    )

    # 4) Gộp: title/description từ YouTube, domain/tags từ AI
    out = {
        "title": yt.get("title", ""),
        "description": (yt.get("description") or "")[:800].strip(),
        "url": yt.get("source_url", clean_url),
        "thumbnail": yt.get("thumbnail", ""),
        "channel": yt.get("channel", ""),
        "duration_sec": yt.get("duration_sec", 0),
        # AI fail → trả chuỗi rỗng để FE buộc user chọn domain thủ công
        # (form upload validate domain bắt buộc, không nhận "mặc định").
        "domain": ai_meta.domain if ai_meta else "",
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
    t0 = time.perf_counter()
    if not url or not url.strip():
        return {"status": "error", "message": "Vui lòng nhập URL playlist.", "results": []}

    try:
        from app.ingestion.video_pipeline import ingest_youtube_playlist as _ingest_pl
        data = _ingest_pl(playlist_url=url.strip())
        t_done = time.perf_counter()
        results = data["results"]
        playlist_info = data.get("playlist_info") or {}
        total_ok = sum(1 for r in results if r["status"] == "ok")
        total_chunks = sum(r["chunks_added"] for r in results)
        pl_title = playlist_info.get("playlist_title") or "playlist"
        logger.info(
            "POST /api/ingest/youtube-playlist steps (%s): fetch+ingest_all=%.3fs total=%.3fs (videos=%d, ok=%d, chunks=%d)",
            url.strip(), t_done - t0, t_done - t0,
            len(results), total_ok, total_chunks,
        )
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
        logger.info(
            "POST /api/ingest/youtube-playlist FAILED (%s): total=%.3fs",
            url, time.perf_counter() - t0,
        )
        return {"status": "error", "message": f"Lỗi playlist: {exc}", "results": []}
    except Exception as exc:
        logger.exception("Playlist ingest error for %s: %s", url, exc)
        logger.info(
            "POST /api/ingest/youtube-playlist FAILED (%s): total=%.3fs",
            url, time.perf_counter() - t0,
        )
        return {"status": "error", "message": f"Lỗi khi nạp playlist: {exc}", "results": []}


# ===========================================================================
# Async ingest — return job_id ngay, runner xử lý trong nền.
# Endpoint cũ (/file, /video/file, /youtube) giữ nguyên cho client cũ.
# ===========================================================================

DOC_SUFFIXES_FULL: frozenset[str] = frozenset(
    {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md", ".xlsx"}
)


async def _persist_upload(file: UploadFile, suffix: str) -> tuple[Path, str]:
    """Stream UploadFile vào UPLOAD_DIR với tên `{hash}_{name}`.

    Trả về (persistent_path, safe_name). Raise ValueError nếu vượt size cap.
    """
    tmp_path = UPLOAD_DIR / f".uploading_{uuid.uuid4().hex}{suffix}"
    _, file_hash = await _stream_to_disk(file, tmp_path)
    safe_name = f"{file_hash}_{_safe_filename(file.filename or 'upload')}"
    if not safe_name.lower().endswith(suffix):
        safe_name += suffix
    persistent_path = UPLOAD_DIR / safe_name
    tmp_path.replace(persistent_path)
    return persistent_path, safe_name


def _build_meta_with_source_url(
    title: str, domain: str, description: str, tags: str, url: str, safe_name: str,
) -> dict:
    meta = _build_metadata_dict(title, domain, description, tags, url)
    if not meta.get("url"):
        meta["source_url"] = f"/files/{safe_name}"
    return meta


async def _submit_file_job(
    *,
    persistent_path: Path,
    safe_name: str,
    job_type: JobType,
    filename: str,
    metadata: dict,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Tạo + submit job, trả về {job_id, filename}."""
    store = get_store()
    job = await store.create_job(
        job_type=job_type,
        filename=filename,
        batch_id=batch_id,
        metadata={"source_url": metadata.get("source_url") or metadata.get("url")},
    )
    await get_runner().submit(job, payload={
        "file_path": str(persistent_path),
        "filename": filename,
        "metadata": metadata,
    })
    return {"job_id": job.job_id, "filename": filename}


# --- /file/async ---

@router.post("/file/async", response_model=JobSubmitResponse)
async def ingest_file_async(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    domain: str = Form(default=""),
    description: str = Form(default=""),
    tags: str = Form(default=""),
    url: str = Form(default=""),
) -> JobSubmitResponse:
    domain_err = _validate_domain(domain)
    if domain_err:
        raise HTTPException(status_code=400, detail=domain_err)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in DOC_SUFFIXES_FULL:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng '{suffix}' không hỗ trợ.",
        )

    try:
        persistent_path, safe_name = await _persist_upload(file, suffix)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    meta = _build_meta_with_source_url(title, domain, description, tags, url, safe_name)
    sub = await _submit_file_job(
        persistent_path=persistent_path,
        safe_name=safe_name,
        job_type="document",
        filename=title.strip() or file.filename,
        metadata=meta,
    )
    return JobSubmitResponse(job_id=sub["job_id"], filename=sub["filename"])


# --- /video/file/async ---

@router.post("/video/file/async", response_model=JobSubmitResponse)
async def ingest_video_file_async(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    domain: str = Form(default=""),
    description: str = Form(default=""),
    tags: str = Form(default=""),
    url: str = Form(default=""),
) -> JobSubmitResponse:
    domain_err = _validate_domain(domain)
    if domain_err:
        raise HTTPException(status_code=400, detail=domain_err)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng '{suffix}' không hỗ trợ.",
        )

    try:
        persistent_path, safe_name = await _persist_upload(file, suffix)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    meta = _build_meta_with_source_url(title, domain, description, tags, url, safe_name)
    sub = await _submit_file_job(
        persistent_path=persistent_path,
        safe_name=safe_name,
        job_type="video_file",
        filename=title.strip() or file.filename,
        metadata=meta,
    )
    return JobSubmitResponse(job_id=sub["job_id"], filename=sub["filename"])


# --- /youtube/async ---

@router.post("/youtube/async", response_model=JobSubmitResponse)
async def ingest_youtube_async(
    url: str = Form(...),
    title: str = Form(default=""),
    domain: str = Form(default=""),
    description: str = Form(default=""),
    tags: str = Form(default=""),
) -> JobSubmitResponse:
    if not url.strip():
        raise HTTPException(status_code=400, detail="URL bắt buộc.")
    domain_err = _validate_domain(domain)
    if domain_err:
        raise HTTPException(status_code=400, detail=domain_err)

    clean_url = url.strip()
    is_playlist = _is_playlist_url(clean_url)
    meta = _build_metadata_dict(title, domain, description, tags, url=clean_url)
    if is_playlist:
        meta = {"domain": meta["domain"]} if meta.get("domain") else None

    store = get_store()
    job = await store.create_job(
        job_type="youtube",
        filename=clean_url,
        metadata={"is_playlist": is_playlist},
    )
    await get_runner().submit(job, payload={
        "url": clean_url,
        "is_playlist": is_playlist,
        "metadata": meta,
    })
    return JobSubmitResponse(job_id=job.job_id, filename=clean_url)


# ===========================================================================
# Batch upload — multipart array hoặc list URLs.
# ===========================================================================

def _classify_job_type(suffix: str) -> JobType | None:
    if suffix in DOC_SUFFIXES_FULL:
        return "document"
    if suffix in VIDEO_SUFFIXES:
        return "video_file"
    return None


@router.post("/files", response_model=BatchSubmitResponse)
async def ingest_files_batch(
    files: list[UploadFile] = File(...),
    domain: str = Form(default=""),
    description: str = Form(default=""),
    tags: str = Form(default=""),
) -> BatchSubmitResponse:
    """Upload nhiều file 1 request. Mỗi file → 1 job. Tự route doc/video theo suffix.

    Title được dùng từ tên file (không cho user override per-file ở batch — nếu cần
    thì gọi /file/async lần lượt, hoặc dùng /from-urls với metadata từng item).
    """
    domain_err = _validate_domain(domain)
    if domain_err:
        raise HTTPException(status_code=400, detail=domain_err)
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất 1 file.")
    if len(files) > INGEST_MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Vượt quá {INGEST_MAX_BATCH_SIZE} file/batch.",
        )

    batch_id = str(uuid.uuid4())
    submitted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        job_type = _classify_job_type(suffix)
        if job_type is None:
            errors.append({"filename": f.filename, "error": f"Định dạng '{suffix}' không hỗ trợ."})
            continue
        try:
            persistent_path, safe_name = await _persist_upload(f, suffix)
        except ValueError as exc:
            errors.append({"filename": f.filename, "error": str(exc)})
            continue

        meta = _build_meta_with_source_url(
            title="", domain=domain, description=description, tags=tags, url="",
            safe_name=safe_name,
        )
        sub = await _submit_file_job(
            persistent_path=persistent_path,
            safe_name=safe_name,
            job_type=job_type,
            filename=f.filename or safe_name,
            metadata=meta,
            batch_id=batch_id,
        )
        submitted.append(sub)

    return BatchSubmitResponse(
        batch_id=batch_id,
        total=len(submitted),
        jobs=submitted,
        errors=errors,
    )


@router.post("/from-urls", response_model=BatchSubmitResponse)
async def ingest_from_urls(payload: FromUrlsRequest = Body(...)) -> BatchSubmitResponse:
    """Ingest từ danh sách URL. Runner tự stream-download → parse.

    Dùng cho:
      - BE đã upload S3, đẩy presigned URL sang AI
      - SharePoint/Graph download URL có Bearer token (BE lo auth)
      - Bất kỳ URL public nào
    """
    items = payload.items
    if not items:
        raise HTTPException(status_code=400, detail="items rỗng.")
    if len(items) > INGEST_MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Vượt quá {INGEST_MAX_BATCH_SIZE} item/batch.",
        )

    batch_id = str(uuid.uuid4())
    submitted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    store = get_store()
    runner = get_runner()

    for item in items:
        meta = item.metadata or {}
        domain_err = _validate_domain(str(meta.get("domain", "")))
        if domain_err:
            errors.append({"filename": item.filename, "error": domain_err})
            continue

        suffix = Path(item.filename).suffix.lower()
        job_type = _classify_job_type(suffix)
        if job_type is None:
            errors.append({"filename": item.filename, "error": f"Định dạng '{suffix}' không hỗ trợ."})
            continue

        job = await store.create_job(
            job_type=job_type,
            filename=item.filename,
            batch_id=batch_id,
        )
        await runner.submit(job, payload={
            "download_url": item.download_url,
            "headers": item.headers or {},
            "filename": item.filename,
            "metadata": meta,
        })
        submitted.append({"job_id": job.job_id, "filename": item.filename})

    return BatchSubmitResponse(
        batch_id=batch_id,
        total=len(submitted),
        jobs=submitted,
        errors=errors,
    )


# ===========================================================================
# Job / batch status polling.
# ===========================================================================

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = await get_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job không tồn tại hoặc đã hết hạn.")
    return JobStatusResponse(**job.to_dict())


@router.get("/batches/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str) -> BatchStatusResponse:
    summary = await get_store().get_batch(batch_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Batch không tồn tại hoặc đã hết hạn.")
    return BatchStatusResponse(
        batch_id=summary.batch_id,
        total=summary.total,
        queued=summary.queued,
        in_progress=summary.in_progress,
        done=summary.done,
        failed=summary.failed,
        chunks_added=summary.chunks_added,
        jobs=summary.jobs,
    )
