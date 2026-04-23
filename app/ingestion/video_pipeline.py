from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core import config
from app.core.chunker import chunk_transcript_with_timestamps
from app.core.voyage_embed import VoyageEmbedder
from app.core.qdrant_store import QdrantStore, QdrantRegistry, PERSONA_TO_DOMAIN

logger = logging.getLogger(__name__)


_registry: QdrantRegistry | None = None


def _get_registry() -> QdrantRegistry:
    """Lazy singleton dùng chung cho cả video_pipeline (độc lập với doc_pipeline)."""
    global _registry
    if _registry is None:
        _registry = QdrantRegistry(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            vector_size=config.VOYAGE_DIM,
            vector_name=getattr(config, "QDRANT_VECTOR_NAME", ""),
        )
    return _registry


def _resolve_domain_store(metadata: dict | None) -> QdrantStore:
    """Pick đúng tdi_videos_{slug} theo metadata['domain']."""
    domain_key = (metadata or {}).get("domain", "").strip()
    if not domain_key:
        raise ValueError(
            "metadata['domain'] là bắt buộc khi ingest video — "
            "hãy chọn lĩnh vực ở form upload."
        )
    if domain_key not in PERSONA_TO_DOMAIN:
        raise ValueError(
            f"domain={domain_key!r} không hợp lệ. "
            f"Chọn 1 trong: {list(PERSONA_TO_DOMAIN.keys())}"
        )
    return _get_registry().get_by_persona(domain_key, "videos")


def format_transcript_string(segments: list[dict]) -> str:
    """Convert segments to timestamped string: '00:00 text 00:07 more text'."""
    parts: list[str] = []
    for seg in segments:
        secs = int(seg.get("start", 0))
        mm, ss = divmod(secs, 60)
        hh, mm = divmod(mm, 60)
        ts = f"{hh:02d}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"
        text = seg.get("text", "").strip()
        if text:
            parts.append(f"{ts} {text}")
    return " ".join(parts)


@dataclass
class IngestResult:
    doc_id: str
    num_chunks: int
    num_pages: int
    source_name: str


def build_video_link(payload: dict) -> str:
    if payload.get("file_source") == "youtube":
        video_id = payload.get("video_id", "")
        start_sec = int(payload.get("start_sec", 0))
        return f"https://www.youtube.com/watch?v={video_id}&t={start_sec}s"
    video_id = payload.get("video_id", "")
    start_sec = int(payload.get("start_sec", 0))
    return f"/files/videos/{video_id}#t={start_sec}"


def ensure_collections() -> None:
    """No-op — registry.ensure_all() ở main.py lifespan đã tạo đủ 20 collection.

    Giữ symbol này để không vỡ import từ app/api/ingest.py (backward compat).
    """
    return


_PLAYLIST_PAYLOAD_KEYS = (
    "playlist_id",
    "playlist_title",
    "playlist_description",
    "playlist_uploader",
    "playlist_thumbnail",
    "playlist_url",
)


def _upsert_video_chunks(
    segments: list[dict],
    video_id: str,
    title: str,
    source_url: str | None,
    file_source: str,
    metadata: dict | None,
    playlist_info: dict | None = None,
) -> int:
    uploaded_at = datetime.now(timezone.utc).isoformat()
    chunks = chunk_transcript_with_timestamps(segments, max_tokens=500)

    if not chunks:
        return 0

    # Route theo domain — raise ValueError nếu metadata thiếu/sai trước khi tốn
    # embedding cost.
    store = _resolve_domain_store(metadata)

    embedder = VoyageEmbedder(api_key=config.VOYAGE_API_KEY, model=config.VOYAGE_MODEL)

    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_documents(texts)

    points: list[dict] = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        start_sec = chunk["start"]
        end_sec = chunk["end"]
        ts_start = int(start_sec)
        mm, ss = divmod(ts_start, 60)
        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{video_id}-{i}",
        ))
        payload: dict = {
            "source_type": "video",
            "video_id": video_id,
            "title": title,
            "source_url": source_url,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "timestamp": f"{mm:02d}:{ss:02d}",
            "text": chunk["text"],
            "segment_ids": chunk.get("segment_ids", []),
            "uploaded_at": uploaded_at,
            "file_source": file_source,
        }
        if file_source == "youtube" and source_url:
            payload["youtube_url"] = f"https://www.youtube.com/watch?v={video_id}&t={int(start_sec)}s"
        # Inject metadata top-level để filter/display đồng bộ với doc pipeline
        if metadata:
            if metadata.get("domain"):
                payload["domain"] = metadata["domain"]
            if metadata.get("description"):
                payload["description"] = metadata["description"]
            if metadata.get("tags"):
                payload["tags"] = metadata["tags"]
            if metadata.get("url") and not payload.get("source_url"):
                payload["source_url"] = metadata["url"]
            payload["extra_metadata"] = metadata
        if playlist_info:
            for key in _PLAYLIST_PAYLOAD_KEYS:
                val = playlist_info.get(key)
                if val:
                    payload[key] = val
        points.append({"id": point_id, "vector": vector, "payload": payload})

    store.upsert(points)
    logger.info(
        "Upserted %d video chunks for video_id=%s title=%s",
        len(points), video_id, title,
    )
    return len(points)


def ingest_video_file(
    local_path: str,
    original_name: str,
    metadata: dict | None = None,
) -> IngestResult:
    from app.ingestion.video_transcriber import get_transcriber

    path = Path(local_path)
    logger.info("Transcribing local video: %s", original_name)

    transcriber = get_transcriber()
    result = transcriber.transcribe(path)
    segments = result["segments"]

    import hashlib
    video_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]

    # Ưu tiên title từ metadata (user sửa hoặc AI gen) → fallback filename
    title = (metadata or {}).get("title") or original_name

    num_chunks = _upsert_video_chunks(
        segments=segments,
        video_id=video_id,
        title=title,
        source_url=None,
        file_source="local",
        metadata=metadata,
    )
    return IngestResult(
        doc_id=video_id,
        num_chunks=num_chunks,
        num_pages=1,
        source_name=title,
    )


def ingest_youtube(
    url: str,
    metadata: dict | None = None,
    playlist_info: dict | None = None,
) -> IngestResult:
    from app.ingestion.youtube_fetcher import (
        fetch_youtube_transcript, fetch_youtube_via_whisper,
    )

    logger.info("Ingesting YouTube video: %s", url)
    try:
        data = fetch_youtube_transcript(url)
    except Exception as exc:
        # Transcript API đã retry qua proxy list mà vẫn bị chặn hoặc không có transcript.
        # Fallback: yt-dlp tải audio → Whisper phiên âm (không cần endpoint transcript).
        logger.warning(
            "Transcript API failed (%s). Fallback yt-dlp + Whisper cho %s",
            type(exc).__name__, url,
        )
        data = fetch_youtube_via_whisper(url)

    # Ưu tiên title từ metadata user-confirmed → fallback transcript/YouTube title
    effective_title = (metadata or {}).get("title") or data["title"]

    num_chunks = _upsert_video_chunks(
        segments=data["segments"],
        video_id=data["video_id"],
        title=effective_title,
        source_url=data["source_url"],
        file_source="youtube",
        metadata=metadata,
        playlist_info=playlist_info,
    )
    return IngestResult(
        doc_id=data["video_id"],
        num_chunks=num_chunks,
        num_pages=1,
        source_name=effective_title or url,
    )


def ingest_youtube_playlist(
    playlist_url: str,
    metadata: dict | None = None,
) -> dict:
    """Ingest all videos from a YouTube playlist, one by one.

    Returns:
      {
        "playlist_info": {"playlist_id", "playlist_title", ...},  # không kèm "videos"
        "results": [{"video_id", "title", "status", "chunks_added", "error"}, ...]
      }
    """
    from app.ingestion.youtube_fetcher import fetch_playlist_info

    logger.info("Fetching playlist: %s", playlist_url)
    info = fetch_playlist_info(playlist_url)
    videos = info.get("videos") or []
    logger.info(
        "Playlist '%s' (%s): %d videos",
        info.get("playlist_title") or "(no title)",
        info.get("playlist_id") or "?",
        len(videos),
    )

    # Stub gắn vào payload con — loại "videos" để tránh nhồi danh sách lớn.
    playlist_payload = {k: v for k, v in info.items() if k != "videos" and v}

    results = []
    for i, v in enumerate(videos, 1):
        vid = v["video_id"]
        title = v["title"]
        logger.info("[%d/%d] Ingesting: %s (%s)", i, len(videos), title, vid)
        try:
            r = ingest_youtube(
                url=f"https://www.youtube.com/watch?v={vid}",
                metadata=metadata,
                playlist_info=playlist_payload or None,
            )
            results.append({
                "video_id": vid,
                "title": r.source_name,
                "status": "ok",
                "chunks_added": r.num_chunks,
                "error": None,
            })
        except Exception as exc:
            logger.exception("Failed to ingest video %s: %s", vid, exc)
            results.append({
                "video_id": vid,
                "title": title,
                "status": "error",
                "chunks_added": 0,
                "error": str(exc),
            })

    return {"playlist_info": playlist_payload, "results": results}
