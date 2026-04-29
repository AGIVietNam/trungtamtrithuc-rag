"""Async job runner: pull job từ asyncio.Queue → dispatch handler.

Concurrency được giới hạn bằng N worker tasks chạy song song. Mỗi worker:
  1. await queue.get()
  2. update store status: parsing
  3. asyncio.to_thread(handler) — pipeline ingest hiện tại là sync (Docling +
     Whisper + Voyage), không thể chạy trong event loop trực tiếp
  4. update store status: done / failed

Khi BE chuyển sang RabbitMQ, file này được thay bằng `aio_pika.Consumer`
gọi cùng `_handle_*` functions — không phải sửa pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

import httpx

from app.core.job_store import InMemoryJobStore, JobStatus

logger = logging.getLogger(__name__)


def _filename_looks_like_html(name: str) -> bool:
    return name.lower().endswith((".html", ".htm"))


class JobRunner:
    def __init__(self, store: InMemoryJobStore, concurrency: int = 2) -> None:
        self._store = store
        self._concurrency = max(1, concurrency)
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker_loop(i), name=f"ingest-worker-{i}")
            for i in range(self._concurrency)
        ]
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="job-cleanup")
        logger.info("JobRunner started: %d workers", self._concurrency)

    async def stop(self) -> None:
        self._running = False
        for w in self._workers:
            w.cancel()
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
        await asyncio.gather(*self._workers, self._cleanup_task, return_exceptions=True)
        self._workers = []
        self._cleanup_task = None
        logger.info("JobRunner stopped")

    async def submit(self, job: JobStatus, payload: dict[str, Any]) -> None:
        """Push job vào queue. Payload chứa file_path / url / metadata cụ thể."""
        await self._queue.put({"job_id": job.job_id, "payload": payload})

    # --- Worker loop ---

    async def _worker_loop(self, worker_idx: int) -> None:
        while self._running:
            try:
                msg = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self._dispatch(msg)
            except Exception as exc:
                logger.exception("worker-%d unhandled error: %s", worker_idx, exc)
            finally:
                self._queue.task_done()

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        job_id = msg["job_id"]
        payload = msg["payload"]
        job = await self._store.get(job_id)
        if job is None:
            logger.warning("Dispatch: job %s not in store, skipped", job_id)
            return

        try:
            if job.job_type == "document":
                await self._handle_document(job, payload)
            elif job.job_type == "video_file":
                await self._handle_video_file(job, payload)
            elif job.job_type == "youtube":
                await self._handle_youtube(job, payload)
            else:
                raise ValueError(f"unknown job_type: {job.job_type}")
        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            await self._store.update(job_id, status="failed", error=str(exc))
        finally:
            # Notify webhook nếu submitter đăng ký callback_url. Best-effort:
            # callback fail KHÔNG đổi job status (job đã done/failed thì vẫn vậy).
            await self._notify_callback(job_id, payload)
            await self._notify_backend_document_webhook(job_id)

    async def _notify_callback(self, job_id: str, payload: dict[str, Any]) -> None:
        callback_url = payload.get("callback_url")
        if not callback_url:
            return
        job = await self._store.get(job_id)
        if job is None or job.status not in ("done", "failed"):
            return

        body = {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "filename": job.filename,
            "status": job.status,
            "chunks_added": job.chunks_added,
            "pages": job.pages,
            "error": job.error,
            "duration_sec": (job.finished_at or 0) - (job.started_at or job.created_at),
            "document_id": job.document_id,
            "metadata": job.metadata,
        }
        try:
            timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(callback_url, json=body)
                logger.info(
                    "Callback %s for job %s → HTTP %d",
                    callback_url[:60], job_id, resp.status_code,
                )
        except Exception as exc:
            # Không retry — BE phải design nhận lại qua poll /jobs/{id} nếu webhook miss.
            logger.warning("Callback failed for job %s: %s", job_id, exc)

    async def _notify_backend_document_webhook(self, job_id: str) -> None:
        """POST {document_id, status} sang BE document webhook khi terminal.

        BE truyền `document_id` lúc submit (`/api/ingest/from-url`); AI giữ trong
        JobStatus và bắn lại để BE đánh dấu document `success`/`failed`.
        Skip nếu job không có `document_id` (flow upload không qua BE).
        Best-effort — webhook fail không đổi trạng thái job nội bộ.
        """
        from app.config import BACKEND_DOCUMENT_WEBHOOK_URL, BACKEND_WEBHOOK_API_KEY

        job = await self._store.get(job_id)
        if job is None or not job.document_id:
            return
        if job.status not in ("done", "failed"):
            return
        if not BACKEND_DOCUMENT_WEBHOOK_URL or not BACKEND_WEBHOOK_API_KEY:
            logger.warning(
                "BE webhook skip for document_id=%s — chưa cấu hình "
                "BACKEND_DOCUMENT_WEBHOOK_URL hoặc BACKEND_WEBHOOK_API_KEY",
                job.document_id,
            )
            return

        body = {
            "document_id": job.document_id,
            "status": "success" if job.status == "done" else "failed",
        }
        headers = {
            "x-api-key": BACKEND_WEBHOOK_API_KEY,
            "Content-Type": "application/json",
            "accept": "*/*",
        }
        try:
            timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    BACKEND_DOCUMENT_WEBHOOK_URL, json=body, headers=headers,
                )
                logger.info(
                    "BE webhook for document_id=%s status=%s → HTTP %d",
                    job.document_id, body["status"], resp.status_code,
                )
        except Exception as exc:
            logger.warning(
                "BE webhook failed for document_id=%s: %s", job.document_id, exc,
            )

    # --- Handlers ---

    async def _handle_document(self, job: JobStatus, payload: dict[str, Any]) -> None:
        from app.ingestion.doc_pipeline import ingest_document

        file_path = await self._resolve_file(job, payload)
        try:
            await self._store.update(job.job_id, status="parsing", progress=0.3)
            result = await asyncio.to_thread(
                ingest_document,
                file_path=str(file_path),
                original_name=payload.get("filename") or job.filename,
                metadata=payload.get("metadata"),
            )
            await self._store.update(
                job.job_id,
                status="done",
                chunks_added=result.num_chunks,
                pages=result.num_pages,
            )
        finally:
            self._cleanup_temp(payload, file_path)

    async def _handle_video_file(self, job: JobStatus, payload: dict[str, Any]) -> None:
        from app.ingestion.video_pipeline import ingest_video_file

        file_path = await self._resolve_file(job, payload)
        try:
            await self._store.update(job.job_id, status="parsing", progress=0.3)
            result = await asyncio.to_thread(
                ingest_video_file,
                local_path=str(file_path),
                original_name=payload.get("filename") or job.filename,
                metadata=payload.get("metadata"),
            )
            await self._store.update(
                job.job_id,
                status="done",
                chunks_added=result.num_chunks,
                pages=result.num_pages,
            )
        finally:
            self._cleanup_temp(payload, file_path)

    async def _handle_youtube(self, job: JobStatus, payload: dict[str, Any]) -> None:
        from app.ingestion.video_pipeline import (
            ingest_youtube,
            ingest_youtube_playlist,
        )

        url = payload["url"]
        await self._store.update(job.job_id, status="parsing", progress=0.3)

        if payload.get("is_playlist"):
            data = await asyncio.to_thread(
                ingest_youtube_playlist,
                playlist_url=url,
                metadata=payload.get("metadata"),
            )
            results = data.get("results") or []
            total_chunks = sum(r.get("chunks_added", 0) for r in results)
            ok_count = sum(1 for r in results if r.get("status") == "ok")
            await self._store.update(
                job.job_id,
                status="done",
                chunks_added=total_chunks,
                pages=ok_count,
            )
            return

        result = await asyncio.to_thread(
            ingest_youtube,
            url=url,
            metadata=payload.get("metadata"),
        )
        await self._store.update(
            job.job_id,
            status="done",
            chunks_added=result.num_chunks,
            pages=result.num_pages,
        )

    # --- File resolution: hoặc đã có local path, hoặc download từ URL ---

    async def _resolve_file(self, job: JobStatus, payload: dict[str, Any]) -> Path:
        if payload.get("file_path"):
            return Path(payload["file_path"])

        url = payload.get("download_url")
        if not url:
            raise ValueError("payload thiếu file_path hoặc download_url")

        await self._store.update(job.job_id, status="downloading", progress=0.1)
        suffix = Path(payload.get("filename") or "download").suffix or ".bin"
        tmp = Path(tempfile.mktemp(suffix=suffix))
        headers = payload.get("headers") or {}
        from app.config import MAX_UPLOAD_MB, UPLOAD_STREAM_CHUNK
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024

        written = 0
        try:
            timeout = httpx.Timeout(connect=30.0, read=None, write=None, pool=30.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                async with client.stream("GET", url, headers=headers) as resp:
                    self._validate_response(resp, url)

                    # Override filename theo Content-Disposition nếu server cung cấp
                    cd_name = self._filename_from_content_disposition(resp.headers.get("content-disposition"))
                    if cd_name:
                        payload["filename"] = cd_name

                    # Detect HTML body sớm = login page / error page → reject thay
                    # vì để Docling parse rác. Generic content-type cho file là
                    # application/*, video/*, image/*, text/plain, ...
                    ctype = (resp.headers.get("content-type") or "").lower()
                    if ctype.startswith("text/html") and not _filename_looks_like_html(payload.get("filename") or ""):
                        raise ValueError(
                            "Nội dung tải về là HTML — link có thể yêu cầu đăng nhập, "
                            "đã bị xoá, hoặc trỏ tới trang web thay vì file. "
                            f"Status: {resp.status_code}, URL cuối: {str(resp.url)[:100]}"
                        )

                    with tmp.open("wb") as f:
                        async for chunk in resp.aiter_bytes(UPLOAD_STREAM_CHUNK):
                            written += len(chunk)
                            if written > max_bytes:
                                raise ValueError(
                                    f"File vượt quá giới hạn {MAX_UPLOAD_MB} MB."
                                )
                            f.write(chunk)
        except httpx.HTTPStatusError as exc:
            tmp.unlink(missing_ok=True)
            raise ValueError(self._friendly_http_error(exc, url)) from exc
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        if written == 0:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                "File tải về 0 byte — có thể link không tồn tại, đã bị xoá, "
                "hoặc cần quyền truy cập."
            )

        # Mark file_path nội bộ — để cleanup biết phải xoá
        payload["_downloaded_tmp"] = str(tmp)
        return tmp

    @staticmethod
    def _validate_response(resp: httpx.Response, original_url: str) -> None:
        """Raise HTTPStatusError với context nếu response không 2xx."""
        if 200 <= resp.status_code < 300:
            return
        # raise_for_status() bỏ vào except phía trên để format friendly
        resp.raise_for_status()

    @staticmethod
    def _friendly_http_error(exc: httpx.HTTPStatusError, url: str) -> str:
        """Format error message thân thiện theo HTTP status."""
        code = exc.response.status_code
        url_short = url[:120]
        if code == 401:
            return f"Link cần đăng nhập (401). Nếu là link nội bộ (SharePoint/OneDrive), BE cần resolve qua Microsoft Graph trước. URL: {url_short}"
        if code == 403:
            return f"Không có quyền truy cập link (403). Link có thể là nội bộ hoặc đã thu hồi quyền share. URL: {url_short}"
        if code == 404:
            return f"Link không tồn tại (404). URL: {url_short}"
        if code == 429:
            return f"Nguồn rate-limit (429). Thử lại sau ít phút. URL: {url_short}"
        if 500 <= code < 600:
            return f"Server nguồn lỗi ({code}). Có thể tạm thời, thử lại sau. URL: {url_short}"
        return f"HTTP {code} khi tải. URL: {url_short}"

    @staticmethod
    def _filename_from_content_disposition(header: str | None) -> str | None:
        """Extract filename từ header `attachment; filename="abc.pdf"` hoặc filename*=UTF-8''..."""
        if not header:
            return None
        import re
        from urllib.parse import unquote
        # filename*=UTF-8''encoded-name (RFC 5987)
        m = re.search(r"filename\*\s*=\s*[^']*'[^']*'([^;]+)", header, re.IGNORECASE)
        if m:
            return unquote(m.group(1).strip().strip('"'))
        # filename="abc.pdf"
        m = re.search(r'filename\s*=\s*"([^"]+)"', header, re.IGNORECASE)
        if m:
            return m.group(1)
        # filename=abc.pdf
        m = re.search(r'filename\s*=\s*([^;]+)', header, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _cleanup_temp(payload: dict[str, Any], file_path: Path) -> None:
        # Chỉ xoá file mà runner tự download; file do API endpoint copy đến
        # `data/uploads/...` sẽ KHÔNG bị xoá (đó là persistent path).
        downloaded = payload.get("_downloaded_tmp")
        if downloaded and Path(downloaded) == file_path:
            file_path.unlink(missing_ok=True)
        elif payload.get("delete_after"):
            file_path.unlink(missing_ok=True)

    # --- Cleanup loop: xoá job cũ khỏi store theo TTL ---

    async def _cleanup_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(300)  # 5 phút/lần
                removed = await self._store.cleanup_expired()
                if removed:
                    logger.info("Cleaned %d expired jobs from store", removed)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("cleanup loop error")


# Singleton — main.py lifespan tạo + start
_runner: JobRunner | None = None


def get_runner() -> JobRunner:
    global _runner
    if _runner is None:
        from app.config import INGEST_WORKER_CONCURRENCY
        from app.core.job_store import get_store
        _runner = JobRunner(store=get_store(), concurrency=INGEST_WORKER_CONCURRENCY)
    return _runner
