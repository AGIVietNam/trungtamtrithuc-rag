"""In-memory job + batch tracking cho async ingest pipeline.

Định danh:
- Job  = 1 file đang xử lý (parse → embed → upsert Qdrant)
- Batch = nhóm nhiều job liên kết bởi cùng `batch_id` (multipart array, from-urls list)

Storage: dict trong RAM. Mất nếu restart process — chấp nhận được vì worker sẽ
tự retry lần sau (BE re-publish). Khi BE chuyển sang RabbitMQ-backed queue,
chỉ cần thay class này bằng RedisJobStore với cùng interface.

Thread-safe: dùng asyncio.Lock vì runner chạy trong asyncio event loop. Nếu
sau này cần access từ thread khác (ví dụ FastAPI sync route), cần đổi sang
threading.Lock + RLock — hiện tại API endpoints đều async.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

JobType = Literal["document", "video_file", "youtube"]
JobStatusName = Literal["queued", "downloading", "parsing", "embedding", "done", "failed"]


@dataclass
class JobStatus:
    """Trạng thái 1 job ingest. Mutable — runner cập nhật progress in-place."""
    job_id: str
    job_type: JobType
    filename: str
    status: JobStatusName = "queued"
    progress: float = 0.0          # 0..1
    chunks_added: int = 0
    pages: int = 0
    error: str | None = None
    batch_id: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchSummary:
    """Aggregate snapshot từ list job_id thuộc 1 batch."""
    batch_id: str
    total: int
    queued: int
    in_progress: int
    done: int
    failed: int
    chunks_added: int
    jobs: list[dict[str, Any]]


_TERMINAL: frozenset[JobStatusName] = frozenset({"done", "failed"})
_IN_PROGRESS: frozenset[JobStatusName] = frozenset({"downloading", "parsing", "embedding"})


class InMemoryJobStore:
    """Job store dạng dict trong RAM, có TTL cleanup cho job đã terminal."""

    def __init__(self, ttl_sec: int = 3600) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._batches: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()
        self._ttl_sec = ttl_sec

    async def create_job(
        self,
        *,
        job_type: JobType,
        filename: str,
        batch_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JobStatus:
        job = JobStatus(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            filename=filename,
            batch_id=batch_id,
            metadata=metadata or {},
        )
        async with self._lock:
            self._jobs[job.job_id] = job
            if batch_id is not None:
                self._batches.setdefault(batch_id, []).append(job.job_id)
        return job

    async def update(
        self,
        job_id: str,
        *,
        status: JobStatusName | None = None,
        progress: float | None = None,
        chunks_added: int | None = None,
        pages: int | None = None,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
                if status in _IN_PROGRESS and job.started_at is None:
                    job.started_at = time.time()
                if status in _TERMINAL:
                    job.finished_at = time.time()
                    job.progress = 1.0 if status == "done" else job.progress
            if progress is not None:
                job.progress = max(0.0, min(1.0, progress))
            if chunks_added is not None:
                job.chunks_added = chunks_added
            if pages is not None:
                job.pages = pages
            if error is not None:
                job.error = error

    async def get(self, job_id: str) -> JobStatus | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def get_batch(self, batch_id: str) -> BatchSummary | None:
        async with self._lock:
            ids = self._batches.get(batch_id)
            if ids is None:
                return None
            jobs = [self._jobs[jid] for jid in ids if jid in self._jobs]
        queued = sum(1 for j in jobs if j.status == "queued")
        in_progress = sum(1 for j in jobs if j.status in _IN_PROGRESS)
        done = sum(1 for j in jobs if j.status == "done")
        failed = sum(1 for j in jobs if j.status == "failed")
        return BatchSummary(
            batch_id=batch_id,
            total=len(jobs),
            queued=queued,
            in_progress=in_progress,
            done=done,
            failed=failed,
            chunks_added=sum(j.chunks_added for j in jobs),
            jobs=[j.to_dict() for j in jobs],
        )

    async def cleanup_expired(self) -> int:
        """Xoá job terminal đã quá TTL. Trả về số job đã xoá."""
        cutoff = time.time() - self._ttl_sec
        removed = 0
        async with self._lock:
            stale = [
                jid for jid, j in self._jobs.items()
                if j.status in _TERMINAL and (j.finished_at or 0) < cutoff
            ]
            for jid in stale:
                job = self._jobs.pop(jid, None)
                if job and job.batch_id is not None:
                    ids = self._batches.get(job.batch_id, [])
                    if jid in ids:
                        ids.remove(jid)
                    if not ids:
                        self._batches.pop(job.batch_id, None)
                removed += 1
        return removed


# Singleton instance — main.py lifespan inject TTL từ config
_store: InMemoryJobStore | None = None


def get_store() -> InMemoryJobStore:
    global _store
    if _store is None:
        from app.config import INGEST_JOB_TTL_SEC
        _store = InMemoryJobStore(ttl_sec=INGEST_JOB_TTL_SEC)
    return _store
