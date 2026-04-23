from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import API_HOST, API_PORT, LOG_DIR

_app_logger = logging.getLogger("app")
_app_logger.setLevel(logging.INFO)

_API_LOG_FILE = LOG_DIR / "api.log"
if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(_API_LOG_FILE)
    for h in _app_logger.handlers
):
    _fh = TimedRotatingFileHandler(
        _API_LOG_FILE, when="midnight", backupCount=14, encoding="utf-8",
    )
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
    ))
    _fh.setLevel(logging.INFO)
    _app_logger.addHandler(_fh)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: tạo 24 Qdrant collections + warmup reranker + ensure memory indexes."""
    import asyncio

    def _warmup() -> None:
        # ── 1. Reranker warmup ────────────────────────────────────────────────
        try:
            from app.rag.reranker import warmup as _rerank_warmup
            _rerank_warmup()
        except Exception:
            logger.exception("Reranker warmup failed")

        # ── 2. Tạo 24 Qdrant collections (12 domain × docs + videos) ─────────
        # Idempotent: chỉ tạo collection nếu chưa tồn tại, không xóa data cũ.
        # Lần đầu deploy: tạo đủ 24. Restart sau: skip (collection đã có).
        try:
            from app.config import QDRANT_URL, QDRANT_API_KEY, VOYAGE_DIM, QDRANT_VECTOR_NAME
            from app.core.qdrant_store import QdrantRegistry
            registry = QdrantRegistry(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                vector_size=VOYAGE_DIM,
                vector_name=QDRANT_VECTOR_NAME,
            )
            registry.ensure_all()   # tạo đủ 24 collections, log tên từng cái
        except Exception:
            logger.exception("QdrantRegistry.ensure_all failed")

        # ── 3. Ensure payload indexes cho ttt_memory ──────────────────────────
        # (collection memory dùng filter user_id + session_id → cần keyword index)
        try:
            from app.api.chat import _get_conv_memory
            _get_conv_memory().ensure_indexes()
        except Exception:
            logger.exception("conv_memory ensure_indexes failed")

        # ── 4. Inject registry vào chain (thay thế qdrant_docs / qdrant_videos) ─
        # Chain sẽ dùng registry để route search đúng collection theo domain.
        try:
            from app.api.chat import _get_chain
            from app.config import QDRANT_URL, QDRANT_API_KEY, VOYAGE_DIM, QDRANT_VECTOR_NAME
            from app.core.qdrant_store import QdrantRegistry

            chain = _get_chain()
            chain.retriever.registry = QdrantRegistry(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                vector_size=VOYAGE_DIM,
                vector_name=QDRANT_VECTOR_NAME,
            )
            logger.info("chain.retriever.registry injected: 24 collections ready")
        except Exception:
            logger.exception("chain registry inject failed")

    try:
        await asyncio.to_thread(_warmup)
    except Exception:
        logger.exception("Warmup task failed")
    yield


app = FastAPI(title="Trung Tâm Tri Thức", version="1.0.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_request_duration(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception(
            "API %s %s failed after %.3fs",
            request.method, request.url.path, elapsed,
        )
        raise
    elapsed = time.perf_counter() - start
    logger.info(
        "API %s %s → %d in %.3fs",
        request.method, request.url.path, response.status_code, elapsed,
    )
    response.headers["X-Process-Time"] = f"{elapsed:.3f}"
    return response


from app.api import ingest, chat  # noqa: E402

app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


try:
    app.mount("/", StaticFiles(directory="web", html=True), name="static")
except Exception:
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)