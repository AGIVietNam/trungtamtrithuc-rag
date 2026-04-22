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

# `app.ingestion.doc_pipeline` gắn FileHandler lên root logger (catch mọi
# module), nhưng level root mặc định = WARNING → INFO bị filter trước khi
# tới handler. Bật INFO cho namespace `app` để các log timing trong
# RAGChain.answer + claude usage trong claude_client xuất hiện. Không động
# tới root → urllib3/anthropic/etc. vẫn yên.
_app_logger = logging.getLogger("app")
_app_logger.setLevel(logging.INFO)

# File handler: ghi mọi log namespace `app.*` (middleware timing, endpoint
# step breakdown, RAGChain, pipelines) ra file xoay vòng theo ngày. Tách
# biệt với ingest log cũ trong doc_pipeline.py để không bị trộn format.
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
    """Eager-load cross-encoder + chain lúc startup → request đầu không delay."""
    import asyncio

    def _warmup() -> None:
        try:
            from app.rag.reranker import warmup as _rerank_warmup
            _rerank_warmup()
        except Exception:
            logger.exception("Reranker warmup failed")

        # Tạo payload index cho ttt_memory (idempotent). Trước fix này, mọi
        # filter user_id/session_id trên ttt_memory đều fail 400, khiến
        # tier-3 vector recall + semantic dedup chưa từng chạy đúng.
        try:
            from app.api.chat import _get_conv_memory
            _get_conv_memory().ensure_indexes()
        except Exception:
            logger.exception("conv_memory ensure_indexes failed")

    # Chạy trong thread riêng để không block event loop (model load CPU-bound).
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


# Log tổng thời gian mỗi request. Step-level timing được log trong từng
# endpoint/pipeline (xem app.api.ingest, app.rag.chain) — middleware này chỉ
# ghi "wall clock" toàn API để so sánh với step breakdown.
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

# Routes registered lazily to avoid circular imports at startup
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
