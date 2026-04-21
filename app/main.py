from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import API_HOST, API_PORT

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
