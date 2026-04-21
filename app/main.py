from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import API_HOST, API_PORT

app = FastAPI(title="Trung Tâm Tri Thức", version="1.0.0")

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
