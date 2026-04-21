"""FastAPI server — RAG Chatbot Trung Tâm Tri Thức.

Routes:
  GET    /health
  POST   /api/chat           — chat đồng bộ
  POST   /api/chat/stream    — chat SSE streaming
  POST   /api/ingest/doc     — nạp tài liệu (PDF/DOCX/TXT/MD)
  POST   /api/ingest/video   — nạp video YouTube
  DELETE /api/session/{id}   — xoá lịch sử hội thoại
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.core import config
from app.core.session_memory import memory
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    SessionClearResponse,
    SourceItem,
    VideoIngestRequest,
    VideoIngestResponse,
)

app = FastAPI(
    title="Trung Tâm Tri Thức — RAG Chatbot",
    version="1.0.0",
    description="Hệ thống hỏi-đáp nội bộ dựa trên Voyage + Qdrant + Claude",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).parent.parent.parent / "web"

# Lazy-loaded heavy dependencies (set during startup)
_rag_chain = None
_doc_pipeline = None
_video_pipeline = None
_qdrant_client = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    global _rag_chain, _doc_pipeline, _video_pipeline, _qdrant_client
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from app.rag.chain import RAGChain
        _rag_chain = RAGChain()
    except Exception as e:
        print(f"[WARN] RAGChain không khởi động được: {e}")

    try:
        from app.ingestion.doc_pipeline import DocPipeline
        _doc_pipeline = DocPipeline()
    except Exception as e:
        print(f"[WARN] DocPipeline không khởi động được: {e}")

    try:
        from app.ingestion.video_pipeline import VideoPipeline
        _video_pipeline = VideoPipeline()
    except Exception as e:
        print(f"[WARN] VideoPipeline không khởi động được: {e}")

    try:
        from app.core.qdrant_client import get_client
        _qdrant_client = get_client()
    except Exception as e:
        print(f"[WARN] Qdrant client không kết nối được: {e}")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    collections: dict = {}
    if _qdrant_client:
        try:
            for col in [config.COLLECTION_DOCS, config.COLLECTION_VIDEOS]:
                info = _qdrant_client.get_collection(col)
                collections[col] = {"points": info.points_count}
        except Exception:
            pass
    return HealthResponse(
        status="ok",
        model=config.CLAUDE_MODEL,
        collections=collections,
    )


# ---------------------------------------------------------------------------
# Chat — đồng bộ
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if _rag_chain is None:
        raise HTTPException(503, detail="RAG chưa sẵn sàng, vui lòng thử lại sau.")
    history = memory.get_history(req.session_id)
    result = _rag_chain.query(req.message, history=list(history), top_k=req.top_k)
    memory.add_turn(req.session_id, req.message, result["answer"])
    sources = [SourceItem(**s) if isinstance(s, dict) else s for s in result.get("sources", [])]
    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        session_id=req.session_id,
        confidence=result.get("confidence", "high"),
        rewritten_query=result.get("rewritten_query", ""),
        suggested_questions=result.get("suggested_questions", []),
    )


# ---------------------------------------------------------------------------
# Chat — SSE streaming
# ---------------------------------------------------------------------------

@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    if _rag_chain is None:
        raise HTTPException(503, detail="RAG chưa sẵn sàng, vui lòng thử lại sau.")

    history_snapshot = list(memory.get_history(req.session_id))

    def _event_gen():
        full_answer: list[str] = []
        try:
            for event in _rag_chain.query_stream(
                req.message, history=history_snapshot, top_k=req.top_k
            ):
                if event["type"] == "done":
                    full_answer.append(event.get("answer", ""))
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            err = {"type": "error", "message": str(exc)}
            yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"
            return

        answer = "".join(full_answer)
        memory.add_turn(req.session_id, req.message, answer)

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Ingest — tài liệu
# ---------------------------------------------------------------------------

@app.post("/api/ingest/doc", response_model=IngestResponse)
async def ingest_doc(
    file: UploadFile = File(...),
    collection: Optional[str] = Form(None),
) -> IngestResponse:
    if _doc_pipeline is None:
        raise HTTPException(503, detail="Pipeline tài liệu chưa sẵn sàng.")

    target_col = collection or config.COLLECTION_DOCS
    dest = config.UPLOADS_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)

    try:
        chunks_added = _doc_pipeline.ingest_file(dest, collection=target_col)
    except Exception as exc:
        raise HTTPException(500, detail=f"Lỗi khi nạp tài liệu: {exc}")

    return IngestResponse(
        status="ok",
        chunks_added=chunks_added,
        collection=target_col,
        filename=file.filename,
    )


# ---------------------------------------------------------------------------
# Ingest — video YouTube
# ---------------------------------------------------------------------------

@app.post("/api/ingest/video", response_model=VideoIngestResponse)
def ingest_video(req: VideoIngestRequest) -> VideoIngestResponse:
    if _video_pipeline is None:
        raise HTTPException(503, detail="Pipeline video chưa sẵn sàng.")

    target_col = req.collection or config.COLLECTION_VIDEOS
    try:
        result = _video_pipeline.ingest_url(req.url, title=req.title, collection=target_col)
    except Exception as exc:
        raise HTTPException(500, detail=f"Lỗi khi nạp video: {exc}")

    return VideoIngestResponse(
        status="ok",
        chunks_added=result.get("chunks_added", 0),
        collection=target_col,
        title=result.get("title", req.title or req.url),
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.delete("/api/session/{session_id}", response_model=SessionClearResponse)
def clear_session(session_id: str) -> SessionClearResponse:
    memory.clear(session_id)
    return SessionClearResponse(session_id=session_id, status="cleared")


# ---------------------------------------------------------------------------
# Static / UI
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    import uvicorn
    uvicorn.run(
        "app.api.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    run()
