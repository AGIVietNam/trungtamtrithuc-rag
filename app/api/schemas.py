"""Pydantic request/response schemas cho tất cả API routes."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="ID phiên hội thoại")
    message: str = Field(..., min_length=1, description="Câu hỏi của người dùng")
    top_k: Optional[int] = Field(None, ge=1, le=20)


class SourceItem(BaseModel):
    title: str = ""
    source: str = ""
    score: float = 0.0
    timestamp: Optional[str] = None
    chunk_index: Optional[int] = None
    collection: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []
    session_id: str
    confidence: str = "high"
    rewritten_query: str = ""
    suggested_questions: list[str] = []


# ---------------------------------------------------------------------------
# Ingest — documents
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    collection: str
    filename: str


# ---------------------------------------------------------------------------
# Ingest — video / YouTube
# ---------------------------------------------------------------------------

class VideoIngestRequest(BaseModel):
    url: str = Field(..., description="YouTube URL hoặc đường dẫn file video")
    title: Optional[str] = None
    collection: Optional[str] = None  # defaults to COLLECTION_VIDEOS


class VideoIngestResponse(BaseModel):
    status: str
    chunks_added: int
    collection: str
    title: str


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class SessionClearResponse(BaseModel):
    session_id: str
    status: str


class HealthResponse(BaseModel):
    status: str
    model: str
    collections: dict[str, Any] = {}
