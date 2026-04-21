from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(default="default")
    user_id: str | None = Field(
        default=None,
        description="Định danh user cho conversation memory. Nếu trống dùng session_id.",
    )
    history: list[ChatMessage] = Field(default_factory=list)
    domain: str = Field(default="general")


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str
    suggested_questions: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    source_type: str = Field(description="pdf | docx | txt | md | youtube | video")
    url: str | None = None
    collection: str = Field(default="ttt_documents")


class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    message: str


class KnowledgeSearchRequest(BaseModel):
    query: str
    collection: str = Field(default="ttt_documents")
    top_k: int = Field(default=5)


class KnowledgeSearchResponse(BaseModel):
    results: list[dict[str, Any]]
