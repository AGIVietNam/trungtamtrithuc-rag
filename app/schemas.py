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
    user_name: str | None = Field(
        default=None,
        description=(
            "Tên hiển thị của user (vd 'Hoàng Quốc Tuấn'). Backend lấy từ JWT/"
            "session auth và truyền vào — AI dùng để personalize meta response "
            "(chào hỏi, trả lời 'Tôi là ai'). Nếu trống → AI fallback generic."
        ),
    )
    user_role: str | None = Field(
        default=None,
        description="Vai trò user (vd 'V365-AI', 'admin', 'editor'). Optional.",
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


# ---------------------------------------------------------------------------
# Async ingest / batch upload
# ---------------------------------------------------------------------------

class JobSubmitResponse(BaseModel):
    job_id: str
    filename: str


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    filename: str
    status: str
    progress: float
    chunks_added: int
    pages: int
    error: str | None = None
    batch_id: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    message: str = ""  # Câu thông báo sẵn cho FE hiển thị khi status terminal


class BatchSubmitResponse(BaseModel):
    batch_id: str
    total: int
    jobs: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class BatchStatusResponse(BaseModel):
    batch_id: str
    total: int
    queued: int
    in_progress: int
    done: int
    failed: int
    chunks_added: int
    jobs: list[dict[str, Any]] = Field(default_factory=list)


class FromUrlsItem(BaseModel):
    """1 item ingest qua URL — runner tự stream-download.

    `metadata.domain` BẮT BUỘC (1 trong 10 slug) để route Qdrant đúng collection.
    `headers` để truyền Bearer token cho SharePoint/Graph download URL.
    `callback_url` để AI POST status khi job done/failed (xem JobCallbackPayload).
    """
    download_url: str
    filename: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] | None = None
    size_bytes: int | None = None
    callback_url: str | None = None


class FromUrlsRequest(BaseModel):
    items: list[FromUrlsItem]


class FromUrlRequest(BaseModel):
    """Ingest 1 URL public — AI tự resolve theo nguồn (Google Drive, OneDrive,
    Dropbox, YouTube, generic HTTP). Cho link private (SharePoint nội bộ),
    BE phải resolve qua Microsoft Graph rồi đẩy `/from-urls` với headers token.
    """
    url: str
    filename: str | None = None      # nếu trống, AI đoán từ URL/Content-Disposition
    metadata: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] | None = None
    callback_url: str | None = None  # AI POST khi job done/failed


class JobCallbackPayload(BaseModel):
    """Payload AI POST sang `callback_url` khi job đạt trạng thái terminal.

    BE/FE expose 1 endpoint nhận POST này → push notification cho user
    (toast/email/Teams/DB), update document status, etc.
    """
    job_id: str
    job_type: str               # "document" | "video_file" | "youtube"
    filename: str
    status: str                 # "done" | "failed"
    chunks_added: int
    pages: int
    error: str | None
    duration_sec: float
    metadata: dict[str, Any] = Field(default_factory=dict)
