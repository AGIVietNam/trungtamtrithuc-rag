from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, ChatResponse
from app.config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    VOYAGE_API_KEY, VOYAGE_MODEL, VOYAGE_DIM,
    QDRANT_URL, QDRANT_API_KEY,
    QDRANT_VMEDIA_URL, QDRANT_VMEDIA_API_KEY, VMEDIA_COLLECTIONS,
    COLLECTION_DOCS, COLLECTION_VIDEOS,
    TOP_K, RERANK_TOP_K,
    CONV_WINDOW_TURNS,
)
from app.core.claude_client import ClaudeClient
from app.core.voyage_embed import VoyageEmbedder
from app.core.qdrant_store import QdrantStore, VMediaReadOnlyStore
from app.core.session_memory import memory
from app.core.conv_memory import ConversationMemory
from app.core.conv_summarizer import summarize as summarize_conv
from app.rag.retriever import Retriever
from app.rag.reranker import CrossEncoderReranker
from app.rag.chain import RAGChain

logger = logging.getLogger(__name__)

router = APIRouter()

# Các marker báo hiệu bot "không biết / không có thông tin".
# Pair có assistant_msg chứa các marker này sẽ KHÔNG được upsert vào
# Qdrant để tránh feedback loop: câu hỏi tương tự về sau recall lại
# chính câu "không tìm thấy" → bot tin rồi lặp lại "không tìm thấy".
_NO_INFO_MARKERS = (
    "không tìm thấy thông tin",
    "không tìm thấy dữ liệu",
    "tôi không có thông tin",
    "tôi không biết",
    "không có trong cơ sở tri thức",
    "không có thông tin về",
)


def _is_no_info_answer(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _NO_INFO_MARKERS)

_chain: RAGChain | None = None
_conv_memory: ConversationMemory | None = None
_claude: ClaudeClient | None = None


def _get_claude() -> ClaudeClient:
    global _claude
    if _claude is None:
        _claude = ClaudeClient(api_key=ANTHROPIC_API_KEY, model=CLAUDE_MODEL)
    return _claude


def _get_conv_memory() -> ConversationMemory:
    global _conv_memory
    if _conv_memory is None:
        voyage = VoyageEmbedder(api_key=VOYAGE_API_KEY, model=VOYAGE_MODEL)
        _conv_memory = ConversationMemory(embedder=voyage)
    return _conv_memory


def _get_chain() -> RAGChain:
    global _chain
    if _chain is not None:
        return _chain

    voyage = VoyageEmbedder(api_key=VOYAGE_API_KEY, model=VOYAGE_MODEL)
    claude = _get_claude()

    qdrant_docs = QdrantStore(
        url=QDRANT_URL, api_key=QDRANT_API_KEY,
        collection=COLLECTION_DOCS, vector_size=VOYAGE_DIM,
    )
    qdrant_videos = QdrantStore(
        url=QDRANT_URL, api_key=QDRANT_API_KEY,
        collection=COLLECTION_VIDEOS, vector_size=VOYAGE_DIM,
    )
    vmedia_store = VMediaReadOnlyStore(
        url=QDRANT_VMEDIA_URL, vmedia_api_key=QDRANT_VMEDIA_API_KEY,
        collections=VMEDIA_COLLECTIONS,
    )

    retriever = Retriever(
        voyage=voyage,
        qdrant_docs=qdrant_docs,
        qdrant_videos=qdrant_videos,
        vmedia_store=vmedia_store,
    )
    reranker = CrossEncoderReranker()

    _chain = RAGChain(
        retriever=retriever,
        reranker=reranker,
        claude=claude,
        top_k=TOP_K,
        rerank_top_k=RERANK_TOP_K,
    )
    return _chain


def _post_turn_memory_update(
    session_id: str,
    user_id: str,
    user_msg: str,
    assistant_msg: str,
    domain: str,
) -> None:
    """Background task: trim sliding window + summarize rolled + upsert pair to Qdrant.

    KHÔNG bao giờ raise — mọi lỗi chỉ log.
    """
    try:
        turn_idx = memory.add_turn(session_id, user_msg, assistant_msg)

        # Rolling summary: pop overflow → summarize → update
        rolled = memory.pop_overflow(session_id, CONV_WINDOW_TURNS)
        if rolled:
            try:
                old_summary = memory.get_summary(session_id)
                new_summary = summarize_conv(_get_claude(), old_summary, rolled)
                if new_summary and new_summary != old_summary:
                    memory.set_summary(session_id, new_summary)
                    logger.info(
                        "session %s summary updated (rolled %d msgs, len=%d)",
                        session_id, len(rolled), len(new_summary),
                    )
            except Exception:
                logger.error("rolling summary failed", exc_info=True)

        # Vector recall upsert — skip pair khi bot trả "không tìm thấy"
        # để tránh feedback loop ô nhiễm memory.
        if _is_no_info_answer(assistant_msg):
            logger.info(
                "conv_memory skip upsert (no-info answer) session=%s turn=%d",
                session_id, turn_idx,
            )
        else:
            try:
                _get_conv_memory().upsert_pair(
                    user_id=user_id,
                    session_id=session_id,
                    turn_idx=turn_idx,
                    user_text=user_msg,
                    assistant_text=assistant_msg,
                    domain=domain or "mặc định",
                )
            except Exception:
                logger.error("conv_memory upsert failed", exc_info=True)
    except Exception:
        logger.error("_post_turn_memory_update failed", exc_info=True)


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    t0 = time.perf_counter()
    chain = _get_chain()
    t_chain = time.perf_counter()

    history = memory.get_history(request.session_id)
    summary = memory.get_summary(request.session_id)
    t_mem = time.perf_counter()

    domain = request.domain if request.domain and request.domain not in ("general", "mặc định") else None
    effective_user_id = (request.user_id or "").strip() or request.session_id

    try:
        result = chain.answer(
            query=request.message,
            history=history,
            expert_domain=domain,
            user_id=effective_user_id,
            session_id=request.session_id,
            summary=summary,
            conv_memory=_get_conv_memory(),
        )
    except Exception as exc:
        logger.exception("RAG chain error: %s", exc)
        logger.info(
            "POST /api/chat FAILED (session=%s): init=%.3fs mem=%.3fs chain=%.3fs total=%.3fs",
            request.session_id,
            t_chain - t0, t_mem - t_chain,
            time.perf_counter() - t_mem,
            time.perf_counter() - t0,
        )
        return ChatResponse(
            answer=f"Lỗi xử lý: {exc}",
            sources=[],
            session_id=request.session_id,
        )
    t_chain_done = time.perf_counter()

    # Schedule memory update AFTER response is returned
    background_tasks.add_task(
        _post_turn_memory_update,
        session_id=request.session_id,
        user_id=effective_user_id,
        user_msg=request.message,
        assistant_msg=result["answer"],
        domain=request.domain or "mặc định",
    )
    t_end = time.perf_counter()
    logger.info(
        "POST /api/chat steps (session=%s): init=%.3fs mem=%.3fs chain=%.3fs schedule_bg=%.3fs total=%.3fs (refused=%s)",
        request.session_id,
        t_chain - t0,
        t_mem - t_chain,
        t_chain_done - t_mem,
        t_end - t_chain_done,
        t_end - t0,
        result.get("refused", False),
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        session_id=request.session_id,
        suggested_questions=result.get("suggested_questions", []),
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest, background_tasks: BackgroundTasks
) -> StreamingResponse:
    """SSE streaming endpoint — emits `meta`, `delta`, `done`, `error` events."""
    t0 = time.perf_counter()
    chain = _get_chain()
    t_chain = time.perf_counter()
    history = memory.get_history(request.session_id)
    summary = memory.get_summary(request.session_id)
    t_mem = time.perf_counter()

    domain = (
        request.domain
        if request.domain and request.domain not in ("general", "mặc định")
        else None
    )
    effective_user_id = (request.user_id or "").strip() or request.session_id

    def event_generator() -> Iterator[str]:
        t_stream_start = time.perf_counter()
        t_first_delta: float | None = None
        final_answer = ""
        try:
            for event in chain.answer_stream(
                query=request.message,
                history=history,
                expert_domain=domain,
                user_id=effective_user_id,
                session_id=request.session_id,
                summary=summary,
                conv_memory=_get_conv_memory(),
            ):
                etype = event.get("type")
                if etype == "delta" and t_first_delta is None:
                    t_first_delta = time.perf_counter()
                if etype == "done":
                    final_answer = event.get("answer", "") or ""
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("RAG stream error: %s", exc)
            logger.info(
                "POST /api/chat/stream FAILED (session=%s): init=%.3fs mem=%.3fs stream=%.3fs total=%.3fs",
                request.session_id,
                t_chain - t0, t_mem - t_chain,
                time.perf_counter() - t_mem,
                time.perf_counter() - t0,
            )
            err = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            return

        t_stream_end = time.perf_counter()
        ttfb = (t_first_delta - t_stream_start) if t_first_delta else None
        logger.info(
            "POST /api/chat/stream steps (session=%s): init=%.3fs mem=%.3fs ttfb=%s stream=%.3fs total=%.3fs (answer=%d chars)",
            request.session_id,
            t_chain - t0,
            t_mem - t_chain,
            f"{ttfb:.3f}s" if ttfb is not None else "n/a",
            t_stream_end - t_stream_start,
            t_stream_end - t0,
            len(final_answer),
        )

        if final_answer:
            background_tasks.add_task(
                _post_turn_memory_update,
                session_id=request.session_id,
                user_id=effective_user_id,
                user_msg=request.message,
                assistant_msg=final_answer,
                domain=request.domain or "mặc định",
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------- GDPR / admin

@router.delete("/memory/user/{user_id}")
async def delete_user_memory(user_id: str) -> dict:
    """Xoá toàn bộ conversation memory của 1 user trong Qdrant."""
    ok = _get_conv_memory().delete_by_user(user_id)
    return {"status": "ok" if ok else "error", "user_id": user_id}


@router.delete("/memory/session/{session_id}")
async def delete_session_memory(session_id: str) -> dict:
    """Xoá 1 session: cả file JSON + các pair cùng session_id trong Qdrant."""
    memory.clear(session_id)
    _get_conv_memory().delete_by_session(session_id)
    return {"status": "ok", "session_id": session_id}
