from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from collections.abc import Iterator
from typing import Any

from app.core.claude_client import ClaudeClient
from app.core.conv_memory import ConversationMemory
from app.core.conv_query_rewriter import rewrite as rewrite_query
from app.rag.retriever import Retriever
from app.rag.reranker import CrossEncoderReranker
from app.rag.prompt_builder import (
    build_system_prompt,
    build_documents_block,
    build_sources_mapping,
    build_conversation_block,
    build_user_turn,
)

logger = logging.getLogger(__name__)

_SUGGESTION_PATTERN = re.compile(
    # Chấp nhận các biến thể marker mà Opus/Sonnet có thể đẻ ra khi không
    # tuân thủ literal 100%: "---GỢI Ý---", "--- Gợi ý ---", "**Gợi ý:**",
    # "## Gợi ý", "Gợi ý câu hỏi:"... Luôn yêu cầu marker đứng riêng 1 dòng
    # (có newline trước + sau) để tránh false-positive với "gợi ý" trong body.
    r"\n[ \t]*"                                    # đầu dòng (newline + indent)
    r"[\*#\-=_|: \t]*"                             # decorator mở (tuỳ chọn)
    r"G[ỢợOo][Ii][ \t]*[ÝýYy]"                    # "Gợi ý" / "GỢI Ý" (diacritic flex)
    r"(?:[ \t]+(?:c[âấ]u[ \t]*h[ỏỎ]i|tiếp[ \t]*theo|theo[ \t]*dõi))?"  # optional suffix
    r"[\*#\-=_|:\. \t]*"                           # decorator đóng (tuỳ chọn)
    r"\n+"
    r"(\s*\d+[\.\)].*)",                           # content phải bắt đầu "1. " hoặc "1) "
    re.IGNORECASE | re.DOTALL,
)

# Fallback: LLM quên hẳn marker nhưng vẫn liệt kê 2-5 câu hỏi đánh số ở cuối.
# Yêu cầu: đứng sau dòng trống + là khối cuối cùng của answer.
_TRAILING_NUMBERED_PATTERN = re.compile(
    r"\n\s*\n((?:[ \t]*\d+[\.\)][ \t]+[^\n]+(?:\n|$)){2,5})\s*$",
    re.DOTALL,
)

# Ngưỡng top rerank score tối thiểu để coi là có context trả lời.
# Dưới ngưỡng → trả refusal cứng (không gọi Claude, tránh hallucinate từ training data).
# BGE reranker score: >0.5 rất liên quan; 0.0-0.5 mờ; <0 không liên quan.
# 0.18 (hạ từ 0.25) — thả lỏng cho query overview/liệt kê match được một chunk
# cụ thể; <refusal_protocol> trong system prompt vẫn là lớp chặn cuối nếu doc
# thực sự không nói tới chủ đề.
_MIN_CONFIDENCE_TO_ANSWER: float = 0.05

_REFUSAL_TEMPLATE = (
    "Tài liệu TDI hiện chưa có thông tin về câu hỏi này.\n\n"
    "Bạn có thể:\n"
    "- Bổ sung tài liệu liên quan qua trang nạp dữ liệu.\n"
    "- Thử đổi sang lĩnh vực 'Tất cả lĩnh vực' để mở rộng tìm kiếm.\n"
    "- Diễn đạt lại câu hỏi với từ khoá cụ thể hơn."
)


def _should_refuse(hits: list) -> bool:
    """True khi không có hit nào hoặc top score dưới ngưỡng tin cậy."""
    if not hits:
        return True
    top = hits[0].score if hits[0].score is not None else 0.0
    return top < _MIN_CONFIDENCE_TO_ANSWER


def _parse_numbered_questions(raw: str) -> list[str]:
    """Trích câu hỏi từ block kiểu "1. ...\\n2. ...\\n3. ..."."""
    questions: list[str] = []
    for line in raw.splitlines():
        stripped = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        # Strip markdown decorators (bold, italic)
        stripped = re.sub(r"^[\*_`]+|[\*_`]+$", "", stripped).strip()
        if stripped:
            questions.append(stripped)
    return questions


def _extract_suggestions(answer: str) -> tuple[str, list[str]]:
    """Split answer into (clean_answer, suggested_questions).

    Thử 2 chiến lược theo thứ tự:
      1. Marker "GỢI Ý" (nhiều biến thể) + numbered list phía sau.
      2. Fallback: trailing numbered block 2-5 dòng ở cuối answer.

    Nếu cả 2 fail → trả về (answer, []) — LLM không emit gợi ý.
    """
    m = _SUGGESTION_PATTERN.search(answer)
    if m:
        clean = answer[: m.start()].rstrip()
        questions = _parse_numbered_questions(m.group(1))
        if questions:
            logger.debug("suggestions extracted via marker: %d items", len(questions))
            return clean, questions

    # Fallback: không có marker nhưng có block đánh số ở cuối.
    m2 = _TRAILING_NUMBERED_PATTERN.search(answer)
    if m2:
        questions = _parse_numbered_questions(m2.group(1))
        if len(questions) >= 2:
            clean = answer[: m2.start()].rstrip()
            logger.info(
                "suggestions extracted via trailing-numbered fallback: %d items "
                "(LLM bỏ marker — xem lại prompt nếu lặp lại)",
                len(questions),
            )
            return clean, questions

    return answer, []


def _confidence(top_score: float) -> str:
    if top_score >= 0.7:
        return "high"
    if top_score >= 0.4:
        return "medium"
    return "low"


class RAGChain:
    def __init__(
        self,
        retriever: Retriever,
        reranker: CrossEncoderReranker,
        claude: ClaudeClient,
        top_k: int = 10,
        rerank_top_k: int = 3,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.claude = claude
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def answer(
        self,
        query: str,
        history: list[dict] | None = None,
        expert_domain: str | None = None,
        sources_filter: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        summary: str = "",
        conv_memory: ConversationMemory | None = None,
    ) -> dict[str, Any]:
        history = history or []
        t0 = time.perf_counter()

        # --- 1. Query rewrite (Haiku) — anaphora-conditional, skip nếu query
        # tự đứng được (xem conv_query_rewriter._has_anaphora). Cắt 1-3s/turn.
        search_query = query
        if conv_memory is not None and user_id and history:
            try:
                search_query = rewrite_query(self.claude, query, history)
            except Exception:
                search_query = query
        t_rewrite = time.perf_counter()

        # --- 2. Embed query 1 LẦN, reuse cho cả retriever + conv_memory.
        # Trước fix này, mỗi turn gọi Voyage 2 lần cho cùng 1 query → góp phần
        # đẩy free-tier 3 RPM vào 429 → backoff 25s × N retry.
        query_vec = self.retriever.voyage.embed_query(search_query)
        t_embed = time.perf_counter()

        # --- 3. Parallel: document retrieval + conversation recall
        # Recall gating: skip Qdrant call cho greeting/yes-no — sẽ chỉ recall
        # ra pair noise lạc đề và tốn 1 round-trip Qdrant.
        recall_pairs: list[dict] = []
        do_recall = (
            conv_memory is not None
            and bool(user_id)
            and not ConversationMemory.should_skip_recall(search_query)
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_docs = ex.submit(
                self.retriever.retrieve,
                search_query, self.top_k, sources_filter, expert_domain,
                query_vec=query_vec,
            )
            f_conv = None
            if do_recall:
                f_conv = ex.submit(
                    conv_memory.retrieve,
                    user_id, search_query, session_id,
                    query_vec=query_vec,
                )
            hits = f_docs.result()
            if f_conv is not None:
                try:
                    recall_pairs = f_conv.result()
                except Exception:
                    recall_pairs = []
        t_retrieve = time.perf_counter()

        hits = self.reranker.rerank(search_query, hits, top_k=self.rerank_top_k)
        t_rerank = time.perf_counter()

        # --- 3b. Guard: không đủ context → refuse cứng, KHÔNG gọi Claude
        # (tránh hallucinate từ training data).
        if _should_refuse(hits):
            logger.info(
                "RAG refused: rewrite=%.2fs embed=%.2fs retrieve=%.2fs "
                "rerank=%.2fs total=%.2fs (recall=%s)",
                t_rewrite - t0, t_embed - t_rewrite, t_retrieve - t_embed,
                t_rerank - t_retrieve, t_rerank - t0,
                "yes" if do_recall else "skipped",
            )
            return {
                "answer": _REFUSAL_TEMPLATE,
                "sources": [],
                "confidence": "low",
                "suggested_questions": [],
                "rewritten_query": search_query if search_query != query else None,
                "recall_count": len(recall_pairs),
                "refused": True,
            }

        # --- 4. Build prompt:
        # - system: persona + _BASE_RULES (STABLE → cache hit mọi turn sau lượt đầu)
        # - user turn cuối: <retrieved_documents> + <user_context>/<session_summary>
        #   + task reminder + query gốc (docs ở top, query ở bottom — long-context tip).
        system_prompt = build_system_prompt(expert_domain)
        docs_block = build_documents_block(hits)
        conv_block = build_conversation_block(summary, recall_pairs)
        source_mapping = build_sources_mapping(hits)

        user_turn = build_user_turn(query, docs_block, conv_block)
        messages = list(history) + [{"role": "user", "content": user_turn}]

        answer_text = self.claude.generate(
            system_prompt=system_prompt,
            messages=messages,
        )
        t_claude = time.perf_counter()

        clean_answer, suggested_questions = _extract_suggestions(answer_text)
        has_sources = "Nguồn:" in clean_answer or "nguồn:" in clean_answer.lower()

        top_score = hits[0].score if hits else 0.0
        logger.info(
            "RAG done: rewrite=%.2fs embed=%.2fs retrieve=%.2fs rerank=%.2fs "
            "claude=%.2fs total=%.2fs (recall=%s, hits=%d)",
            t_rewrite - t0, t_embed - t_rewrite, t_retrieve - t_embed,
            t_rerank - t_retrieve, t_claude - t_rerank, t_claude - t0,
            "yes" if do_recall else "skipped", len(hits),
        )
        return {
            "answer": clean_answer,
            "sources": source_mapping if has_sources else [],
            "confidence": _confidence(top_score),
            "suggested_questions": suggested_questions,
            "rewritten_query": search_query if search_query != query else None,
            "recall_count": len(recall_pairs),
        }

    def answer_stream(
        self,
        query: str,
        history: list[dict] | None = None,
        expert_domain: str | None = None,
        sources_filter: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        summary: str = "",
        conv_memory: ConversationMemory | None = None,
    ) -> Iterator[dict[str, Any]]:
        history = history or []
        t0 = time.perf_counter()

        search_query = query
        if conv_memory is not None and user_id and history:
            try:
                search_query = rewrite_query(self.claude, query, history)
            except Exception:
                search_query = query
        t_rewrite = time.perf_counter()

        # Embed query 1 lần, reuse cho retriever + conv_memory.
        query_vec = self.retriever.voyage.embed_query(search_query)
        t_embed = time.perf_counter()

        recall_pairs: list[dict] = []
        do_recall = (
            conv_memory is not None
            and bool(user_id)
            and not ConversationMemory.should_skip_recall(search_query)
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_docs = ex.submit(
                self.retriever.retrieve,
                search_query, self.top_k, sources_filter, expert_domain,
                query_vec=query_vec,
            )
            f_conv = None
            if do_recall:
                f_conv = ex.submit(
                    conv_memory.retrieve,
                    user_id, search_query, session_id,
                    query_vec=query_vec,
                )
            hits = f_docs.result()
            if f_conv is not None:
                try:
                    recall_pairs = f_conv.result()
                except Exception:
                    recall_pairs = []
        t_retrieve = time.perf_counter()

        hits = self.reranker.rerank(search_query, hits, top_k=self.rerank_top_k)
        t_rerank = time.perf_counter()
        logger.info(
            "RAG stream pre-llm: rewrite=%.2fs embed=%.2fs retrieve=%.2fs "
            "rerank=%.2fs total=%.2fs (recall=%s, hits=%d)",
            t_rewrite - t0, t_embed - t_rewrite, t_retrieve - t_embed,
            t_rerank - t_retrieve, t_rerank - t0,
            "yes" if do_recall else "skipped", len(hits),
        )

        # Guard: không đủ context → emit refusal events và dừng, không gọi Claude.
        if _should_refuse(hits):
            yield {
                "type": "meta", "confidence": "low",
                "rewritten_query": search_query if search_query != query else None,
                "recall_count": len(recall_pairs),
                "refused": True,
            }
            yield {"type": "delta", "text": _REFUSAL_TEMPLATE}
            yield {
                "type": "done",
                "answer": _REFUSAL_TEMPLATE,
                "sources": [],
                "suggested_questions": [],
            }
            return

        system_prompt = build_system_prompt(expert_domain)
        docs_block = build_documents_block(hits)
        conv_block = build_conversation_block(summary, recall_pairs)
        source_mapping = build_sources_mapping(hits)

        user_turn = build_user_turn(query, docs_block, conv_block)
        messages = list(history) + [{"role": "user", "content": user_turn}]

        top_score = hits[0].score if hits else 0.0
        yield {
            "type": "meta",
            "confidence": _confidence(top_score),
            "rewritten_query": search_query if search_query != query else None,
            "recall_count": len(recall_pairs),
        }

        buffer_parts: list[str] = []
        for chunk in self.claude.generate_stream(
            system_prompt=system_prompt,
            messages=messages,
        ):
            buffer_parts.append(chunk)
            yield {"type": "delta", "text": chunk}

        full_text = "".join(buffer_parts)
        clean_answer, suggested_questions = _extract_suggestions(full_text)
        has_sources = "Nguồn:" in clean_answer or "nguồn:" in clean_answer.lower()

        yield {
            "type": "done",
            "answer": clean_answer,
            "sources": source_mapping if has_sources else [],
            "suggested_questions": suggested_questions,
        }
