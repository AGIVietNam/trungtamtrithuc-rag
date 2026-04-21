from __future__ import annotations

import concurrent.futures
import re
from collections.abc import Iterator
from typing import Any

from app.core.claude_client import ClaudeClient
from app.core.conv_memory import ConversationMemory
from app.core.conv_query_rewriter import rewrite as rewrite_query
from app.rag.retriever import Retriever
from app.rag.reranker import CrossEncoderReranker
from app.rag.prompt_builder import (
    build_system_prompt,
    build_context_block,
    build_conversation_block,
)

_SUGGESTION_PATTERN = re.compile(
    r"\n*---GỢI Ý---\s*\n(.*)",
    re.DOTALL,
)


def _extract_suggestions(answer: str) -> tuple[str, list[str]]:
    """Split answer into (clean_answer, suggested_questions)."""
    m = _SUGGESTION_PATTERN.search(answer)
    if not m:
        return answer, []
    clean = answer[: m.start()].rstrip()
    raw = m.group(1).strip()
    questions = [
        re.sub(r"^\d+\.\s*", "", line).strip()
        for line in raw.splitlines()
        if line.strip()
    ]
    return clean, [q for q in questions if q]


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

        # --- 1. Query rewrite (Haiku) để xử lý đại từ/zero anaphora tiếng Việt
        search_query = query
        if conv_memory is not None and user_id and history:
            try:
                search_query = rewrite_query(self.claude, query, history)
            except Exception:
                search_query = query

        # --- 2. Parallel: document retrieval + conversation recall
        recall_pairs: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_docs = ex.submit(
                self.retriever.retrieve,
                search_query, self.top_k, sources_filter, expert_domain,
            )
            f_conv = None
            if conv_memory is not None and user_id:
                f_conv = ex.submit(
                    conv_memory.retrieve, user_id, search_query, session_id,
                )
            hits = f_docs.result()
            if f_conv is not None:
                try:
                    recall_pairs = f_conv.result()
                except Exception:
                    recall_pairs = []

        hits = self.reranker.rerank(search_query, hits, top_k=self.rerank_top_k)

        # --- 3. Build prompt: system + conversation_block + doc context
        system_prompt = build_system_prompt(expert_domain)
        conv_block = build_conversation_block(summary, recall_pairs)
        if conv_block:
            system_prompt = system_prompt + "\n\n" + conv_block

        context_block, source_mapping = build_context_block(hits)

        # Messages: sliding window + query GỐC (không rewritten) để bot đáp đúng ý user
        messages = list(history) + [{"role": "user", "content": query}]

        answer_text = self.claude.generate(
            system_prompt=system_prompt,
            context_block=context_block,
            messages=messages,
        )

        clean_answer, suggested_questions = _extract_suggestions(answer_text)
        has_sources = "Nguồn:" in clean_answer or "nguồn:" in clean_answer.lower()

        top_score = hits[0].score if hits else 0.0
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

        search_query = query
        if conv_memory is not None and user_id and history:
            try:
                search_query = rewrite_query(self.claude, query, history)
            except Exception:
                search_query = query

        recall_pairs: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_docs = ex.submit(
                self.retriever.retrieve,
                search_query, self.top_k, sources_filter, expert_domain,
            )
            f_conv = None
            if conv_memory is not None and user_id:
                f_conv = ex.submit(
                    conv_memory.retrieve, user_id, search_query, session_id,
                )
            hits = f_docs.result()
            if f_conv is not None:
                try:
                    recall_pairs = f_conv.result()
                except Exception:
                    recall_pairs = []

        hits = self.reranker.rerank(search_query, hits, top_k=self.rerank_top_k)

        system_prompt = build_system_prompt(expert_domain)
        conv_block = build_conversation_block(summary, recall_pairs)
        if conv_block:
            system_prompt = system_prompt + "\n\n" + conv_block

        context_block, source_mapping = build_context_block(hits)
        messages = list(history) + [{"role": "user", "content": query}]

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
            context_block=context_block,
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
