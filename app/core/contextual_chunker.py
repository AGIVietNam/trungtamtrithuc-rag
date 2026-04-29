"""Contextual chunking theo Anthropic Contextual Retrieval (Sept 2024).

Idea: với mỗi chunk, gửi cả tài liệu + chunk đó cho Haiku → trả 50-100 token
mô tả "chunk này nói phần nào trong tài liệu, ngữ cảnh là gì". Prepend mô tả
này vào text trước khi embed/sparse-encode. Anthropic công bố: + Contextual
Embeddings → -35% fail rate; + Contextual BM25 → -49%; + reranker → -67%.

Tối ưu chi phí qua Anthropic Prompt Caching:
- Toàn bộ tài liệu đặt trong text block với ``cache_control: ephemeral``.
- Lần đầu: cache write (đắt 1.25x).
- N-1 chunks sau dùng cache (rẻ 0.1x).
- Tài liệu 50 chunks → cost ≈ 1.25 + 49*0.1 = 6.15 unit thay vì 50 unit.

Quan trọng — context KHÔNG đi vào ``payload.text``:
- ``payload.text`` giữ nguyên chunk gốc (Citations API trích lại đúng câu thật).
- Context lưu ở ``payload.context`` (tham khảo / debug).
- Embed/sparse encode dùng ``context + "\n\n" + chunk.text`` (concat).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app import config
from app.core.claude_client import ClaudeClient

logger = logging.getLogger(__name__)


# Length cap ngữ cảnh tài liệu inject vào prompt — Haiku 200k context dư sức,
# nhưng giới hạn 80k chars (~20k tokens) để control cost cache write.
# Doc dài hơn ngưỡng này: truncate, vẫn cache được, chunk ở phần cuối có thể
# kém ngữ cảnh — chấp nhận trade-off vì doc >80k chars hiếm trong corpus TDI.
_DOC_TEXT_CAP_CHARS: int = 80_000

# Cap output mỗi context — Anthropic recipe gợi ý 50-100 tokens.
_CONTEXT_MAX_TOKENS: int = 200

# System prompt — short, single instruction. Không cần persona dài.
_SYSTEM_PROMPT: str = (
    "Bạn là module bổ sung ngữ cảnh cho chunk tài liệu để tăng độ chính xác "
    "tìm kiếm. Trả lời bằng tiếng Việt tự nhiên, ngắn gọn."
)

# User prompt template — đặt tài liệu trong block CACHED, chunk ở block không cache.
_USER_DOC_TEMPLATE: str = (
    "<document>\n{doc_text}\n</document>"
)
_USER_CHUNK_TEMPLATE: str = (
    "Đây là chunk cần đặt ngữ cảnh:\n"
    "<chunk>\n{chunk_text}\n</chunk>\n\n"
    "Hãy viết 1-2 câu (tối đa 100 từ) đặt chunk này vào ngữ cảnh tổng thể "
    "của tài liệu, để cải thiện khả năng tìm kiếm.\n\n"
    "QUY TẮC:\n"
    "- Tiếng Việt tự nhiên, không markdown.\n"
    "- Không thêm fact mới ngoài tài liệu.\n"
    "- Không lặp lại nguyên văn chunk.\n"
    "- Nêu phần/chương/mục mà chunk thuộc về (nếu thấy được trong tài liệu).\n"
    "- Trả về CHỈ ngữ cảnh, không prefix, không giải thích."
)


@dataclass(frozen=True)
class ContextualChunk:
    """Chunk gốc + ngữ cảnh đã sinh ra.

    ``embed_text``: text dùng cho embedding/sparse encode (= context + "\\n\\n" + text).
    ``text``: chunk gốc — vào payload.text, dùng cho Citations API.
    ``context``: prefix ngữ cảnh — vào payload.context, dùng để debug.
    """

    text: str
    context: str
    embed_text: str


def _truncate_doc(doc_text: str) -> str:
    """Cắt tài liệu để vừa cap, giữ phần đầu (thường là TOC + intro)."""
    if len(doc_text) <= _DOC_TEXT_CAP_CHARS:
        return doc_text
    return doc_text[:_DOC_TEXT_CAP_CHARS] + "\n\n[...tài liệu bị cắt do quá dài...]"


def _generate_context(
    claude: ClaudeClient,
    doc_text: str,
    chunk_text: str,
) -> str:
    """Gọi Haiku 1 lần → trả ngữ cảnh ngắn cho 1 chunk.

    Doc text đặt trong text block CACHED (cache_control ephemeral) để N-1
    chunks sau dùng cache. Trả "" nếu Haiku fail/raise — caller quyết định
    có embed thuần (không context) hay raise.
    """
    user_content: list[dict] = [
        {
            "type": "text",
            "text": _USER_DOC_TEMPLATE.format(doc_text=_truncate_doc(doc_text)),
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": _USER_CHUNK_TEMPLATE.format(chunk_text=chunk_text),
        },
    ]
    try:
        result = claude.quick_text(
            system_prompt=_SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=_CONTEXT_MAX_TOKENS,
            model=config.CLAUDE_HAIKU_MODEL,
        ).strip()
        return result
    except Exception:
        logger.warning("Contextual chunking failed for chunk (len=%d)", len(chunk_text), exc_info=True)
        return ""


def add_contexts(
    claude: ClaudeClient,
    doc_text: str,
    chunk_texts: list[str],
) -> list[ContextualChunk]:
    """Sinh ngữ cảnh cho mọi chunk thuộc 1 tài liệu (sequential — caller batch).

    Sequential vì:
    1. Claude SDK threadsafe nhưng prompt caching dùng tốt nhất khi gọi tuần tự
       trong cửa sổ TTL 5 phút (cache write rồi cache hit liên tục).
    2. Anthropic free-tier rate limit dễ trip nếu fan-out parallel.

    Nếu disabled (env CONTEXTUAL_CHUNKING=0) → trả ContextualChunk với
    ``context=""`` và ``embed_text=text`` (no-op nhưng dùng API thống nhất).
    """
    if not chunk_texts:
        return []

    if not config.CONTEXTUAL_CHUNKING:
        return [
            ContextualChunk(text=t, context="", embed_text=t) for t in chunk_texts
        ]

    out: list[ContextualChunk] = []
    cache_hits = 0
    for i, ct in enumerate(chunk_texts):
        ctx = _generate_context(claude, doc_text, ct)
        if ctx:
            embed_text = f"{ctx}\n\n{ct}"
            cache_hits += 1
        else:
            embed_text = ct
        out.append(ContextualChunk(text=ct, context=ctx, embed_text=embed_text))
    logger.info(
        "contextual chunking: %d/%d chunks got context (doc_len=%d chars)",
        cache_hits, len(chunk_texts), len(doc_text),
    )
    return out
