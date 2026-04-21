"""Rewrite follow-up queries thành standalone queries (Haiku).

Tiếng Việt có rất nhiều zero anaphora (câu lược chủ ngữ):
  "Còn tiền không?", "Làm sao đây?", "Cái đó thế nào?"
Những câu này embed ra vector gần như vô nghĩa ngữ cảnh.
Cần rewrite bằng Haiku + 2 turn gần nhất → câu đầy đủ chủ ngữ.

Nếu query đã dài/đầy đủ → không rewrite (tiết kiệm).
"""
from __future__ import annotations

import logging

from app.core.claude_client import ClaudeClient
from app.config import CLAUDE_HAIKU_MODEL, CONV_REWRITE_MIN_LEN

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Bạn là module rewrite câu hỏi tiếng Việt thành câu đầy đủ (standalone).\n\n"
    "Nhận:\n"
    "  - 2 lượt hội thoại gần nhất (context)\n"
    "  - Câu hỏi mới của user (có thể lược chủ ngữ / dùng đại từ 'cái đó', 'vụ kia'...)\n\n"
    "Nhiệm vụ: viết lại câu hỏi mới để ĐỨNG MỘT MÌNH vẫn hiểu được, \n"
    "thay đại từ / chỗ lược bằng danh từ cụ thể từ context.\n\n"
    "QUY TẮC:\n"
    "- Tiếng Việt tự nhiên.\n"
    "- Giữ NGUYÊN ý định của user, không đoán thêm.\n"
    "- Nếu câu hỏi đã đầy đủ rồi, trả về nguyên văn.\n"
    "- Trả về CHỈ câu đã rewrite, không giải thích, không prefix."
)


def _format_recent(recent_turns: list[dict], limit: int = 4) -> str:
    """Format last few messages (up to limit)."""
    if not recent_turns:
        return "(chưa có lượt nào trước)"
    last = recent_turns[-limit:]
    lines = []
    for t in last:
        role = "USER" if t.get("role") == "user" else "BOT"
        lines.append(f"{role}: {t.get('content', '').strip()}")
    return "\n".join(lines)


def rewrite(
    claude: ClaudeClient,
    query: str,
    recent_turns: list[dict],
) -> str:
    """Rewrite query thành standalone. Không có context → trả nguyên."""
    query = query.strip()
    if not query:
        return query

    # Không có history → không cần rewrite
    if not recent_turns:
        return query

    # Query đã đủ dài → có khả năng đã tự chứa đủ ngữ cảnh
    if len(query) >= CONV_REWRITE_MIN_LEN:
        return query

    try:
        user_content = (
            f"Context (2 lượt gần nhất):\n{_format_recent(recent_turns)}\n\n"
            f"Câu hỏi mới: {query}\n\n"
            f"Câu rewrite:"
        )
        rewritten = claude.quick_text(
            system_prompt=_SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=150,
            model=CLAUDE_HAIKU_MODEL,
        ).strip()

        # Guardrail: nếu rewrite quá dài, giữ nguyên query gốc
        if len(rewritten) > len(query) * 8 or not rewritten:
            return query

        logger.info("query rewritten: %r → %r", query, rewritten)
        return rewritten
    except Exception:
        logger.warning("query rewrite failed, using original", exc_info=True)
        return query
