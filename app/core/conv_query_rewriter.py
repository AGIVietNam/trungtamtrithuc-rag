"""Rewrite follow-up queries thành standalone queries (Haiku).

Tiếng Việt có rất nhiều zero anaphora (câu lược chủ ngữ):
  "Còn tiền không?", "Làm sao đây?", "Cái đó thế nào?"
Những câu này embed ra vector gần như vô nghĩa ngữ cảnh.
Cần rewrite bằng Haiku + 2 turn gần nhất → câu đầy đủ chủ ngữ.

Nếu query đã dài/đầy đủ → không rewrite (tiết kiệm).
"""
from __future__ import annotations

import logging
import re

from app.core.claude_client import ClaudeClient
from app.config import CLAUDE_HAIKU_MODEL, CONV_REWRITE_MIN_LEN

logger = logging.getLogger(__name__)

# Anaphora / zero-anaphora markers tiếng Việt — query chứa các từ này thường
# cần resolve về danh từ cụ thể từ context. Query KHÔNG có markers + đủ thông
# tin nội dung thì tự đứng được → skip Haiku call (~1-3s save mỗi turn 2+).
# Microsoft RAG production guide khuyến nghị conditional rewrite: 50-70% turn
# có thể bỏ qua bước này.
_ANAPHORA_MARKERS = re.compile(
    r"(?:^|\W)("
    r"nó|hắn|họ|chúng|"
    r"cái\s+(?:đó|này|ấy|kia)|"
    r"vụ\s+(?:đó|này|ấy|kia)|"
    r"chỗ\s+(?:đó|này|ấy|kia)|"
    r"điều\s+(?:đó|này|ấy)|"
    r"(?:cô|anh|chị|ông|bà)\s+ấy|"
    r"vậy|thế|đó|đấy|ấy"
    r")(?:\W|$)",
    re.IGNORECASE | re.UNICODE,
)


def _has_anaphora(text: str) -> bool:
    return bool(_ANAPHORA_MARKERS.search(text))

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

    # Conditional rewrite: chỉ gọi Haiku khi query có dấu hiệu anaphora.
    # Câu ngắn nhưng đã đầy đủ chủ ngữ (vd "Doanh thu Q1 bao nhiêu?")
    # đứng độc lập được — embed thẳng, save 1 round-trip Haiku ~1-3s.
    if not _has_anaphora(query):
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
