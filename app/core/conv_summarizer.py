"""Rolling summary cho session hiện tại bằng Claude Haiku.

Khi sliding window vượt ngưỡng, các turn cũ bị rớt khỏi window sẽ
được tóm tắt thành 1 đoạn (~400 tokens) để giữ ngữ cảnh.
Tóm tắt được incremental: new = summarize(old_summary + rolled_turns).
"""
from __future__ import annotations

import logging
from typing import Iterable

from app.core.claude_client import ClaudeClient
from app.config import CLAUDE_HAIKU_MODEL, CONV_SUMMARY_MAX_TOKENS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Bạn là module tóm tắt hội thoại tiếng Việt.\n"
    "Nhận vào:\n"
    "  (A) Tóm tắt trước đó của cuộc trò chuyện (có thể trống)\n"
    "  (B) Các lượt hội thoại mới đã diễn ra sau đó\n\n"
    "Nhiệm vụ: tạo ra tóm tắt MỚI gộp (A) + (B) thành 1 đoạn ≤ 400 từ.\n\n"
    "QUY TẮC:\n"
    "- Tiếng Việt, súc tích, giữ nguyên sự kiện, tên, số liệu, quyết định.\n"
    "- Ưu tiên thông tin MỚI (B) nếu mâu thuẫn với (A).\n"
    "- Không bịa, không suy đoán.\n"
    "- Không mở đầu bằng 'Tóm tắt:', trả về NGUYÊN văn bản tóm tắt.\n"
    "- Bỏ qua câu chào, cảm ơn, câu rỗng nghĩa."
)


def _format_turns(turns: Iterable[dict]) -> str:
    lines: list[str] = []
    for t in turns:
        role = "USER" if t.get("role") == "user" else "BOT"
        lines.append(f"{role}: {t.get('content', '').strip()}")
    return "\n".join(lines)


def summarize(
    claude: ClaudeClient,
    old_summary: str,
    rolled_turns: list[dict],
) -> str:
    """Summarize old_summary + rolled_turns → new summary. Graceful on error."""
    if not rolled_turns:
        return old_summary

    try:
        user_content = (
            f"(A) Tóm tắt trước đó:\n{old_summary or '(chưa có)'}\n\n"
            f"(B) Các lượt hội thoại mới:\n{_format_turns(rolled_turns)}"
        )
        result = claude.quick_text(
            system_prompt=_SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=CONV_SUMMARY_MAX_TOKENS + 50,
            model=CLAUDE_HAIKU_MODEL,
        )
        return result.strip()
    except Exception:
        logger.error("conv_summarizer failed, keeping old summary", exc_info=True)
        return old_summary
