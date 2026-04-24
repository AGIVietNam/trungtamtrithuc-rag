"""Rewrite follow-up / short queries thành standalone queries (Haiku).

2 chiến lược rewrite — trigger độc lập, cùng dùng Haiku:

1. **Anaphora rewrite** (cần history): câu lược chủ ngữ / dùng đại từ
   ("Còn tiền không?", "Cái đó thế nào?") → resolve về danh từ cụ thể từ
   2 turn gần nhất.

2. **Short-query expansion** (không cần history): câu rất ngắn, thiếu
   interrogative intent ("bài này chill phết", "báo cáo Q1", "tên sếp") →
   expand thành câu hỏi đầy đủ để embed khớp hơn với document chunks.
   Fix case dense embedding yếu với title/keyword match query.
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

# Interrogative / question markers tiếng Việt. Query có ít nhất 1 marker →
# đã tự mang intent hỏi → embedding sẽ khớp chunk dạng "câu trả lời" tốt hơn.
# Query thiếu HẾT các marker này + lại ngắn → có thể là statement/title-match
# ("bài này chill phết") → cần expand để gần với intent-based document chunks.
_QUESTION_MARKERS = re.compile(
    r"(?:^|\W)("
    r"\?|"
    r"ai|gì|nào|sao|đâu|"
    r"bao\s*(?:nhiêu|lâu|giờ)|"
    r"khi\s+nào|lúc\s+nào|ở\s+đâu|"
    r"thế\s+nào|như\s+thế\s+nào|ra\s+sao|làm\s+sao|"
    r"có\s+.+?\s+(?:không|chưa|chăng)|"
    r"là\s+gì|là\s+ai|"
    r"phải\s+(?:không|chăng)|"
    r"vì\s+sao|tại\s+sao|do\s+đâu|"
    r"bao\s+gồm|gồm\s+(?:gì|những\s+gì)|"
    r"có\s+(?:những|mấy)\s+"
    r")(?:\W|$)",
    re.IGNORECASE | re.UNICODE,
)

# Query ngắn dưới ngưỡng này (token count) + không có question marker →
# trigger short-query expansion. 5 = "bài này chill phết" (4 tokens) +
# biên an toàn cho "báo cáo tài chính Q1" (5 tokens).
_SHORT_QUERY_MAX_TOKENS = 5


def _has_anaphora(text: str) -> bool:
    return bool(_ANAPHORA_MARKERS.search(text))


def _has_question_intent(text: str) -> bool:
    return bool(_QUESTION_MARKERS.search(text))


def _is_short_query(text: str) -> bool:
    return len(text.split()) <= _SHORT_QUERY_MAX_TOKENS


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

_EXPAND_SYSTEM_PROMPT = (
    "Bạn là module mở rộng câu truy vấn tiếng Việt NGẮN thành câu hỏi đầy đủ.\n\n"
    "Câu truy vấn người dùng rất ngắn (ví dụ: tên bài hát, tên tài liệu, mã sản phẩm, "
    "tên người, cụm từ khoá) — thiếu intent hỏi rõ ràng nên embedding yếu, khó khớp "
    "với chunks tài liệu dạng câu đầy đủ.\n\n"
    "Nhiệm vụ: viết lại thành 1 câu hỏi tiếng Việt rõ intent, GIỮ NGUYÊN mọi keyword "
    "gốc, chỉ bổ sung từ khoá intent (của ai / là gì / như thế nào / nội dung / thông tin về...).\n\n"
    "VÍ DỤ:\n"
    "- 'bài này chill phết' → 'Thông tin về bài hát \"Bài này chill phết\" là gì?'\n"
    "- 'báo cáo Q1' → 'Nội dung báo cáo Q1 như thế nào?'\n"
    "- 'authen teams' → 'Authen Teams là gì và dùng như thế nào?'\n"
    "- 'quy trình onboard' → 'Quy trình onboard nhân viên mới như thế nào?'\n\n"
    "QUY TẮC:\n"
    "- KHÔNG đổi keyword gốc, KHÔNG dịch, KHÔNG đoán nội dung.\n"
    "- Không thêm tên công ty, domain, hay ngữ cảnh không có trong câu gốc.\n"
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


def _call_haiku(
    claude: ClaudeClient,
    system_prompt: str,
    user_content: str,
    query: str,
    mode: str,
) -> str:
    """Gọi Haiku + guardrail chiều dài. Fail → trả query gốc."""
    try:
        rewritten = claude.quick_text(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=150,
            model=CLAUDE_HAIKU_MODEL,
        ).strip()
        if not rewritten or len(rewritten) > len(query) * 8:
            return query
        logger.info("query rewritten (%s): %r → %r", mode, query, rewritten)
        return rewritten
    except Exception:
        logger.warning("query rewrite failed (%s), using original", mode, exc_info=True)
        return query


def rewrite(
    claude: ClaudeClient,
    query: str,
    recent_turns: list[dict],
) -> str:
    """Rewrite query → standalone.

    Thứ tự trigger:
      1. Anaphora (có history) → resolve đại từ từ context.
      2. Short query thiếu intent → expand thành câu hỏi đầy đủ.
      3. Query đủ dài / đủ intent → giữ nguyên.
    """
    query = query.strip()
    if not query:
        return query

    # Query đã đủ dài → tự chứa đủ ngữ cảnh, không rewrite.
    if len(query) >= CONV_REWRITE_MIN_LEN:
        return query

    # 1. Anaphora rewrite — cần history để resolve.
    if recent_turns and _has_anaphora(query):
        user_content = (
            f"Context (2 lượt gần nhất):\n{_format_recent(recent_turns)}\n\n"
            f"Câu hỏi mới: {query}\n\n"
            f"Câu rewrite:"
        )
        return _call_haiku(claude, _SYSTEM_PROMPT, user_content, query, mode="anaphora")

    # 2. Short-query expansion — không cần history. Fix dense embedding yếu
    # với query ngắn thiếu intent (title-match, keyword, tên riêng).
    if _is_short_query(query) and not _has_question_intent(query):
        user_content = f"Câu truy vấn ngắn: {query}\n\nCâu hỏi mở rộng:"
        return _call_haiku(claude, _EXPAND_SYSTEM_PROMPT, user_content, query, mode="expand")

    return query
