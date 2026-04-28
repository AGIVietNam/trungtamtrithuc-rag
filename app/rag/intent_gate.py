"""Intent gate — phân loại pre-retrieval để skip Claude/faithfulness cho meta-question.

Bug context: faithfulness gate (chain.py: _is_hallucinated_uncited / Layer 2)
replace câu chào dài bằng _REFUSAL_TEMPLATE → user thấy stream "Xin chào..."
rồi done event mang refusal. Root cause: greeting/identity/capability KHÔNG
retrieval-able, Claude tự nhiên trả lời >100 chars + citations==0 → trip
faithfulness judge.

Fix: phân loại intent bằng regex pre-retrieval. Meta-question → canned
response, KHÔNG gọi Claude / retrieval / faithfulness. Deterministic, ~0ms
cost, không tốn API call.

Disable qua env INTENT_GATE=0 (debug / A/B test).
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


# --- Intent constants
INTENT_GREETING = "greeting"
INTENT_IDENTITY_BOT = "identity_bot"
INTENT_IDENTITY_USER = "identity_user"
INTENT_CAPABILITY = "capability"
INTENT_THANKS = "thanks"
INTENT_GOODBYE = "goodbye"
INTENT_RAG = "rag"


# --- Regex patterns
# Anchor `^...$` cuối cùng — không match câu RAG dài có chứa từ khoá greeting
# (vd "chào bạn, cho tôi hỏi quy trình BIM" → INTENT_RAG, không phải GREETING).
# Allow trailing punctuation/từ thân mật ngắn (bạn/nhé/ạ) để bắt biến thể tự nhiên.
_GREETING_RE = re.compile(
    r"^\s*(xin\s+chào|chào|hello|hi|hey|halo)"
    r"(\s+(bạn|ai|mọi\s+người|cả\s+nhà|ad|admin))?"
    r"[\s.!?,]*$",
    re.IGNORECASE,
)

_IDENTITY_BOT_RE = re.compile(
    r"^\s*(bạn|cậu|em|anh|chị|mày|you)\s+"
    r"(là\s+ai|tên\s+(là\s+)?gì|là\s+gì|tên\s+là\s+gì)"
    r"[\s.!?,]*\??$",
    re.IGNORECASE,
)

_IDENTITY_USER_RE = re.compile(
    r"^\s*(tôi|mình|em|tao|tớ|i)\s+"
    r"(là\s+ai|tên\s+(là\s+)?gì|là\s+người\s+nào)"
    r"[\s.!?,]*\??$",
    re.IGNORECASE,
)

_CAPABILITY_RE = re.compile(
    r"^\s*(bạn|cậu|em|mày)\s+("
    r"làm\s+(được\s+)?(gì|những\s+gì|được\s+những\s+gì)|"
    r"giúp\s+(được\s+)?(gì|những\s+gì|tôi\s+(được\s+)?gì|được\s+tôi)|"
    r"có\s+thể\s+(làm\s+(gì|những\s+gì)?|giúp\s+(gì|tôi\s+gì)?)|"
    r"biết\s+(làm\s+)?gì"
    r")[\s.!?,]*\??$",
    re.IGNORECASE,
)

_THANKS_RE = re.compile(
    r"^\s*(cảm\s+ơn|cám\s+ơn|thanks?|thx|tks|ty)"
    r"(\s+(bạn|nhiều|nha|nhé|ạ|you))?"
    r"[\s.!?,]*$",
    re.IGNORECASE,
)

_GOODBYE_RE = re.compile(
    r"^\s*(tạm\s+biệt|bye|goodbye|gặp\s+lại\s+sau|hẹn\s+gặp\s+lại)"
    r"(\s+(bạn|nhé|ạ))?"
    r"[\s.!?,]*$",
    re.IGNORECASE,
)


def _is_enabled() -> bool:
    """Intent gate có bật không (default ON). Env INTENT_GATE=0 → tắt."""
    val = (os.getenv("INTENT_GATE", "1") or "1").strip().lower()
    return val not in ("0", "false", "no", "off", "")


def classify_intent(query: str) -> str:
    """Phân loại intent từ raw query.

    Trả 1 trong INTENT_* constants. INTENT_RAG là default — đi qua retrieval
    pipeline bình thường.

    Order matters: identity/capability check TRƯỚC greeting để tránh case
    "chào bạn là ai" bị match GREETING trước khi check IDENTITY_BOT.
    """
    if not _is_enabled():
        return INTENT_RAG
    q = (query or "").strip()
    if not q:
        return INTENT_RAG

    if _IDENTITY_BOT_RE.match(q):
        return INTENT_IDENTITY_BOT
    if _IDENTITY_USER_RE.match(q):
        return INTENT_IDENTITY_USER
    if _CAPABILITY_RE.match(q):
        return INTENT_CAPABILITY
    if _THANKS_RE.match(q):
        return INTENT_THANKS
    if _GOODBYE_RE.match(q):
        return INTENT_GOODBYE
    if _GREETING_RE.match(q):
        return INTENT_GREETING
    return INTENT_RAG


# --- Canned responses
# Tone phải match DOMAIN_PERSONAS["mac_dinh"]: "Trợ lý đọc tài liệu của TDI Group".
# Lý do dùng canned thay vì gọi Claude: deterministic, 0 latency, 0 API cost,
# 0 risk hallucinate, FE thấy stream và done KHỚP nhau (không còn flap UX).
_CANNED: dict[str, str] = {
    INTENT_GREETING: (
        "Chào bạn! Tôi là Trợ lý đọc tài liệu của TDI Group. "
        "Bạn cần tra cứu thông tin gì về dự án, quy trình hoặc tài liệu nội bộ TDI hôm nay?"
    ),
    INTENT_IDENTITY_BOT: (
        "Tôi là Trợ lý đọc tài liệu của TDI Group — chuyên trả lời câu hỏi dựa "
        "trên tài liệu, quy trình và dự án nội bộ đã được nạp vào hệ thống."
    ),
    INTENT_IDENTITY_USER: (
        "Tôi không có thông tin định danh về bạn — phiên chat hiện không lưu "
        "thông tin cá nhân. Nếu muốn tôi xưng hô đúng, bạn có thể cho biết tên "
        "hoặc vai trò của mình nhé."
    ),
    INTENT_CAPABILITY: (
        "Tôi có thể giúp bạn:\n"
        "- Tra cứu tài liệu nội bộ TDI (quy trình, dự án, hợp đồng, BIM/MEP/CNTT…).\n"
        "- Trích xuất thông tin từ PDF/Word/video đã nạp vào hệ thống, kèm trích dẫn nguồn.\n"
        "- Lọc theo lĩnh vực chuyên môn: BIM, MEP, Marketing, Pháp lý, Sản xuất, CNTT, "
        "Nhân sự, Tài chính, Kinh doanh, Thiết kế.\n\n"
        "Bạn muốn tra cứu chủ đề nào?"
    ),
    INTENT_THANKS: (
        "Không có gì! Khi nào cần tra cứu thêm tài liệu, cứ hỏi tôi nhé."
    ),
    INTENT_GOODBYE: (
        "Tạm biệt bạn! Khi nào cần tra cứu tài liệu TDI, cứ quay lại nhé."
    ),
}


def canned_response(intent: str) -> str:
    """Lấy canned response cho intent. Trả "" nếu intent không có canned (RAG)."""
    return _CANNED.get(intent, "")


def is_meta_intent(intent: str) -> bool:
    """True nếu intent NON-RAG — cần short-circuit, KHÔNG gọi retrieval/Claude."""
    return intent != INTENT_RAG
