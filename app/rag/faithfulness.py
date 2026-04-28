"""Faithfulness gate (Phase 3.3) — kiểm tra answer có grounded vào context không.

Dùng Claude Haiku 4.5 làm cheap judge thay vì HHEM-2.1 model:
  - HHEM-2.1 (Vectara, T5-base, 600MB) train chủ yếu tiếng Anh — performance
    trên tiếng Việt chưa benchmark, có thể false-positive/negative.
  - Haiku 4.5 native tiếng Việt, ~500-800ms/call, $0.0002 — rẻ hơn full
    Opus call mà reasoning đủ để kiểm tra entailment đơn giản.

Gate chỉ trip khi: answer dài + có citations + KHÔNG phải refusal.
Phase 1 Citations gate đã catch case 0-citations; gate này catch case
"có citations nhưng paraphrase sai logic" hoặc "cherry-pick 1 word từ doc
rồi bịa rest".

Để skip gate (vd test latency-sensitive): set env FAITHFULNESS_GATE=0.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


_JUDGE_PROMPT = """Bạn là kiểm soát viên tính trung thực cho hệ thống RAG.

NHIỆM VỤ: chấm xem câu trả lời (HYPOTHESIS) có được hỗ trợ bởi tài liệu (PREMISE) không.

PREMISE (các đoạn tài liệu được trích):
---
{premise}
---

HYPOTHESIS (câu trả lời cần kiểm tra):
---
{hypothesis}
---

Tiêu chí GROUNDED=yes:
- Mọi fact định lượng (số liệu, tên hệ thống, URL, mốc thời gian) trong HYPOTHESIS đều có trong PREMISE hoặc suy luận trực tiếp từ PREMISE.
- Cho phép paraphrase, tóm tắt, sắp xếp lại — miễn nội dung gốc còn nguyên.

Tiêu chí GROUNDED=no:
- HYPOTHESIS chứa URL, tên sản phẩm, số liệu KHÔNG xuất hiện trong PREMISE.
- HYPOTHESIS đảo ngược ý hoặc thêm chi tiết suy đoán.
- HYPOTHESIS dùng kiến thức ngoài PREMISE để mở rộng.

CHÚ Ý: Câu chuyển tiếp ("Dưới đây là...", "Để trả lời câu hỏi...") KHÔNG cần grounded.
Câu refusal ("Tài liệu chưa có thông tin...") luôn grounded=yes.

Trả lời CHÍNH XÁC theo format (không thêm gì khác):
GROUNDED: <yes|no>
REASON: <1 câu ngắn giải thích, < 30 từ>"""


def _is_enabled() -> bool:
    """Faithfulness gate có bật không (default ON)."""
    val = (os.getenv("FAITHFULNESS_GATE", "1") or "1").strip().lower()
    return val not in ("0", "false", "no", "off", "")


def _parse_judge_output(text: str) -> tuple[bool, str]:
    """Parse 'GROUNDED: yes/no\\nREASON: ...' từ Haiku output.

    Lenient: nếu không match exact format, fallback: lookup 'yes' trong dòng GROUNDED.
    """
    if not text:
        return True, "empty judge output → fallback grounded=true"

    grounded = True
    reason = ""

    m = re.search(r"GROUNDED\s*:\s*(yes|no)", text, re.IGNORECASE)
    if m:
        grounded = m.group(1).lower() == "yes"
    else:
        # Fallback heuristic: tìm yes/no đứng đầu
        first_line = text.strip().splitlines()[0].lower() if text.strip() else ""
        if "no" in first_line and "yes" not in first_line:
            grounded = False

    m = re.search(r"REASON\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()[:300]

    return grounded, reason


def check_faithfulness(
    claude_client,
    premise: str,
    hypothesis: str,
    judge_model: str = "claude-haiku-4-5",
) -> tuple[bool, str]:
    """Score xem hypothesis có grounded vào premise không.

    Returns (grounded, reason). Mọi lỗi → grounded=True (fail-open) để
    không chặn user khi judge model lỗi tạm thời.
    """
    if not _is_enabled():
        return True, "gate disabled via FAITHFULNESS_GATE=0"

    if not hypothesis or not hypothesis.strip():
        return True, "empty hypothesis"
    if not premise or not premise.strip():
        # Không có context → Phase 1 gate đã refuse; nếu lọt qua coi như grounded.
        return True, "empty premise (skip gate)"

    prompt = _JUDGE_PROMPT.format(
        premise=premise[:6000],  # cap to avoid blowing Haiku context
        hypothesis=hypothesis[:3000],
    )
    try:
        out = claude_client.quick_text(
            system_prompt="",
            user_content=prompt,
            max_tokens=120,
            model=judge_model,
        )
    except Exception as e:
        logger.warning("faithfulness judge failed (fail-open): %s", e)
        return True, f"judge_error: {e}"

    grounded, reason = _parse_judge_output(out)
    logger.info(
        "faithfulness: grounded=%s | reason=%s | hyp_len=%d",
        grounded, reason[:120], len(hypothesis),
    )
    return grounded, reason
