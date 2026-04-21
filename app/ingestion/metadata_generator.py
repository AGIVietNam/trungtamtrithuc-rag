"""AI auto-generate document metadata (title, description, domain, tags).

Dùng Anthropic Claude Haiku 4.5 + tool use để ép LLM trả output đúng schema:
- Controlled vocabulary cho `domain` (7 label cố định) → không hallucinate.
- Pydantic validate sau khi nhận tool_use.input.
- Fallback an toàn: nếu LLM fail hoặc input thiếu ngữ cảnh → trả None.

Chi phí: ~$0.004/file (~3000 token input, ~200 token output).
Latency: ~1-1.5s/file.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import ANTHROPIC_API_KEY, CLAUDE_HAIKU_MODEL

logger = logging.getLogger(__name__)

DomainLiteral = Literal[
    "bim", "mep", "kết cấu", "marketing", "pháp lý", "sản xuất", "mặc định"
]

DOMAIN_VALUES: list[str] = [
    "bim", "mep", "kết cấu", "marketing", "pháp lý", "sản xuất", "mặc định"
]

MIN_TEXT_LEN = 200       # dưới ngưỡng này input không đủ ngữ cảnh, skip LLM
MAX_SAMPLE_CHARS = 6000  # ~3000 token tiếng Việt
MAX_TAGS = 8


class AIMetadata(BaseModel):
    """Schema metadata AI trả về. Tất cả field đều required."""

    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=500)
    domain: DomainLiteral
    tags: list[str] = Field(default_factory=list, max_length=MAX_TAGS + 2)


# Tool schema gửi cho Claude — viết tay thay vì dùng Pydantic JSON schema để
# tránh các trường `$defs`, `title` tự sinh dư thừa. Anthropic tools nhận JSON
# Schema chuẩn.
_TOOL_DEF: dict[str, Any] = {
    "name": "save_document_metadata",
    "description": (
        "Lưu metadata đã phân tích từ tài liệu vào hệ thống danh mục tri thức."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Tên tài liệu rõ ràng, 5-120 ký tự, tiếng Việt có dấu",
                "minLength": 3,
                "maxLength": 200,
            },
            "description": {
                "type": "string",
                "description": "2-3 câu tóm tắt nội dung, plain text tiếng Việt, tối đa 80 từ",
                "minLength": 10,
                "maxLength": 500,
            },
            "domain": {
                "type": "string",
                "enum": DOMAIN_VALUES,
                "description": "Đúng 1 trong 7 domain cố định",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 40},
                "minItems": 0,
                "maxItems": 10,
                "description": "3-8 từ khoá kỹ thuật ngắn, lowercase, tối đa 3 từ/tag",
            },
        },
        "required": ["title", "description", "domain", "tags"],
    },
}


_SYSTEM_PROMPT = """Bạn là trợ lý phân loại tài liệu kỹ thuật tiếng Việt.
Nhiệm vụ: Đọc trích đoạn tài liệu và GỌI tool save_document_metadata với 4 field.

<domain_guide>
Chọn ĐÚNG 1 domain từ danh sách:
- bim: BIM, IFC, Revit, LOD, clash detection, mô hình 3D công trình
- mep: điện, cơ, nước, HVAC, PCCC, sprinkler, ELV, BMS
- kết cấu: bê tông, thép, móng, tải trọng, FEM, kết cấu công trình
- marketing: truyền thông, thương hiệu, campaign, content, funnel, KPI
- pháp lý: hợp đồng, nghị định, thông tư, luật xây dựng/doanh nghiệp
- sản xuất: thi công, an toàn lao động, tiến độ, Lean, 5S, OEE
- mặc định: không khớp rõ với bất kỳ nhóm nào ở trên
</domain_guide>

<rules>
- title: Dùng gợi ý <heading> và <filename>, 5-120 ký tự, ngắn gọn rõ ràng.
  KHÔNG giữ noise như "Copy of", "_final_v2".
- description: 2-3 câu, tối đa 80 từ, plain text, tiếng Việt khách quan.
  KHÔNG bắt đầu bằng "Tài liệu này...", "Đây là...".
- tags: 3-8 từ khoá kỹ thuật ngắn, lowercase, tối đa 3 từ/tag.
  KHÔNG trùng domain, KHÔNG lặp lại nguyên văn các từ đã có trong title.
- Giữ chính xác tên riêng, số liệu, thuật ngữ EN.
- KHÔNG bịa thông tin không có trong trích đoạn.
</rules>"""


def _clean_filename_hint(filename: str) -> str:
    """Filename → hint text: strip ext, strip 'Copy of', 'v1/v2/_final', underscores."""
    stem = Path(filename).stem
    stem = re.sub(r"(?i)\b(copy of|bản sao của|bản sao)\b", "", stem)
    stem = re.sub(r"(?i)[_\- ]?v\d+(\.\d+)?$", "", stem)
    stem = re.sub(r"(?i)[_\- ]?(final|draft|last|new|edited|updated)\b", "", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def _extract_first_heading(text: str) -> str:
    """Trích heading Markdown đầu tiên (# Heading) từ text."""
    m = re.search(r"(?m)^#{1,6}\s+(.{3,120}?)\s*$", text)
    return m.group(1).strip() if m else ""


def _build_user_content(text_sample: str, filename_hint: str, heading_hint: str) -> str:
    parts = [f"<filename>{filename_hint}</filename>"]
    if heading_hint:
        parts.append(f"<heading>{heading_hint}</heading>")
    parts.append(f"<content>\n{text_sample}\n</content>")
    return "\n".join(parts)


def _normalize_tags(tags: list[str]) -> list[str]:
    """Lowercase + collapse whitespace + dedup + cap ở MAX_TAGS."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in tags or []:
        t = (raw or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        t = t.strip(".,;:!?\"'()[]{}")
        if 2 <= len(t) <= 40 and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= MAX_TAGS:
            break
    return out


def generate_document_metadata(
    text_sample: str,
    filename: str,
    heading_hint: str | None = None,
) -> AIMetadata | None:
    """Gọi Haiku với tool use, trả AIMetadata đã validate. None nếu fail.

    Caller luôn phải xử lý None → fallback title từ filename.
    """
    if not text_sample or len(text_sample.strip()) < MIN_TEXT_LEN:
        logger.info(
            "metadata_generator: text_sample quá ngắn (%d < %d), skip LLM",
            len(text_sample or ""), MIN_TEXT_LEN,
        )
        return None

    if not ANTHROPIC_API_KEY:
        logger.warning("metadata_generator: ANTHROPIC_API_KEY trống, skip")
        return None

    sample = text_sample[:MAX_SAMPLE_CHARS]
    name_hint = _clean_filename_hint(filename) or Path(filename).stem
    h_hint = (heading_hint or "").strip() or _extract_first_heading(sample)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        user_content = _build_user_content(sample, name_hint, h_hint)

        response = client.messages.create(
            model=CLAUDE_HAIKU_MODEL,
            max_tokens=600,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            tools=[_TOOL_DEF],
            # ép LLM phải gọi tool này — không trả text tự do
            tool_choice={"type": "tool", "name": "save_document_metadata"},
            temperature=0.2,
        )

        # Lấy tool_use block đầu tiên
        tool_input: dict[str, Any] | None = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_input = getattr(block, "input", None)
                break

        if not tool_input:
            logger.warning("metadata_generator: response không có tool_use block")
            return None

        try:
            meta = AIMetadata.model_validate(tool_input)
        except ValidationError as ve:
            logger.warning("metadata_generator: Pydantic validation fail: %s", ve)
            return None

        # Normalize tags post-validation (Pydantic không strip/lowercase)
        normalized = meta.model_copy(update={"tags": _normalize_tags(meta.tags)})
        logger.info(
            "metadata_generator OK: domain=%s title=%r tags=%d",
            normalized.domain, normalized.title[:60], len(normalized.tags),
        )
        return normalized

    except Exception:
        logger.exception("metadata_generator failed")
        return None
