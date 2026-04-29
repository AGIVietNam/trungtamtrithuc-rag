"""Inline image markers — Strategy B (PM yêu cầu).

Flow:
    1. Pipeline retrieve → có hits với payload.images[] (image_id, url, caption).
    2. Trước khi gọi Claude, ghép `build_available_images_section(hits)` vào
       user_turn → Claude biết image_id và caption nào dùng được.
    3. Claude generate answer có chèn marker `[IMG:image_id]` đúng vị trí câu
       liên quan ảnh.
    4. Post-process bằng `resolve_image_markers(answer, hits)` → replace marker
       thành markdown `![caption](url)` (FE render inline).

Khác với Strategy A (sources-attached, ảnh ở card cuối): ảnh xuất hiện ngay
trong dòng văn của câu trả lời tương ứng.
"""
from __future__ import annotations

import re
from typing import Iterable

# Marker pattern: [IMG:abc123def456...] — image_id luôn là 16 hex chars (sha256[:16]).
# Strict pattern để Claude không thể "bịa" image_id ngắn/dài khác.
_MARKER_PATTERN = re.compile(r"\[IMG:([a-f0-9]{16})\]")


_INSTRUCTIONS = (
    "<available_images>\n"
    "Đây là danh sách ảnh đã được trích từ <retrieved_documents> ở trên, "
    "cùng image_id (16 ký tự hex) và caption.\n\n"
    "DANH SÁCH:\n"
    "{image_list}\n\n"
    "CÁCH CHÈN ẢNH VÀO CÂU TRẢ LỜI:\n"
    "Khi câu trả lời mô tả nội dung khớp với caption của 1 ảnh, chèn marker "
    "`[IMG:image_id]` NGAY SAU câu đó để hệ thống hiển thị ảnh inline ở FE.\n\n"
    "QUY TẮC TUYỆT ĐỐI:\n"
    "- Chỉ chèn marker cho image_id có trong DANH SÁCH trên.\n"
    "- Mỗi câu tối đa 1 marker.\n"
    "- Mỗi image_id chỉ chèn 1 lần trong toàn bộ câu trả lời.\n"
    "- Nếu không có ảnh nào thực sự minh họa câu, KHÔNG chèn marker.\n"
    "- KHÔNG bịa image_id mới; KHÔNG dùng định dạng khác (vd <img>, ![x], (image)).\n"
    "- Marker đặt SAU câu kết thúc bằng dấu chấm/?/! (không xen giữa câu).\n\n"
    "VÍ DỤ ĐÚNG:\n"
    "\"YOLOv8 có 3 thành phần: Backbone, Neck, Head. [IMG:73424ff78759fd51] "
    "Mô hình phát hiện đối tượng với bounding box trong thời gian thực. "
    "[IMG:f0775bf0a2aa02bf]\"\n"
    "</available_images>"
)


def _collect_unique_images(hits: Iterable) -> list[dict]:
    """Gom unique images từ hits, dedupe theo image_id, giữ score cao nhất."""
    seen: dict[str, dict] = {}
    for hit in hits:
        for img in (hit.payload.get("images") or []):
            iid = img.get("image_id")
            url = img.get("url")
            cap = (img.get("caption") or "").strip()
            if not iid or not url or not cap:
                continue
            existing = seen.get(iid)
            if existing is None or hit.score > existing["score"]:
                seen[iid] = {
                    "image_id": iid,
                    "url": url,
                    "caption": cap,
                    "score": hit.score,
                }
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)


def build_available_images_section(hits, max_images: int = 8) -> str:
    """Trả về XML block `<available_images>` để chèn vào user_turn cho Claude.

    Empty string nếu không có ảnh nào — tiết kiệm token, tránh prompt rỗng.

    Cap `max_images` để tránh prompt phình to khi doc có hàng chục ảnh —
    Claude khó dùng marker chính xác nếu list quá dài; lấy top theo score.
    """
    images = _collect_unique_images(hits)[:max_images]
    if not images:
        return ""
    # Format mỗi dòng: "- {image_id}: {caption}"
    lines = [f"- {img['image_id']}: {img['caption']}" for img in images]
    return _INSTRUCTIONS.format(image_list="\n".join(lines))


def resolve_image_markers(answer: str, hits) -> str:
    """Post-process: thay `[IMG:image_id]` bằng markdown `![caption](url)`.

    Lookup image_id → (url, caption) từ payload.images[] của hits.
    Marker không match (Claude bịa hoặc image_id sai) → bị xóa khỏi answer.

    Markdown image syntax: `![alt](url)` → FE markdown renderer sẽ render
    thành `<img src="url" alt="alt">`. Đặt giữa cặp `\n\n` để render thành
    block riêng (tránh dính text trên/dưới khi FE render markdown).
    """
    lookup: dict[str, dict] = {}
    for hit in hits:
        for img in (hit.payload.get("images") or []):
            iid = img.get("image_id")
            if not iid or iid in lookup:
                continue
            lookup[iid] = {
                "url": img.get("url", ""),
                "caption": (img.get("caption") or "").strip(),
            }

    def _replace(match: re.Match) -> str:
        iid = match.group(1)
        meta = lookup.get(iid)
        if not meta or not meta["url"]:
            # Marker không match → silently drop (Claude đôi khi bịa).
            return ""
        # Escape ký tự ngoặc trong caption để không vỡ markdown syntax.
        cap = meta["caption"].replace("(", "（").replace(")", "）") or "Hình ảnh"
        return f"\n\n![{cap}]({meta['url']})\n\n"

    resolved = _MARKER_PATTERN.sub(_replace, answer)
    # Cleanup nhiều \n liên tiếp do replace tạo ra.
    resolved = re.sub(r"\n{3,}", "\n\n", resolved)
    return resolved.strip()
