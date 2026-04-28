"""Spike test: trích ảnh + sinh caption từ 1 file PDF, KHÔNG ingest Qdrant.

Cách dùng:
    source venv/bin/activate
    python scratch/test_image_extract.py path/to/file.pdf

Mục tiêu test (Tầng 1 — local, độc lập server):
    1. PyMuPDF mở file OK, trích được số ảnh hợp lý.
    2. Filter dimension/bytes loại được logo/icon nhỏ.
    3. File PNG được lưu đúng vào data/images/{doc_id}/.
    4. Caption Haiku trả về tiếng Việt 1-2 câu, không Markdown.
    5. parse_pdf() trả về schema mới: per-page với "images" list.

KHÔNG cần Qdrant/Voyage cho test này — chỉ cần ANTHROPIC_API_KEY trong .env.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Logging verbose để xem từng page extract bao nhiêu ảnh
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.ingestion.doc_parser import (  # noqa: E402
    _extract_images_from_pdf,
    _caption_images,
    _sha256,
    parse_pdf,
)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scratch/test_image_extract.py <pdf_path>")
        return 1

    pdf_path = Path(sys.argv[1]).resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        print(f"PDF không tồn tại hoặc không phải .pdf: {pdf_path}")
        return 1

    doc_id = _sha256(pdf_path)
    print(f"\n=== File: {pdf_path.name} (doc_id={doc_id}) ===\n")

    # --- Phase 1: Extract only (không gọi Haiku) ---
    print("[Phase 1] Extract images only (không caption)\n")
    images_by_page = _extract_images_from_pdf(pdf_path, doc_id)
    total = sum(len(imgs) for imgs in images_by_page.values())
    print(f"→ Total: {total} ảnh trên {len(images_by_page)} trang có ảnh\n")
    for page_num in sorted(images_by_page):
        for img in images_by_page[page_num]:
            print(f"  page={page_num} ord={img['ord']} "
                  f"id={img['image_id']} "
                  f"size={img['width']}x{img['height']} "
                  f"file={img['filename']}")

    if total == 0:
        print("\nKhông có ảnh nào pass filter — dừng test (không gọi Haiku).")
        return 0

    # --- Phase 2: Caption (chỉ chạy nếu có ảnh) ---
    print("\n[Phase 2] Sinh caption Haiku (cần ANTHROPIC_API_KEY)\n")
    page_texts: dict[int, str] = {}
    _caption_images(images_by_page, page_texts, doc_id)
    for page_num in sorted(images_by_page):
        for img in images_by_page[page_num]:
            cap = img.get("caption", "") or "(rỗng)"
            print(f"  [{img['image_id']}] {cap}")

    # --- Phase 3: Full parse_pdf — verify schema mới ---
    print("\n[Phase 3] parse_pdf() full pipeline — verify schema\n")
    pages = parse_pdf(pdf_path, doc_id)
    summary = [
        {
            "page": p["page"],
            "text_chars": len(p.get("text", "")),
            "num_images": len(p.get("images", [])),
        }
        for p in pages
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # In path tuyệt đối của 1 ảnh để user mở xem trực tiếp
    first_page_imgs = next(
        (p["images"] for p in pages if p.get("images")), []
    )
    if first_page_imgs:
        from app.core.config import DATA_DIR
        first_path = DATA_DIR / "images" / doc_id / first_page_imgs[0]["filename"]
        print(f"\nẢnh mẫu để xem: {first_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
