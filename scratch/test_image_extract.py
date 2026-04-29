"""Spike test: trích ảnh + sinh caption + upload S3 từ 1 file PDF.

KHÔNG ingest Qdrant — chỉ test ingestion sub-pipeline (extract → caption → S3).

Cách dùng:
    source venv/bin/activate
    python scratch/test_image_extract.py path/to/file.pdf

Cần trong .env:
    ANTHROPIC_API_KEY (caption Haiku)
    S3_ENDPOINT, S3_BUCKET_NAME, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY (upload)

Mục tiêu test:
    1. PyMuPDF mở file OK, trích được số ảnh hợp lý.
    2. Filter dimension/bytes loại được logo/icon nhỏ.
    3. PNG được upload đúng lên S3 với key `images/{doc_id}/{image_id}.png`.
    4. Caption Haiku trả về tiếng Việt 1-2 câu, có ngữ cảnh page.
    5. URL public truy cập được (mở browser → thấy ảnh).
    6. parse_pdf() trả về schema mới: per-page với "images" có "url".
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.ingestion.doc_parser import (  # noqa: E402
    _extract_caption_upload_images,
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

    # Verify S3 configured trước khi tốn Haiku call.
    from app.core import s3_client
    if not s3_client.is_configured():
        print("❌ S3 chưa config (S3_ENDPOINT/S3_BUCKET_NAME rỗng) — kiểm tra .env")
        return 1

    doc_id = _sha256(pdf_path)
    print(f"\n=== File: {pdf_path.name} (doc_id={doc_id}) ===\n")

    # --- Phase 1: 1-pass extract + caption + upload ---
    print("[Phase 1] Extract → Caption → Upload S3 (1 pass)\n")
    # Empty page_texts → caption không có context, vẫn chạy để verify pipeline.
    # Pipeline thật ở parse_pdf sẽ pass page_texts từ Docling/Vision.
    images_by_page = _extract_caption_upload_images(pdf_path, doc_id, {})
    total = sum(len(imgs) for imgs in images_by_page.values())
    print(f"→ Total: {total} ảnh trên {len(images_by_page)} trang\n")
    for page_num in sorted(images_by_page):
        for img in images_by_page[page_num]:
            cap = img.get("caption", "") or "(rỗng)"
            print(f"  page={page_num} id={img['image_id']} size={img['width']}x{img['height']}")
            print(f"    url:     {img.get('url', '(skip — upload fail)')}")
            print(f"    caption: {cap}")
            print()

    if total == 0:
        print("Không có ảnh nào pass filter — dừng test.")
        return 0

    # --- Phase 2: Full parse_pdf — verify schema (text + images) ---
    print("[Phase 2] parse_pdf() full pipeline — verify schema có URL\n")
    pages = parse_pdf(pdf_path, doc_id=doc_id)
    summary = [
        {
            "page": p["page"],
            "text_chars": len(p.get("text", "")),
            "num_images": len(p.get("images", [])),
            "first_image_url": (
                p["images"][0].get("url", "") if p.get("images") else None
            ),
        }
        for p in pages
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n→ Mở URL trong browser để verify ảnh hiển thị đúng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
