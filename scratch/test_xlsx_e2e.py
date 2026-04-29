"""E2E test cho XLSX: ingest → upload S3 ảnh → query → verify URL.

Giống test_docx_e2e.py — khác ở extension check + default queries (hỏi
sheet/dữ liệu thay vì paragraph).

Usage:
    venv/bin/python scratch/test_xlsx_e2e.py docs/test.xlsx [domain]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
for noisy in ("httpx", "huggingface_hub", "docling", "openpyxl"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scratch/test_xlsx_e2e.py <xlsx_path> [domain]")
        return 1

    xlsx_path = Path(sys.argv[1]).resolve()
    domain = sys.argv[2] if len(sys.argv) >= 3 else "bim"

    if not xlsx_path.exists() or xlsx_path.suffix.lower() != ".xlsx":
        print(f"File không tồn tại / không phải .xlsx: {xlsx_path}")
        return 1

    valid_domains = {"bim", "mep", "marketing", "phap_ly", "san_xuat",
                     "cntt", "nhan_su", "tai_chinh", "kinh_doanh", "thiet_ke"}
    if domain not in valid_domains:
        print(f"❌ domain={domain!r} không hợp lệ.")
        return 1

    from app.core.config import (
        VOYAGE_API_KEY, QDRANT_URL, QDRANT_API_KEY,
        VOYAGE_MODEL, VOYAGE_DIM, QDRANT_VECTOR_NAME,
        QDRANT_VMEDIA_URL, QDRANT_VMEDIA_API_KEY, VMEDIA_COLLECTIONS,
    )
    for name, val in [("VOYAGE_API_KEY", VOYAGE_API_KEY),
                      ("QDRANT_URL", QDRANT_URL),
                      ("QDRANT_API_KEY", QDRANT_API_KEY)]:
        if not val or "placeholder" in val:
            print(f"❌ {name} chưa set thật.")
            return 1

    print(f"\n=== XLSX E2E test ===")
    print(f"File:   {xlsx_path.name}")
    print(f"Domain: {domain}\n")

    # --- INGEST ---
    print("─── INGEST ────────────────────────────────────────")
    from app.ingestion.doc_pipeline import ingest_document
    result = ingest_document(
        str(xlsx_path), xlsx_path.name,
        metadata={"domain": domain, "title": xlsx_path.stem},
    )
    print(f"\n→ doc_id={result.doc_id} pages(=sheets)={result.num_pages} chunks={result.num_chunks}\n")

    # --- RETRIEVE + RERANK ---
    print("─── RETRIEVE + RERANK ────────────────────────────")
    from app.core.voyage_embed import VoyageEmbedder
    from app.core.qdrant_store import QdrantRegistry, VMediaReadOnlyStore
    from app.rag.retriever import Retriever
    from app.rag.reranker import CrossEncoderReranker

    voyage = VoyageEmbedder(api_key=VOYAGE_API_KEY, model=VOYAGE_MODEL)
    registry = QdrantRegistry(
        url=QDRANT_URL, api_key=QDRANT_API_KEY,
        vector_size=VOYAGE_DIM, vector_name=QDRANT_VECTOR_NAME,
    )
    vmedia = VMediaReadOnlyStore(
        url=QDRANT_VMEDIA_URL,
        vmedia_api_key=QDRANT_VMEDIA_API_KEY,
        collections=VMEDIA_COLLECTIONS,
    )
    retriever = Retriever(voyage=voyage, registry=registry, vmedia_store=vmedia)
    reranker = CrossEncoderReranker(min_score=0.0)

    queries = [
        "Tài liệu này nói về chủ đề gì?",
        "Có hình ảnh, biểu đồ hoặc sơ đồ nào trong file Excel không?",
    ]

    for q_idx, query in enumerate(queries, 1):
        print(f"\n┌── Q{q_idx}: {query}")
        hits = retriever.retrieve(
            query=query, top_k=10, domain=domain, sources=["documents"],
            doc_id=result.doc_id,  # Filter chỉ hits của file vừa ingest
        )
        if not hits:
            print("│  (no hits)")
            continue
        hits = reranker.rerank(query, hits, top_k=5)

        for h_idx, hit in enumerate(hits, 1):
            payload = hit.payload
            chunk_type = payload.get("chunk_type", "text")
            page = payload.get("page", "?")
            sheet = payload.get("sheet_name", "")
            sheet_disp = f" sheet={sheet!r}" if sheet else ""
            doc_match = "✓" if payload.get("doc_id") == result.doc_id else " "
            text_snippet = (hit.text[:100] + "…") if len(hit.text) > 100 else hit.text
            text_snippet = text_snippet.replace("\n", " ")
            print(f"│  [{h_idx}] score={hit.score:.4f} {doc_match} type={chunk_type:14s} page={page}{sheet_disp}")
            print(f"│      text: {text_snippet}")
            for img in (payload.get("images") or []):
                cap = (img.get("caption", "") or "(no cap)")[:80]
                print(f"│        img {img['image_id']}: {cap}")
                print(f"│          url: {img.get('url', '(no url)')}")
        print(f"└──")

    print(f"\n=== Done. Mở 1 URL trong browser để verify ảnh XLSX upload S3 OK. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
