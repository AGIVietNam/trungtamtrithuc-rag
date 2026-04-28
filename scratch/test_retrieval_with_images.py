"""Test retrieval E2E — verify caption ảnh giúp ảnh match câu hỏi liên quan.

Pipeline:
    1. Ingest <pdf> qua ingest_document() — upsert chunks (có images metadata + caption gộp embed)
       vào Qdrant collection tdi_docs_{domain}.
    2. Với mỗi câu hỏi test: embed → retrieve top-K → in score, source, page, ảnh+caption.
    3. Đánh giá thủ công: ảnh đúng có xuất hiện trong top hits không, score có hợp lý không.

Usage:
    venv/bin/python scratch/test_retrieval_with_images.py <pdf_path> [domain]

    # Custom queries (sau dấu --):
    venv/bin/python scratch/test_retrieval_with_images.py docs/test.pdf mac_dinh \\
        -- "câu hỏi 1?" "câu hỏi 2?"

Default domain: mac_dinh. Default queries: lấy từ caption của ảnh đầu tiên (LLM
sinh paraphrase) — best-effort để có câu hỏi liên quan đến ảnh.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
# Giảm noise từ httpx/docling khi đã warm
for noisy in ("httpx", "huggingface_hub", "docling"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv at '--': trước là positional, sau là queries."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _default_queries() -> list[str]:
    """Default queries — mix giữa text-relevant + caption-specific để test cả
    text chunk recall và synthetic image-caption chunk precision."""
    return [
        # Text-relevant: text chunk gốc nên thắng, kéo theo cả lô images.
        "Tài liệu này nói về chủ đề gì?",
        # Caption-specific: synthetic chunk caption (sơ đồ kiến trúc) nên thắng,
        # chỉ trả về 1 ảnh đó. Score nên CAO (>0.5) so với cross-lingual text chunk.
        "Có sơ đồ kiến trúc mạng nơ-ron nào trong tài liệu không?",
        # Caption-specific khác: bounding box / nhận diện đối tượng.
        "Hình ảnh nhận diện đối tượng với bounding box ở đâu?",
    ]


def main() -> int:
    pos, custom_queries = _split_argv(sys.argv[1:])
    if len(pos) < 1:
        print("Usage: python scratch/test_retrieval_with_images.py <pdf_path> [domain] [-- 'q1' 'q2']")
        return 1

    pdf_path = Path(pos[0]).resolve()
    # Default 'bim' — chọn 1 slug hợp lệ. 'mac_dinh' là persona chat, KHÔNG phải
    # collection Qdrant. Pass slug khác để route đúng tdi_docs_{slug}.
    domain = pos[1] if len(pos) >= 2 else "bim"
    queries = custom_queries or _default_queries()

    valid_domains = {"bim", "mep", "marketing", "phap_ly", "san_xuat",
                     "cntt", "nhan_su", "tai_chinh", "kinh_doanh", "thiet_ke"}
    if domain not in valid_domains:
        print(f"❌ domain={domain!r} không hợp lệ. Chọn 1 trong: {sorted(valid_domains)}")
        return 1

    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        print(f"PDF không tồn tại hoặc không phải .pdf: {pdf_path}")
        return 1

    # Validate keys trước khi import nặng
    from app.core.config import (
        VOYAGE_API_KEY, QDRANT_URL, QDRANT_API_KEY,
        VOYAGE_MODEL, VOYAGE_DIM,
        QDRANT_VMEDIA_URL, QDRANT_VMEDIA_API_KEY, VMEDIA_COLLECTIONS,
    )
    for name, val in [("VOYAGE_API_KEY", VOYAGE_API_KEY),
                      ("QDRANT_URL", QDRANT_URL),
                      ("QDRANT_API_KEY", QDRANT_API_KEY)]:
        if not val or "placeholder" in val:
            print(f"❌ {name} chưa được set thật trong .env (đang là: {val[:30]}...)")
            return 1

    print(f"\n=== Test retrieval E2E ===")
    print(f"PDF:    {pdf_path.name}")
    print(f"Domain: {domain}")
    print(f"Queries: {len(queries)}\n")

    # --- Step 1: Ingest ---
    print("─── [1/2] INGEST ─────────────────────────────────────────")
    from app.ingestion.doc_pipeline import ingest_document
    result = ingest_document(
        str(pdf_path), pdf_path.name,
        metadata={"domain": domain, "title": pdf_path.stem},
    )
    print(f"\n→ doc_id={result.doc_id} pages={result.num_pages} chunks={result.num_chunks}\n")

    # --- Step 2: Build Retriever + Reranker ---
    print("─── [2/2] RETRIEVE + RERANK ──────────────────────────────")
    from app.core.voyage_embed import VoyageEmbedder
    from app.core.qdrant_store import QdrantRegistry, VMediaReadOnlyStore
    from app.rag.retriever import Retriever
    from app.rag.reranker import CrossEncoderReranker

    voyage = VoyageEmbedder(api_key=VOYAGE_API_KEY, model=VOYAGE_MODEL)
    registry = QdrantRegistry(
        url=QDRANT_URL, api_key=QDRANT_API_KEY,
        vector_size=VOYAGE_DIM,
    )
    vmedia = VMediaReadOnlyStore(
        url=QDRANT_VMEDIA_URL,
        vmedia_api_key=QDRANT_VMEDIA_API_KEY,
        collections=VMEDIA_COLLECTIONS,
    )
    retriever = Retriever(voyage=voyage, registry=registry, vmedia_store=vmedia)
    # min_score=0 để KHÔNG filter — script test, muốn xem full ranking trước/sau.
    reranker = CrossEncoderReranker(min_score=0.0)

    def _print_hit(h_idx: int, hit, retr_score: float | None = None) -> None:
        payload = hit.payload
        page = payload.get("page", "?")
        source = payload.get("source_name", "?")
        images = payload.get("images", [])
        chunk_type = payload.get("chunk_type", "text")
        doc_id_match = "✓" if payload.get("doc_id") == result.doc_id else " "
        text_snippet = (hit.text[:120] + "…") if len(hit.text) > 120 else hit.text
        text_snippet = text_snippet.replace("\n", " ")
        if retr_score is not None:
            score_part = f"score={hit.score:+.4f} (retr={retr_score:.4f})"
        else:
            score_part = f"score={hit.score:.4f}"
        print(f"│  [{h_idx}] {score_part} {doc_id_match} type={chunk_type:14s} "
              f"{source} (page {page})")
        print(f"│      text: {text_snippet}")
        if images:
            print(f"│      images ({len(images)}):")
            for img in images:
                cap = img.get("caption", "(no caption)")
                cap_short = (cap[:100] + "…") if len(cap) > 100 else cap
                print(f"│        - {img['image_id']} (page {img.get('page')}) {cap_short}")
        else:
            print(f"│      images: (none)")

    for q_idx, query in enumerate(queries, 1):
        print(f"\n┌── Q{q_idx}: {query}")
        hits = retriever.retrieve(
            query=query, top_k=5, domain=domain, sources=["documents"],
        )
        if not hits:
            print("│  (không có hit nào)")
            continue

        # Stage 1: retrieve scores (Voyage cosine similarity)
        print(f"│  --- STAGE 1: Retrieve (Voyage cosine, top {len(hits)}) ---")
        for h_idx, hit in enumerate(hits, 1):
            _print_hit(h_idx, hit)

        # Stage 2: rerank với BGE cross-encoder. rerank() mutate hit.score
        # in-place → save retrieve scores trước để in compare.
        retr_scores = {id(h): h.score for h in hits}
        reranked = reranker.rerank(query, hits, top_k=5)

        print(f"│  --- STAGE 2: Rerank (BGE cross-encoder) ---")
        for h_idx, hit in enumerate(reranked, 1):
            _print_hit(h_idx, hit, retr_score=retr_scores.get(id(hit), 0.0))
        print(f"└──")

    return 0


if __name__ == "__main__":
    sys.exit(main())
