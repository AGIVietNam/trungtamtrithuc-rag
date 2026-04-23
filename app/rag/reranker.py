from __future__ import annotations

import logging
import os

from app.rag.retriever import Hit

logger = logging.getLogger(__name__)

_cross_encoder = None


def _detect_device() -> str:
    """Auto-detect best device: MPS (Mac) > CUDA (NVIDIA) > CPU.

    Override bằng env RERANKER_DEVICE (cpu/cuda/mps).
    """
    forced = (os.getenv("RERANKER_DEVICE") or "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    from app import config
    if config.REQUIRE_GPU:
        raise RuntimeError(
            "REQUIRE_GPU=1 but no CUDA/MPS device found! "
            "Please check your server GPU drivers or set REQUIRE_GPU=0."
        )
    return "cpu"


def _get_cross_encoder():
    """Lazy-load cross-encoder model (cached after first call)."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    from sentence_transformers import CrossEncoder
    model_name = "BAAI/bge-reranker-v2-m3"
    device = _detect_device()
    logger.info("Loading cross-encoder %s on device=%s", model_name, device)
    _cross_encoder = CrossEncoder(model_name, device=device)
    logger.info("Cross-encoder loaded (device=%s)", device)
    return _cross_encoder


def warmup() -> None:
    """Force-load model + chạy 1 inference giả để cache kernels GPU.

    Gọi lúc FastAPI startup → request đầu tiên không gánh load time.
    """
    model = _get_cross_encoder()
    try:
        model.predict([["warmup query", "warmup doc"]])
        logger.info("Cross-encoder warmup OK")
    except Exception:
        logger.warning("Cross-encoder warmup failed (will retry on first real call)", exc_info=True)


class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Scores each (query, document) pair for semantic relevance.
    Filters out chunks below the relevance threshold.
    """

    def __init__(self, min_score: float = 0.01):
        self.min_score = min_score

    def rerank(self, query: str, hits: list[Hit], top_k: int = 3) -> list[Hit]:
        if not hits:
            return hits

        model = _get_cross_encoder()

        pairs = [[query, h.text] for h in hits]
        scores = model.predict(pairs)

        for h, score in zip(hits, scores):
            h.score = float(score)

        # Filter out irrelevant chunks
        relevant = [h for h in hits if h.score >= self.min_score]

        if not relevant:
            # Fallback: keep best hit even if below threshold
            hits.sort(key=lambda h: h.score, reverse=True)
            return hits[:1]

        relevant.sort(key=lambda h: h.score, reverse=True)
        return relevant[:top_k]
