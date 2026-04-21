from __future__ import annotations

import logging
from typing import Any

from app.rag.retriever import Hit

logger = logging.getLogger(__name__)

_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load cross-encoder model (cached after first call)."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    from sentence_transformers import CrossEncoder
    model_name = "BAAI/bge-reranker-v2-m3"
    logger.info("Loading cross-encoder: %s", model_name)
    _cross_encoder = CrossEncoder(model_name)
    logger.info("Cross-encoder loaded")
    return _cross_encoder


class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Scores each (query, document) pair for semantic relevance.
    Filters out chunks below the relevance threshold.
    """

    def __init__(self, min_score: float = 0.1):
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
