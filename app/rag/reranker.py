from __future__ import annotations

import logging
import math
import os

from app.rag.retriever import Hit

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    """Convert BGE raw logit → [0,1] probability.

    BGE-reranker-v2-m3 mặc định trả raw cross-encoder logit (≈ -10..+10).
    Để dùng làm threshold absolute (vd "≥ 0.6 mới đáng tin"), sigmoid hoá:
      logit  0   → 0.5  (random)
      logit  2   → 0.88 (strong relevance)
      logit -2   → 0.12 (clearly irrelevant)
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def top_score_gap(hits: list[Hit]) -> float:
    """Khoảng cách giữa hit top-1 và top-2 (sigmoid).

    Gap nhỏ → reranker không phân biệt được top-1 với phần còn lại; gap lớn
    nhưng top-1 thấp → "least bad" not "actually relevant" (case BKVN).
    """
    if len(hits) < 2:
        return 1.0 if hits else 0.0
    return float(hits[0].score) - float(hits[1].score)

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

    Scores each (query, document) pair for semantic relevance, normalize qua
    sigmoid → [0,1]. Filter chunks dưới min_score (sigmoid).

    Phase 2 đổi từ raw logit sang sigmoid để:
      - Threshold absolute có ý nghĩa cố định (0.5 = coin-flip).
      - Có thể compare với threshold của conv_memory rerank (Phase 3.2).
      - Confidence label trong chain.py mapping ổn định.
    """

    def __init__(self, min_score: float = 0.05):
        # min_score giờ trên thang sigmoid. 0.05 ≈ raw logit -3 (rất irrelevant).
        # Cũ là 0.01 trên raw logit → khác hệ, KHÔNG dùng được nữa.
        self.min_score = min_score

    def rerank(self, query: str, hits: list[Hit], top_k: int = 3) -> list[Hit]:
        if not hits:
            return hits

        model = _get_cross_encoder()

        pairs = [[query, h.text] for h in hits]
        raw_scores = model.predict(pairs)

        for h, raw in zip(hits, raw_scores):
            # Lưu cả 2 để debug — score chính dùng cho threshold/sort là sigmoid.
            raw_f = float(raw)
            h.score = sigmoid(raw_f)
            # Stash raw vào payload để debug khi cần (không affect citation).
            try:
                h.payload["_rerank_raw"] = raw_f
                h.payload["_rerank_sigmoid"] = h.score
            except Exception:
                pass

        # Filter out irrelevant chunks (sigmoid scale)
        relevant = [h for h in hits if h.score >= self.min_score]

        if not relevant:
            # Fallback: keep best hit even if below threshold (caller có gate riêng)
            hits.sort(key=lambda h: h.score, reverse=True)
            logger.info(
                "rerank: no chunk passed min_score=%.3f → keep best (sigmoid=%.3f)",
                self.min_score, hits[0].score if hits else 0.0,
            )
            return hits[:1]

        relevant.sort(key=lambda h: h.score, reverse=True)
        top = relevant[:top_k]
        logger.info(
            "rerank: %d/%d passed min_score=%.3f, top_sigmoid=%.3f, gap=%.3f",
            len(relevant), len(hits), self.min_score,
            top[0].score, top_score_gap(top),
        )
        return top

    def rerank_memory(
        self,
        query: str,
        pairs: list[dict],
        min_sigmoid: float = 0.75,
        max_kept: int = 3,
    ) -> list[dict]:
        """Rerank conv_memory recall pairs với threshold cao hơn doc rerank.

        Phase 3.2 anti-contamination: pair có vector similarity với query đủ
        để được Qdrant return KHÔNG đủ tin để xem là context. Dùng cùng BGE
        cross-encoder + threshold 0.75 (sigmoid) — pair phải "thực sự liên
        quan ngữ nghĩa" mới được giữ. Threshold cao vì memory KHÔNG phải
        evidence — over-cautious tốt hơn over-trust.

        Mỗi pair được augment thêm field ``rerank_sigmoid``. Trả pair sorted
        theo score giảm dần, tối đa max_kept entries (vẫn dưới ngưỡng → []).
        """
        if not pairs:
            return []

        model = _get_cross_encoder()
        cross_pairs = [[query, p.get("text", "")] for p in pairs]
        raw_scores = model.predict(cross_pairs)

        scored: list[tuple[float, dict]] = []
        for p, raw in zip(pairs, raw_scores):
            sig = sigmoid(float(raw))
            p_aug = {**p, "rerank_sigmoid": sig, "rerank_raw": float(raw)}
            scored.append((sig, p_aug))

        kept = [p for sig, p in scored if sig >= min_sigmoid]
        kept.sort(key=lambda p: p["rerank_sigmoid"], reverse=True)

        dropped = len(pairs) - len(kept)
        logger.info(
            "rerank_memory: kept=%d dropped=%d (min_sigmoid=%.2f) — top_score=%.3f",
            len(kept), dropped, min_sigmoid,
            kept[0]["rerank_sigmoid"] if kept else 0.0,
        )
        return kept[:max_kept]
