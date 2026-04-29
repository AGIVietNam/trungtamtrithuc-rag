"""BM25 sparse encoder cho hybrid retrieval (Anthropic Contextual Retrieval recipe).

Pipeline:
  text → underthesea word_tokenize → normalize → hash → sparse vector {indices, values}

Sparse vector emit chỉ TF (term frequency); IDF do Qdrant tự nhân lúc query
qua collection config ``sparse_vectors.sparse.modifier = "idf"``. Như vậy IDF
luôn cập nhật khi corpus đổi (không phải re-encode lại documents).

Vocab id: stable hash 64-bit FNV-1a → modulo ``BM25_HASH_BUCKETS`` (default
2^24 = 16M buckets). Collisions rare, cho phép ingest và query trong process
khác nhau vẫn ra cùng id mà không cần persist vocabulary.

Tokenizer: underthesea — accuracy 80% trên benchmark VN, gộp đúng cụm tiếng
Việt như "máng cáp" → "máng_cáp" (một token), "kinh doanh" → "kinh_doanh".
Chọn underthesea thay pyvi vì accuracy cao hơn nhiều, vẫn pure Python.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import TypedDict

from app import config

logger = logging.getLogger(__name__)


class SparseVector(TypedDict):
    """Sparse vector format Qdrant chấp nhận trong Query API."""

    indices: list[int]
    values: list[float]


# FNV-1a 64-bit constants — stable hash, không phụ thuộc Python's randomized hash.
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_FNV_MASK = 0xFFFFFFFFFFFFFFFF


def _fnv1a_64(text: str) -> int:
    """Stable 64-bit hash. Cùng text → cùng hash bất kỳ process nào."""
    h = _FNV_OFFSET
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * _FNV_PRIME) & _FNV_MASK
    return h


# Tokens dưới ngưỡng độ dài bị bỏ — vd "a", "ở", "1" thường noise cho BM25.
# 2 ký tự minimum bắt được "5G", "AI", "ML"... vẫn lọc ký tự đơn.
_MIN_TOKEN_LEN = 2

# Stopwords tiếng Việt phổ biến — drop để giảm noise + giữ IDF có ý nghĩa cho
# từ thực sự content. Danh sách rút gọn (không quá tham); đủ chặn các filler
# kéo embedding lệch như "hướng dẫn cho nhân viên".
_STOPWORDS_VI: frozenset[str] = frozenset({
    "và", "của", "là", "có", "được", "cho", "với", "từ", "đến", "trong",
    "ngoài", "trên", "dưới", "tại", "về", "này", "đó", "kia", "ấy", "vậy",
    "thì", "mà", "nhưng", "nên", "thế", "rằng", "khi", "lúc", "nếu",
    "hay", "hoặc", "cả", "đều", "rất", "quá", "lắm", "thôi", "đã",
    "đang", "sẽ", "vẫn", "còn", "chưa", "không", "chẳng", "đừng",
    "tôi", "bạn", "mình", "em", "anh", "chị", "ông", "bà", "ai",
    "gì", "nào", "sao", "đâu",
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
    "and", "or", "but", "in", "on", "at", "for", "with", "by",
})


def _normalize_token(tok: str) -> str:
    """Lowercase + strip diacritics-aware, giữ tiếng Việt có dấu nguyên vẹn.

    Không bỏ dấu — diacritic mang nghĩa ("má", "mã", "mà" khác nhau).
    Chỉ lowercase + NFC normalize để "máng" và "máng" cùng dạng.
    """
    tok = unicodedata.normalize("NFC", tok).strip()
    return tok.lower()


def _tokenize(text: str) -> list[str]:
    """Word-segment + normalize. Trả unigram tokens đã filter stopwords + min length.

    Quan trọng: underthesea ghép compound theo context (vd "vật tư máng cáp" →
    "vật_tư_máng_cáp" trong 1 câu nhưng "vật_tư_máng cáp" trong câu khác). Nếu
    dùng compound làm token BM25, hai query gần giống nhau ra index khác nhau
    → mất overlap. Để tránh: tách compound `_` thành unigrams sau khi segment.
    Underthesea vẫn có ích vì dấu `_` đánh dấu "đây là cụm có nghĩa" — nhưng
    với BM25 ta chỉ cần unigram-level match (đã đủ với IDF cho từ hiếm).

    Lazy-import underthesea để khỏi load model lúc app startup nếu sparse
    chưa được kích hoạt.
    """
    if not text or not text.strip():
        return []
    try:
        from underthesea import word_tokenize  # type: ignore
        segmented = word_tokenize(text, format="text")
    except Exception:
        # Fallback whitespace nếu underthesea không cài / lỗi runtime — không fatal.
        logger.warning("underthesea unavailable, fallback whitespace tokenize", exc_info=True)
        segmented = text

    # Split compounds (vật_tư_máng_cáp → vật, tư, máng, cáp) AND extract individual
    # words. ``[^\W_]`` matches Unicode word char nhưng KHÔNG match underscore →
    # tách "_" tự nhiên. Hyphen / dấu nối khác cũng được tách (mã SKU
    # "C04XXX-PKMC..." sẽ split thành 2 token để khớp khi user hỏi từng nửa).
    raw_tokens = re.findall(r"[^\W_]+", segmented, flags=re.UNICODE)
    out: list[str] = []
    for t in raw_tokens:
        n = _normalize_token(t)
        if len(n) < _MIN_TOKEN_LEN:
            continue
        if n in _STOPWORDS_VI:
            continue
        out.append(n)
    return out


def encode(text: str) -> SparseVector:
    """Encode 1 text → sparse vector {indices, values} với TF raw.

    IDF do Qdrant tự nhân lúc query. Nếu text rỗng / toàn stopword → trả
    sparse rỗng (Qdrant chấp nhận).
    """
    tokens = _tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}
    counts = Counter(tokens)
    buckets = config.BM25_HASH_BUCKETS
    bucket_counts: dict[int, int] = {}
    for tok, tf in counts.items():
        idx = _fnv1a_64(tok) % buckets
        bucket_counts[idx] = bucket_counts.get(idx, 0) + tf
    indices = sorted(bucket_counts.keys())
    values = [float(bucket_counts[i]) for i in indices]
    return {"indices": indices, "values": values}


def encode_batch(texts: list[str]) -> list[SparseVector]:
    """Encode nhiều text — sequential vì underthesea không thread-safe.

    Caller có thể wrap trong ThreadPoolExecutor nếu muốn parallel ingest,
    nhưng underthesea load model 1 lần process nên sequential thường đủ nhanh.
    """
    return [encode(t) for t in texts]


def is_empty(sv: SparseVector) -> bool:
    return not sv.get("indices")
