from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Literal

import requests

logger = logging.getLogger(__name__)

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
BATCH_SIZE = 32
MAX_RETRIES = 5

# Backoff cho 429 — trước đây hardcoded 25s (over-defensive cho free tier 3 RPM)
# khiến mỗi lần rate-limit → cộng dồn 75s+ trên 3 retry. Ưu tiên `Retry-After`
# header thật từ server, fallback ngắn để không idle vô lý khi cache + dedup
# đã giảm hẳn số lần gọi.
_DEFAULT_429_BACKOFF_SEC = 5.0
_MAX_429_BACKOFF_SEC = 15.0

# LRU cache cho embed_query — mỗi turn chat hiện gọi cùng query nhiều chỗ
# (rewriter check + retriever + conv_memory.retrieve). Hash exact đủ chặn
# hầu hết lặp; chat thực tế cũng có 15-25% turn user gõ lại query gần giống
# (typo, follow-up). Cache theo (model, text_normalized) — không leak giữa
# Voyage models nếu có ngày dùng song song.
_QUERY_CACHE_SIZE = 256
_query_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()


def _query_cache_key(model: str, text: str) -> tuple[str, str]:
    return (model, " ".join(text.lower().split()))


class VoyageEmbedder:
    def __init__(self, api_key: str, model: str = "voyage-3"):
        self.api_key = api_key
        self.model = model

    def _embed_batch(
        self,
        texts: list[str],
        input_type: Literal["document", "query"],
    ) -> list[list[float]]:
        for attempt in range(MAX_RETRIES):
            resp = None
            try:
                resp = requests.post(
                    VOYAGE_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"input": texts, "model": self.model, "input_type": input_type},
                    timeout=60,
                )
                if resp.status_code == 429:
                    if attempt == MAX_RETRIES - 1:
                        resp.raise_for_status()
                    wait = _retry_after_seconds(resp.headers.get("Retry-After"))
                    logger.warning(
                        "voyage 429 retry %d/%d after %.1fs (Retry-After=%r)",
                        attempt + 1, MAX_RETRIES, wait,
                        resp.headers.get("Retry-After"),
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                logger.info(
                    "voyage embed model=%s input_type=%s texts=%d tokens=%s",
                    self.model,
                    input_type,
                    len(texts),
                    usage.get("total_tokens"),
                )
                return [item["embedding"] for item in data["data"]]
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = float(2 ** attempt)
                logger.warning(
                    "voyage retry %d/%d after %.1fs: %s",
                    attempt + 1, MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
        raise RuntimeError("voyage embed failed after retries")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            results.extend(self._embed_batch(batch, "document"))
        return results

    def embed_query(self, text: str) -> list[float]:
        key = _query_cache_key(self.model, text)
        cached = _query_cache.get(key)
        if cached is not None:
            _query_cache.move_to_end(key)
            return cached
        vec = self._embed_batch([text], "query")[0]
        _query_cache[key] = vec
        while len(_query_cache) > _QUERY_CACHE_SIZE:
            _query_cache.popitem(last=False)
        return vec


def _retry_after_seconds(header_val: str | None) -> float:
    """Parse Retry-After header → seconds, cap để khỏi idle vô lý.

    Voyage trả số giây dạng string. Một số server trả HTTP-date (RFC 7231),
    không xử lý ở đây — dùng fallback nếu parse fail.
    """
    if not header_val:
        return _DEFAULT_429_BACKOFF_SEC
    try:
        secs = float(header_val)
    except ValueError:
        return _DEFAULT_429_BACKOFF_SEC
    return min(max(secs, 1.0), _MAX_429_BACKOFF_SEC)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    embedder = VoyageEmbedder(api_key=os.getenv("VOYAGE_API_KEY", ""))
    vec = embedder.embed_query("xin chào")
    print(f"dim={len(vec)} first5={vec[:5]}")
