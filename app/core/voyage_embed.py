from __future__ import annotations

import logging
import time
from typing import Literal

import requests

logger = logging.getLogger(__name__)

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
BATCH_SIZE = 32
MAX_RETRIES = 5


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
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                # 429 rate limit needs longer backoff (free tier = 3 RPM)
                is_rate_limit = "429" in str(exc) or "Too Many" in str(exc)
                wait = 25 if is_rate_limit else 2 ** attempt
                logger.warning("voyage retry %d/%d after %ds: %s", attempt + 1, MAX_RETRIES, wait, exc)
                time.sleep(wait)
        raise RuntimeError("voyage embed failed after retries")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            results.extend(self._embed_batch(batch, "document"))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], "query")[0]


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    embedder = VoyageEmbedder(api_key=os.getenv("VOYAGE_API_KEY", ""))
    vec = embedder.embed_query("xin chào")
    print(f"dim={len(vec)} first5={vec[:5]}")
