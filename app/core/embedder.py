"""Voyage AI embedder wrapper."""
from __future__ import annotations

import voyageai
from app.core import config


class VoyageEmbedder:
    def __init__(self) -> None:
        self._client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
        self.model = config.VOYAGE_MODEL
        self.dim = config.VOYAGE_DIM

    def embed(self, texts: list[str], input_type: str = "query") -> list[list[float]]:
        """Embed a list of texts. input_type: 'query' or 'document'."""
        if not texts:
            return []
        result = self._client.embed(texts, model=self.model, input_type=input_type)
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], input_type="query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts, input_type="document")
