"""Qdrant client factory + collection helpers."""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core import config

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return _client


def ensure_collection(client: QdrantClient, name: str, dim: int | None = None) -> None:
    """Create collection if it doesn't exist."""
    dim = dim or config.VOYAGE_DIM
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    id_offset: int = 0,
) -> int:
    """Upsert chunk payloads with their vectors. Returns count upserted."""
    from qdrant_client.models import PointStruct

    ensure_collection(client, collection)
    points = [
        PointStruct(id=id_offset + i, vector=emb, payload=chunk)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    if points:
        client.upsert(collection_name=collection, points=points)
    return len(points)


def next_id(client: QdrantClient, collection: str) -> int:
    """Return next available integer ID for a collection."""
    try:
        info = client.get_collection(collection)
        return info.points_count or 0
    except Exception:
        return 0
