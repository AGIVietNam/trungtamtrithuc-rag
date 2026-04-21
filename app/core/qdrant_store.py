from __future__ import annotations

import logging
import uuid
from typing import Any

import requests

logger = logging.getLogger(__name__)

UPSERT_BATCH = 64


class QdrantStore:
    """Read-write store for ttt_documents and ttt_videos collections."""

    def __init__(
        self,
        url: str,
        api_key: str,
        collection: str,
        vector_size: int = 1024,
        vector_name: str = "",
    ):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.collection = collection
        self.vector_size = vector_size
        self.vector_name = vector_name

    def _headers(self) -> dict:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def _req(self, method: str, path: str, json_body: Any = None) -> Any:
        resp = requests.request(
            method,
            f"{self.url}{path}",
            headers=self._headers(),
            json=json_body,
            timeout=60,
        )
        if not resp.ok:
            logger.error("qdrant %s %s -> %s: %s", method, path, resp.status_code, resp.text)
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    def ensure_collection(self) -> None:
        r = requests.get(
            f"{self.url}/collections/{self.collection}",
            headers=self._headers(),
            timeout=30,
        )
        if r.status_code == 200:
            return
        if r.status_code != 404:
            r.raise_for_status()
        body = {
            "vectors": {
                self.vector_name: {
                    "size": self.vector_size,
                    "distance": "Cosine",
                    "on_disk": False,
                    "hnsw_config": {"m": 24, "payload_m": 24, "ef_construct": 256},
                    "datatype": "float32",
                }
            }
        }
        self._req("PUT", f"/collections/{self.collection}", body)
        logger.info("qdrant collection '%s' created", self.collection)

    def upsert(self, points: list[dict], wait: bool = True) -> None:
        for i in range(0, len(points), UPSERT_BATCH):
            batch = points[i : i + UPSERT_BATCH]
            formatted = []
            for p in batch:
                formatted.append(
                    {
                        "id": p.get("id", str(uuid.uuid4())),
                        "vector": {self.vector_name: p["vector"]},
                        "payload": p.get("payload", {}),
                    }
                )
            self._req(
                "PUT",
                f"/collections/{self.collection}/points?wait={str(wait).lower()}",
                {"points": formatted},
            )
            logger.info("qdrant upserted %d points to '%s'", len(batch), self.collection)

    def search(
        self,
        query_vector: list[float],
        limit: int = 7,
        filter: dict | None = None,
        with_payload: bool = True,
    ) -> list[dict]:
        body: dict[str, Any] = {
            "vector": {"name": self.vector_name, "vector": query_vector},
            "limit": limit,
            "with_payload": with_payload,
        }
        if filter:
            body["filter"] = filter
        result = self._req("POST", f"/collections/{self.collection}/points/search", body)
        return result.get("result", [])

    def scroll(self, filter: dict | None = None, limit: int = 100) -> list[dict]:
        body: dict[str, Any] = {"limit": limit, "with_payload": True}
        if filter:
            body["filter"] = filter
        result = self._req("POST", f"/collections/{self.collection}/points/scroll", body)
        return result.get("result", {}).get("points", [])

    def delete_by_filter(self, filter: dict) -> None:
        self._req("POST", f"/collections/{self.collection}/points/delete", {"filter": filter})


class VMediaReadOnlyStore:
    """Read-only store for vmedia collections on a SEPARATE Qdrant cluster.

    Searches across multiple vmedia_* collections and merges results.
    Uses QDRANT_VMEDIA_API_KEY + QDRANT_VMEDIA_URL — NEVER writes.
    """

    def __init__(
        self,
        url: str,
        vmedia_api_key: str,
        collections: list[str],
        vector_name: str = "",
    ):
        self.url = url.rstrip("/")
        self.api_key = vmedia_api_key
        self.collections = collections
        self.vector_name = vector_name

    def _headers(self) -> dict:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        filter: dict | None = None,
        with_payload: bool = True,
    ) -> list[dict]:
        all_results: list[dict] = []
        per_collection_limit = max(3, limit // len(self.collections) + 1) if self.collections else limit
        for col in self.collections:
            # vmedia collections use unnamed vectors → send flat array
            vec_payload = query_vector if not self.vector_name else {"name": self.vector_name, "vector": query_vector}
            body: dict[str, Any] = {
                "vector": vec_payload,
                "limit": per_collection_limit,
                "with_payload": with_payload,
            }
            if filter:
                body["filter"] = filter
            try:
                resp = requests.post(
                    f"{self.url}/collections/{col}/points/search",
                    headers=self._headers(),
                    json=body,
                    timeout=30,
                )
                if resp.ok:
                    for hit in resp.json().get("result", []):
                        hit["_vmedia_collection"] = col
                        all_results.append(hit)
            except Exception as exc:
                logger.warning("vmedia search '%s' failed: %s", col, exc)
        all_results.sort(key=lambda h: h.get("score", 0), reverse=True)
        return all_results[:limit]

    def upsert(self, *args, **kwargs):
        raise RuntimeError("VMediaReadOnlyStore: upsert không được phép. Key vmedia chỉ đọc.")

    def ensure_collection(self, *args, **kwargs):
        raise RuntimeError("VMediaReadOnlyStore: ensure_collection không được phép. Key vmedia chỉ đọc.")

    def delete(self, *args, **kwargs):
        raise RuntimeError("VMediaReadOnlyStore: delete không được phép. Key vmedia chỉ đọc.")
