from __future__ import annotations

import logging
import uuid
from typing import Any

import requests

logger = logging.getLogger(__name__)

UPSERT_BATCH = 64

# ── 10 domain slugs (ASCII) — khớp với NestJS Categories seeder ─────────────
# Source: knowledge_center_backend/src/database/seeds/categories.seeder.ts
# Upload form bắt buộc chọn 1 trong 10 slug. Chat chung fanout toàn bộ.
DOMAINS: list[str] = [
    "marketing",
    "mep",
    "bim",
    "phap_ly",
    "san_xuat",
    "cntt",
    "nhan_su",
    "tai_chinh",
    "kinh_doanh",
    "thiet_ke",
]

# Legacy persona key (Vietnamese có dấu) → slug. Dùng cho backward compat khi
# client cũ vẫn gửi persona VN ("công nghệ thông tin") thay vì slug ("cntt").
# Không có entry "mặc định" vì không có collection tương ứng — chat chung đi
# qua nhánh fanout trong Retriever (domain=None).
PERSONA_TO_DOMAIN: dict[str, str] = {
    "bim":                   "bim",
    "mep":                   "mep",
    "marketing":             "marketing",
    "pháp lý":               "phap_ly",
    "sản xuất":              "san_xuat",
    "công nghệ thông tin":   "cntt",
    "nhân sự":               "nhan_su",
    "tài chính":             "tai_chinh",
    "kinh doanh":            "kinh_doanh",
    "thiết kế":              "thiet_ke",
}

# Reverse map: slug → persona key (dùng khi cần lookup ngược)
DOMAIN_TO_PERSONA: dict[str, str] = {v: k for k, v in PERSONA_TO_DOMAIN.items()}


class QdrantStore:
    """Read-write store cho 1 Qdrant collection cụ thể."""

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

    def ensure_payload_indexes(self, fields: list[str]) -> None:
        """Tạo payload keyword index (idempotent). Không raise khi collection chưa tồn tại."""
        for field in fields:
            try:
                self._req(
                    "PUT",
                    f"/collections/{self.collection}/index?wait=true",
                    {"field_name": field, "field_schema": "keyword"},
                )
                logger.info("qdrant ensured payload index: %s.%s", self.collection, field)
            except Exception as exc:
                logger.warning(
                    "qdrant ensure_payload_indexes(%s.%s) skipped: %s",
                    self.collection, field, exc,
                )

    def upsert(self, points: list[dict], wait: bool = True) -> None:
        for i in range(0, len(points), UPSERT_BATCH):
            batch = points[i : i + UPSERT_BATCH]
            formatted = [
                {
                    "id": p.get("id", str(uuid.uuid4())),
                    "vector": {self.vector_name: p["vector"]},
                    "payload": p.get("payload", {}),
                }
                for p in batch
            ]
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


# ── QdrantRegistry ────────────────────────────────────────────────────────────

class QdrantRegistry:
    """
    Quản lý toàn bộ QdrantStore theo (domain_slug, source_type).

    Tạo sẵn 20 stores = 10 domain × 2 source (docs + videos).
    Collection name format: tdi_docs_{slug} / tdi_videos_{slug}

    Ví dụ:
        registry.get("bim", "docs")          → store cho tdi_docs_bim
        registry.get("cntt", "videos")       → store cho tdi_videos_cntt
        registry.get_by_persona("công nghệ thông tin", "docs")  → tương tự trên
    """

    # 20 collections
    # tdi_docs_bim          tdi_videos_bim
    # tdi_docs_mep          tdi_videos_mep
    # tdi_docs_marketing    tdi_videos_marketing
    # tdi_docs_phap_ly      tdi_videos_phap_ly
    # tdi_docs_san_xuat     tdi_videos_san_xuat
    # tdi_docs_cong_nghe    tdi_videos_cong_nghe
    # tdi_docs_nhan_su      tdi_videos_nhan_su
    # tdi_docs_tai_chinh    tdi_videos_tai_chinh
    # tdi_docs_kinh_doanh   tdi_videos_kinh_doanh
    # tdi_docs_thiet_ke     tdi_videos_thiet_ke

    def __init__(
        self,
        url: str,
        api_key: str,
        vector_size: int = 1024,
        vector_name: str = "",
    ):
        self.url = url
        self.api_key = api_key
        self._stores: dict[str, QdrantStore] = {}

        for domain in DOMAINS:
            for source in ("docs", "videos"):
                key = f"{domain}__{source}"
                self._stores[key] = QdrantStore(
                    url=url,
                    api_key=api_key,
                    collection=f"tdi_{source}_{domain}",
                    vector_size=vector_size,
                    vector_name=vector_name,
                )

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def get(self, domain: str, source: str) -> QdrantStore:
        """
        Lấy store theo domain slug + source type.
            registry.get("bim", "docs")
            registry.get("tong_quat", "videos")
        Raise KeyError nếu domain/source không hợp lệ.
        """
        return self._stores[f"{domain}__{source}"]

    def get_by_persona(self, persona_key: str, source: str) -> QdrantStore:
        """
        Dùng trực tiếp key từ DOMAIN_PERSONAS — tiện khi nhận domain từ FE.
            registry.get_by_persona("công nghệ thông tin", "docs")
            registry.get_by_persona("bim", "videos")

        Nhận cả persona key ("công nghệ thông tin") lẫn slug ("cntt").
        Raise KeyError nếu không nhận ra — chat chung phải đi qua Retriever
        với domain=None, không được rơi vào hàm này.
        """
        slug = PERSONA_TO_DOMAIN.get(persona_key, persona_key)
        return self.get(slug, source)

    def stores_for_domain(self, domain: str) -> list[QdrantStore]:
        """Trả [docs_store, videos_store] cho 1 domain slug."""
        return [self.get(domain, "docs"), self.get(domain, "videos")]

    def all_docs_stores(self) -> list[QdrantStore]:
        return [v for k, v in self._stores.items() if k.endswith("__docs")]

    def all_videos_stores(self) -> list[QdrantStore]:
        return [v for k, v in self._stores.items() if k.endswith("__videos")]

    def all_stores(self) -> list[QdrantStore]:
        return list(self._stores.values())

    def collection_names(self) -> list[str]:
        """Danh sách 24 tên collection — dùng để log/debug."""
        return [s.collection for s in self._stores.values()]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def ensure_all(self) -> None:
        """
        Gọi 1 lần lúc lifespan startup.
        """
        created = 0
        for store in self._stores.values():
            try:
                store.ensure_collection()
                created += 1
            except Exception:
                logger.exception("ensure_all: failed for collection '%s'", store.collection)

        logger.info(
            "QdrantRegistry.ensure_all: %d/%d collections ensured. Names: %s",
            created,
            len(self._stores),
            ", ".join(self.collection_names()),
        )


# ── VMediaReadOnlyStore (giữ nguyên) ─────────────────────────────────────────

class VMediaReadOnlyStore:
    """
    Read-only store cho vmedia collections trên cluster RIÊNG.
    Không bao giờ ghi.
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
        per_col_limit = max(3, limit // len(self.collections) + 1) if self.collections else limit

        for col in self.collections:
            vec_payload = (
                query_vector
                if not self.vector_name
                else {"name": self.vector_name, "vector": query_vector}
            )
            body: dict[str, Any] = {
                "vector": vec_payload,
                "limit": per_col_limit,
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
        raise RuntimeError("VMediaReadOnlyStore: ensure_collection không được phép.")

    def delete(self, *args, **kwargs):
        raise RuntimeError("VMediaReadOnlyStore: delete không được phép.")