from __future__ import annotations

import uuid
from typing import Any

import requests

from app import config

UPSERT_BATCH = 64

# Named vector IDs cho hybrid retrieval.
# - "dense":  Voyage 1024-dim, distance Cosine.
# - "sparse": BM25 với modifier IDF — Qdrant tự nhân IDF lúc query.
DENSE_VECTOR_NAME: str = "dense"
SPARSE_VECTOR_NAME: str = "sparse"

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
    """Read-write store cho 1 Qdrant collection — hybrid native (dense + sparse).

    Schema collection được tạo bởi ``ensure_collection``:
        vectors:        { "dense":  size=1024, Cosine, HNSW }
        sparse_vectors: { "sparse": modifier="idf" }

    Ingest call ``upsert(points)`` với mỗi point có ``vector`` (dense list)
    và ``sparse`` (dict {indices, values}) — sparse có thể None để skip.

    Search dùng Qdrant Query API với prefetch fusion RRF khi cả 2 vector có
    sẵn; nếu sparse query None hoặc HYBRID_RETRIEVAL=0 → fallback dense-only
    qua cùng Query API (vẫn 1 HTTP call).
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        collection: str,
        vector_size: int = 1024,
    ):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.collection = collection
        self.vector_size = vector_size

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
            print(f"qdrant {method} {path} -> {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    def ensure_collection(self) -> None:
        try:
            r = requests.get(
                f"{self.url}/collections/{self.collection}",
                headers=self._headers(),
                timeout=30,
            )
            if r.status_code == 200:
                return
            if r.status_code != 404:
                r.raise_for_status()
        except Exception as e:
            print(f"FAILED to check collection '{self.collection}' at {self.url}: {e}")
            raise
        body = {
            "vectors": {
                DENSE_VECTOR_NAME: {
                    "size": self.vector_size,
                    "distance": "Cosine",
                    "on_disk": False,
                    "hnsw_config": {"m": 24, "payload_m": 24, "ef_construct": 256},
                    "datatype": "float32",
                }
            },
            "sparse_vectors": {
                SPARSE_VECTOR_NAME: {
                    "modifier": "idf",
                    "index": {"on_disk": False},
                }
            },
        }
        self._req("PUT", f"/collections/{self.collection}", body)
        print(f"qdrant collection '{self.collection}' created (hybrid: dense+sparse)")

    def ensure_payload_indexes(self, fields: list[str]) -> None:
        """Tạo payload keyword index (idempotent). Không raise khi collection chưa tồn tại."""
        for field in fields:
            try:
                self._req(
                    "PUT",
                    f"/collections/{self.collection}/index?wait=true",
                    {"field_name": field, "field_schema": "keyword"},
                )
                print(f"qdrant ensured payload index: {self.collection}.{field}")
            except Exception as exc:
                print(f"qdrant ensure_payload_indexes({self.collection}.{field}) skipped: {exc}")

    def upsert(self, points: list[dict], wait: bool = True) -> None:
        """Upsert points với dense + (optional) sparse vector.

        Mỗi point dict format:
            { id, vector: [1024 floats], sparse: {indices, values} | None,
              payload: {...} }

        Sparse None hoặc rỗng → chỉ upsert dense (point vẫn searchable qua
        prefetch dense, miss prefetch sparse — RRF tự bỏ).
        """
        # Đảm bảo collection tồn tại trước khi đẩy dữ liệu (phòng trường hợp bị xoá khi server đang chạy)
        self.ensure_collection()

        for i in range(0, len(points), UPSERT_BATCH):
            batch = points[i : i + UPSERT_BATCH]
            formatted = []
            for p in batch:
                vec_dict: dict[str, Any] = {DENSE_VECTOR_NAME: p["vector"]}
                sparse = p.get("sparse")
                if sparse and sparse.get("indices"):
                    vec_dict[SPARSE_VECTOR_NAME] = sparse
                formatted.append({
                    "id": p.get("id", str(uuid.uuid4())),
                    "vector": vec_dict,
                    "payload": p.get("payload", {}),
                })
            self._req(
                "PUT",
                f"/collections/{self.collection}/points?wait={str(wait).lower()}",
                {"points": formatted},
            )
            print(f"qdrant upserted {len(batch)} points to '{self.collection}'")

    def search(
        self,
        query_vector: list[float],
        limit: int = 7,
        filter: dict | None = None,
        with_payload: bool = True,
        sparse_vector: dict | None = None,
    ) -> list[dict]:
        """Hybrid search qua Query API.

        Khi ``sparse_vector`` có giá trị + HYBRID_RETRIEVAL bật → 2 prefetch
        (dense + sparse) fuse RRF server-side. Ngược lại → dense-only prefetch.
        Cả hai trường hợp đều 1 HTTP call duy nhất.
        """
        prefetch_limit = config.HYBRID_PREFETCH_LIMIT
        prefetch: list[dict] = [
            {
                "query": query_vector,
                "using": DENSE_VECTOR_NAME,
                "limit": prefetch_limit,
            }
        ]
        use_sparse = (
            config.HYBRID_RETRIEVAL
            and sparse_vector is not None
            and bool(sparse_vector.get("indices"))
        )
        if use_sparse:
            prefetch.append({
                "query": sparse_vector,
                "using": SPARSE_VECTOR_NAME,
                "limit": prefetch_limit,
            })

        body: dict[str, Any] = {
            "prefetch": prefetch,
            "limit": limit,
            "with_payload": with_payload,
        }
        if use_sparse:
            body["query"] = {"fusion": "rrf"}
        else:
            # Dense-only: không cần fusion, chỉ trả prefetch top.
            # Qdrant Query API yêu cầu "query" field — copy dense vec.
            body["query"] = query_vector
            body["using"] = DENSE_VECTOR_NAME
            body.pop("prefetch")
        if filter:
            body["filter"] = filter

        result = self._req("POST", f"/collections/{self.collection}/points/query", body)
        # Query API trả {"result": {"points": [...]}} hoặc {"result": [...]} tuỳ version.
        raw = result.get("result", [])
        if isinstance(raw, dict):
            return raw.get("points", [])
        return raw

    def scroll(self, filter: dict | None = None, limit: int = 100) -> list[dict]:
        body: dict[str, Any] = {"limit": limit, "with_payload": True}
        if filter:
            body["filter"] = filter
        result = self._req("POST", f"/collections/{self.collection}/points/scroll", body)
        return result.get("result", {}).get("points", [])

    def delete_by_filter(self, filter: dict) -> None:
        self.ensure_collection()
        self._req("POST", f"/collections/{self.collection}/points/delete", {"filter": filter})

    def count_by_filter(self, filter: dict, exact: bool = True) -> int:
        """Đếm số point khớp filter. Dùng cho audit trước khi delete."""
        body = {"filter": filter, "exact": exact}
        result = self._req("POST", f"/collections/{self.collection}/points/count", body)
        return int(result.get("result", {}).get("count", 0))


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

    def __init__(
        self,
        url: str,
        api_key: str,
        vector_size: int = 1024,
    ):
        print(f"Initializing QdrantRegistry for cluster: {url}")
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
                )

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def get(self, domain: str, source: str) -> QdrantStore:
        return self._stores[f"{domain}__{source}"]

    def get_by_persona(self, persona_key: str, source: str) -> QdrantStore:
        slug = PERSONA_TO_DOMAIN.get(persona_key, persona_key)
        return self.get(slug, source)

    def stores_for_domain(self, domain: str) -> list[QdrantStore]:
        return [self.get(domain, "docs"), self.get(domain, "videos")]

    def all_docs_stores(self) -> list[QdrantStore]:
        return [v for k, v in self._stores.items() if k.endswith("__docs")]

    def all_videos_stores(self) -> list[QdrantStore]:
        return [v for k, v in self._stores.items() if k.endswith("__videos")]

    def all_stores(self) -> list[QdrantStore]:
        return list(self._stores.values())

    def collection_names(self) -> list[str]:
        return [s.collection for s in self._stores.values()]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def ensure_all(self) -> None:
        created = 0
        for store in self._stores.values():
            try:
                store.ensure_collection()
                store.ensure_payload_indexes(["doc_id", "document_id", "domain", "metadata.domain"])
                created += 1
            except Exception as e:
                print(f"ensure_all: FAILED for collection '{store.collection}': {e}")

        print(
            f"QdrantRegistry.ensure_all: {created}/{len(self._stores)} collections ensured. "
            f"Names: {', '.join(self.collection_names())}"
        )


# ── VMediaReadOnlyStore ──────────────────────────────────────────────────────
# Cluster RIÊNG, schema legacy (single dense vector). Không tham gia hybrid —
# vmedia search vẫn dùng /points/search dense-only như cũ.

class VMediaReadOnlyStore:
    """Read-only store cho vmedia collections trên cluster ngoài (single dense vec)."""

    def __init__(
        self,
        url: str,
        vmedia_api_key: str,
        collections: list[str],
        vector_name: str = "",
    ):
        print(f"Initializing VMediaReadOnlyStore for cluster: {url}")
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
        sparse_vector: dict | None = None,  # signature compat — vmedia ignore sparse
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
                print(f"vmedia search '{col}' failed: {exc}")

        all_results.sort(key=lambda h: h.get("score", 0), reverse=True)
        return all_results[:limit]

    def upsert(self, *args, **kwargs):
        raise RuntimeError("VMediaReadOnlyStore: upsert không được phép. Key vmedia chỉ đọc.")

    def ensure_collection(self, *args, **kwargs):
        raise RuntimeError("VMediaReadOnlyStore: ensure_collection không được phép.")

    def delete(self, *args, **kwargs):
        raise RuntimeError("VMediaReadOnlyStore: delete không được phép.")
