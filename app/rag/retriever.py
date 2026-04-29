from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any

from app import config
from app.core import sparse_encoder
from app.core.voyage_embed import VoyageEmbedder
from app.core.qdrant_store import QdrantStore, VMediaReadOnlyStore, QdrantRegistry


@dataclass
class Hit:
    text: str
    score: float
    source_type: str  # "document" | "video" | "vmedia"
    payload: dict[str, Any] = field(default_factory=dict)
    collection: str = ""


class Retriever:
    def __init__(
        self,
        voyage: VoyageEmbedder,
        registry: QdrantRegistry,
        vmedia_store: VMediaReadOnlyStore,
    ):
        self.voyage = voyage
        self.registry = registry
        self.vmedia_store = vmedia_store

    def _search_one(
        self,
        store: QdrantStore | VMediaReadOnlyStore,
        source_type: str,
        query_vec: list[float],
        top_k: int,
        qdrant_filter: dict | None = None,
        sparse_vec: dict | None = None,
    ) -> list[Hit]:
        col = getattr(store, "collection", "vmedia")
        try:
            hits = store.search(
                query_vec,
                limit=top_k,
                filter=qdrant_filter,
                sparse_vector=sparse_vec,
            )
            print(
                f"retriever _search_one {col}/{source_type}: {len(hits)} hits "
                f"(filter={bool(qdrant_filter)}, sparse={bool(sparse_vec and sparse_vec.get('indices'))})"
            )
            return [
                Hit(
                    text=h.get("payload", {}).get("text", ""),
                    score=h.get("score", 0.0),
                    source_type=source_type,
                    payload=h.get("payload", {}),
                    collection=h.get("_vmedia_collection", col) if source_type == "vmedia" else col,
                )
                for h in hits
            ]
        except Exception as e:
            print(
                f"retriever _search_one FAILED: collection={col} source_type={source_type} "
                f"filter={qdrant_filter}: {e}"
            )
            return []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        domain: str | None = None,          # slug: "bim", "cong-nghe", None=chat chung
        sources: list[str] | None = None,
        query_vec: list[float] | None = None,
        sparse_vec: dict | None = None,
        doc_id: str | None = None,          # Hạn chế chỉ chunks của 1 doc (E2E test)
    ) -> list[Hit]:
        if sources is None:
            sources = ["documents", "videos"]
        if query_vec is None:
            query_vec = self.voyage.embed_query(query)
        # Sparse query encode lazy — chỉ khi hybrid bật và caller chưa pass.
        # Encode local (~5-20ms) nên hợp lệ trong synchronous path.
        if sparse_vec is None and config.HYBRID_RETRIEVAL:
            sparse_vec = sparse_encoder.encode(query)

        # Filter Qdrant payload — hiện chỉ hỗ trợ doc_id, mở rộng dễ về sau.
        # Áp dụng cho doc/video stores, KHÔNG áp dụng vmedia (collection ngoài).
        doc_filter: dict | None = None
        if doc_id:
            doc_filter = {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}

        # ── Chọn stores theo domain ──────────────────────────────────────────
        if domain:
            try:
                doc_stores  = [self.registry.get_by_persona(domain, "docs")]   if "documents" in sources else []
                vid_stores  = [self.registry.get_by_persona(domain, "videos")] if "videos"    in sources else []
            except KeyError:
                print(f"retriever: domain={domain!r} không map được — fallback fanout 20 collections")
                doc_stores  = self.registry.all_docs_stores()   if "documents" in sources else []
                vid_stores  = self.registry.all_videos_stores() if "videos"    in sources else []
        else:
            # Chat chung → search song song toàn bộ 20 collections
            doc_stores  = self.registry.all_docs_stores()   if "documents" in sources else []
            vid_stores  = self.registry.all_videos_stores() if "videos"    in sources else []

        # ── Build tasks ──
        # Sparse vec only attaches to TDI hybrid stores (docs + videos), KHÔNG
        # pass cho vmedia (cluster ngoài, single-vector legacy schema).
        tasks: list[tuple] = []
        for store in doc_stores:
            tasks.append((store, "document", query_vec, top_k, doc_filter, sparse_vec))
        for store in vid_stores:
            tasks.append((store, "video", query_vec, top_k, doc_filter, sparse_vec))
        if "vmedia" in sources and self.vmedia_store.api_key:
            # vmedia là collection ngoài (Viettel media), không có doc_id
            # của hệ thống ta + không hỗ trợ sparse — bỏ cả 2 filter để tránh
            # search trả 0 hit.
            tasks.append((self.vmedia_store, "vmedia", query_vec, top_k, None, None))

        # ── Search song song ─────────────────────────────────────────────────
        all_hits: list[Hit] = []
        max_workers = min(len(tasks), 12)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self._search_one, *t) for t in tasks]
            for f in concurrent.futures.as_completed(futures):
                all_hits.extend(f.result())

        # ── Dedup + sort ─────────────────────────────────────────────────────
        seen: set[str] = set()
        deduped: list[Hit] = []
        for hit in sorted(all_hits, key=lambda h: h.score, reverse=True):
            key = f"{hit.source_type}::{hit.text[:80]}"
            if key not in seen:
                seen.add(key)
                deduped.append(hit)

        result = deduped[:top_k]
        top_score = result[0].score if result else 0.0
        print(
            f"retrieve: domain={domain!r} tasks={len(tasks)} raw={len(all_hits)} "
            f"deduped={len(deduped)} returned={len(result)} top_score={top_score:.4f} "
            f"hybrid={bool(sparse_vec and sparse_vec.get('indices'))}"
        )
        return result
