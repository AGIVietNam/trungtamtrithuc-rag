from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.voyage_embed import VoyageEmbedder
from app.core.qdrant_store import QdrantStore, VMediaReadOnlyStore, QdrantRegistry

logger = logging.getLogger(__name__)


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
    ) -> list[Hit]:
        col = getattr(store, "collection", "vmedia")
        try:
            hits = store.search(query_vec, limit=top_k, filter=qdrant_filter)
            logger.info(
                "retriever _search_one %s/%s: %d hits (filter=%s)",
                col, source_type, len(hits),
                bool(qdrant_filter),
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
        except Exception:
            logger.exception(
                "retriever _search_one FAILED: collection=%s source_type=%s "
                "filter=%s",
                col, source_type, qdrant_filter,
            )
            return []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        domain: str | None = None,          # slug: "bim", "cong-nghe", None=chat chung
        sources: list[str] | None = None,
        query_vec: list[float] | None = None,
    ) -> list[Hit]:
        if sources is None:
            sources = ["documents", "videos"]
        if query_vec is None:
            query_vec = self.voyage.embed_query(query)

        # ── Chọn stores theo domain ──────────────────────────────────────────
        # domain truyền vào từ chat.py thường là persona key
        # ("bim", "công nghệ thông tin", ...). get_by_persona chấp nhận cả
        # persona key lẫn slug ("cong_nghe"), nên an toàn 2 chiều.
        # domain=None → nhánh fanout (chat chung, user không chọn lĩnh vực).
        if domain:
            try:
                doc_stores  = [self.registry.get_by_persona(domain, "docs")]   if "documents" in sources else []
                vid_stores  = [self.registry.get_by_persona(domain, "videos")] if "videos"    in sources else []
            except KeyError:
                logger.warning(
                    "retriever: domain=%r không map được — fallback fanout 20 collections",
                    domain,
                )
                doc_stores  = self.registry.all_docs_stores()   if "documents" in sources else []
                vid_stores  = self.registry.all_videos_stores() if "videos"    in sources else []
        else:
            # Chat chung → search song song toàn bộ 20 collections
            doc_stores  = self.registry.all_docs_stores()   if "documents" in sources else []
            vid_stores  = self.registry.all_videos_stores() if "videos"    in sources else []

        # ── Build tasks — không còn qdrant_filter vì đã routing bằng collection ──
        tasks: list[tuple] = []
        for store in doc_stores:
            tasks.append((store, "document", query_vec, top_k, None))
        for store in vid_stores:
            tasks.append((store, "video", query_vec, top_k, None))
        if "vmedia" in sources and self.vmedia_store.api_key:
            tasks.append((self.vmedia_store, "vmedia", query_vec, top_k, None))

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
        logger.info(
            "retrieve: domain=%r tasks=%d raw=%d deduped=%d returned=%d top_score=%.4f",
            domain, len(tasks), len(all_hits), len(deduped), len(result),
            result[0].score if result else 0.0,
        )
        return result