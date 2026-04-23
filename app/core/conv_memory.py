"""Conversation Memory — vector recall cross-session cho mỗi user.

Lưu mỗi pair (user + bot) như 1 point trong Qdrant collection ttt_memory.
Payload: {conv_id, user_id, session_id, turn_idx, text, created_at, domain}.

Recall:
  - filter must: user_id
  - top-K + score threshold
  - exclude các pair cùng session gần đây (tránh recall chính cái đang ở
    trong sliding window)

Graceful: mọi lỗi đều log rồi trả về list rỗng / False, KHÔNG raise
ra ngoài để không làm fail chain chat.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from collections import OrderedDict
from typing import Any

import requests

from app.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    CONV_COLLECTION,
    CONV_RECALL_TOP_K,
    CONV_RECALL_MIN_SCORE,
    CONV_MIN_USER_CHARS,
    CONV_MIN_BOT_CHARS,
    CONV_DEDUP_THRESHOLD,
    CONV_HASH_CACHE_SIZE,
    VOYAGE_DIM,
)
from app.core.voyage_embed import VoyageEmbedder

logger = logging.getLogger(__name__)

MAX_PAIR_CHARS = 3000  # ~800 tokens, truncate trước khi embed

# Regex chặn câu xã giao tiếng Việt + Anh. Anchor `\W*$` để chỉ match cả câu
# (không chặn nhầm "ok bạn ơi cho hỏi...").
_SKIP_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        r"^(xin chào|chào|hello|hi|hey)\W*$",
        r"^(chào buổi (sáng|chiều|tối))\W*$",
        r"^(cảm ơn|cám ơn|thank|thanks|tks|thx)\W*$",
        r"^(ok|oke|okay|được|được rồi|đã hiểu|hiểu rồi)\W*$",
        r"^(vâng|dạ|ừ|uhm|ah|ờ|à)\W*$",
        r"^(tốt lắm|hay quá|tuyệt|great|nice)\W*$",
        r"^(tạm biệt|bye|goodbye|hẹn gặp lại)\W*$",
    )
]


def _format_pair_text(user_text: str, assistant_text: str) -> str:
    """Tạo text để embed + inject vào prompt."""
    user_text = (user_text or "").strip()
    assistant_text = (assistant_text or "").strip()
    combined = f"USER: {user_text}\nBOT: {assistant_text}"
    if len(combined) > MAX_PAIR_CHARS:
        combined = combined[:MAX_PAIR_CHARS] + "…"
    return combined


def _is_worth_storing(user_msg: str, bot_msg: str) -> tuple[bool, str]:
    """Heuristic filter — loại pair không đáng lưu vào vector memory.

    Returns:
        (True, "") nếu đáng lưu.
        (False, "<lý do>") nếu skip — caller log lý do.

    Ba lớp lọc nối tiếp:
      A. Length gate — câu quá ngắn hoặc bot chỉ chitchat (không có "Nguồn:")
      B. Pattern gate — regex câu xã giao / acknowledgement
      C. Density gate — câu ngắn + không có số + không có từ dài → thiếu thông tin
    """
    u = (user_msg or "").strip()
    b = (bot_msg or "").strip()

    if len(u) < CONV_MIN_USER_CHARS:
        return False, f"user_msg quá ngắn ({len(u)} < {CONV_MIN_USER_CHARS})"

    if len(b) < CONV_MIN_BOT_CHARS and "nguồn:" not in b.lower():
        return False, f"bot_msg chitchat ({len(b)} chars, no 'Nguồn:')"

    for pat in _SKIP_PATTERNS:
        if pat.match(u):
            return False, f"user_msg match skip pattern: {pat.pattern}"

    # Layer C: information density — câu < 6 từ, không có số, không có từ > 5 ký tự
    words = u.split()
    if len(words) < 6:
        has_digit = any(ch.isdigit() for ch in u)
        has_long_word = any(len(w) > 5 for w in words)
        if not has_digit and not has_long_word:
            return False, "user_msg density thấp (ít từ, không số, không từ dài)"

    return True, ""


def _normalize_for_hash(text: str) -> str:
    """Normalize để hash: lowercase + collapse whitespace + strip punctuation cuối."""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = t.rstrip(".!?,;:… ")
    return t


def _hash_pair(user_msg: str, bot_msg: str) -> str:
    """MD5 của pair đã normalize — dùng làm exact-dup key."""
    key = f"{_normalize_for_hash(user_msg)}||{_normalize_for_hash(bot_msg)}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


class ConversationMemory:
    """CRUD + retrieve cho conversation pairs trong Qdrant ttt_memory.

    Vector name "" (unnamed) theo cấu hình collection hiện có.
    """

    def __init__(self, embedder: VoyageEmbedder | None = None):
        self.url = QDRANT_URL.rstrip("/")
        self.api_key = QDRANT_API_KEY
        self.collection = CONV_COLLECTION
        self.embedder = embedder
        self._vector_name = ""  # match collection config
        # LRU cache hash pair gần nhất theo từng user → chặn exact dup trước khi embed.
        # OrderedDict[user_id, OrderedDict[pair_hash, None]] — simple LRU per user.
        self._hash_cache: OrderedDict[str, OrderedDict[str, None]] = OrderedDict()

    # ------------------------------------------------------------- hash cache

    def _hash_seen(self, user_id: str, pair_hash: str) -> bool:
        """Check+update LRU hash cache. Trả True nếu đã thấy hash này."""
        bucket = self._hash_cache.get(user_id)
        if bucket is None:
            bucket = OrderedDict()
            self._hash_cache[user_id] = bucket
        if pair_hash in bucket:
            bucket.move_to_end(pair_hash)
            return True
        bucket[pair_hash] = None
        # Trim bucket + global để tổng không vượt CONV_HASH_CACHE_SIZE
        while len(bucket) > CONV_HASH_CACHE_SIZE:
            bucket.popitem(last=False)
        total = sum(len(b) for b in self._hash_cache.values())
        while total > CONV_HASH_CACHE_SIZE * 4 and self._hash_cache:
            oldest_user = next(iter(self._hash_cache))
            self._hash_cache.pop(oldest_user)
            total = sum(len(b) for b in self._hash_cache.values())
        return False

    # ---------------------------------------------------------- semantic dedup

    def _find_near_duplicate(
        self,
        user_id: str,
        vec: list[float],
        threshold: float,
    ) -> dict | None:
        """Search pair cũ cùng user có cosine >= threshold. Trả hit đầu tiên hoặc None."""
        try:
            body = {
                "vector": {"name": self._vector_name, "vector": vec},
                "limit": 1,
                "with_payload": False,
                "filter": {
                    "must": [{"key": "user_id", "match": {"value": user_id}}]
                },
                "score_threshold": threshold,
            }
            result = self._req(
                "POST", f"/collections/{self.collection}/points/search", body
            )
            hits = result.get("result", [])
            return hits[0] if hits else None
        except Exception:
            logger.error("conv_memory near-dup search failed", exc_info=True)
            return None

    def _touch_last_seen(self, point_id: str | int, now: int) -> None:
        """Cập nhật payload last_seen_at cho point đã có (không tăng hit_count
        để tránh race condition — Qdrant không có atomic increment)."""
        try:
            self._req(
                "POST",
                f"/collections/{self.collection}/points/payload?wait=false",
                {"payload": {"last_seen_at": now}, "points": [point_id]},
            )
        except Exception:
            logger.error("conv_memory touch last_seen_at failed", exc_info=True)

    def _headers(self) -> dict:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def _req(self, method: str, path: str, body: Any = None, timeout: int = 30) -> dict:
        resp = requests.request(
            method, f"{self.url}{path}",
            headers=self._headers(), json=body, timeout=timeout,
        )
        if not resp.ok:
            logger.error("qdrant %s %s -> %s %s", method, path, resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    def _embed(self, text: str) -> list[float]:
        if self.embedder is None:
            raise RuntimeError("ConversationMemory: embedder chưa được inject")
        return self.embedder.embed_query(text)

    # -------------------------------------------------------------- bootstrap
    def ensure_collection(self) -> None:
        """Tạo ttt_memory collection nếu chưa tồn tại (idempotent).

        Qdrant KHÔNG auto-create collection khi upsert — phải tạo trước.
        Trước đây upsert đầu tiên sẽ fail 404, pair bị mất lặng lẽ cho tới
        khi ai đó tạo collection thủ công. Gọi 1 lần lúc startup là đủ.

        Vector name để rỗng ("") theo convention đã dùng ở upsert/search.
        """
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

            body = {
                "vectors": {
                    self._vector_name: {
                        "size": VOYAGE_DIM,
                        "distance": "Cosine",
                    }
                }
            }
            self._req("PUT", f"/collections/{self.collection}", body)
            logger.info("conv_memory collection '%s' created", self.collection)
        except Exception:
            logger.exception(
                "conv_memory ensure_collection failed — "
                "upsert/recall sẽ bị bỏ qua tới khi collection tồn tại",
            )

    def ensure_indexes(self) -> None:
        """Tạo collection + payload index cho user_id + session_id (idempotent).

        Qdrant yêu cầu payload index để filter exact match. Khi chưa có index,
        mọi search có `filter.must.user_id` đều fail 400 — bị nuốt im lặng bởi
        try/except ở `retrieve()` và `_find_near_duplicate()`. Gọi 1 lần lúc
        startup là đủ — Qdrant trả 200 idempotent nếu index đã tồn tại.
        """
        self.ensure_collection()
        for field in ("user_id", "session_id"):
            try:
                self._req(
                    "PUT",
                    f"/collections/{self.collection}/index?wait=true",
                    {"field_name": field, "field_schema": "keyword"},
                )
                logger.info("conv_memory ensured payload index: %s", field)
            except Exception as exc:
                logger.warning(
                    "conv_memory ensure_indexes(%s) skipped: %s", field, exc
                )

    @staticmethod
    def should_skip_recall(query: str) -> bool:
        """True nếu query không đáng tốn 1 Qdrant call để recall.

        Reuse `_SKIP_PATTERNS` (greeting/ack/yes-no) đã dùng cho upsert filter.
        Câu chào / "ok" / "cảm ơn" → recall ra pair lạc đề chỉ làm noise prompt.
        """
        q = (query or "").strip()
        if len(q) < 6:
            return True
        for pat in _SKIP_PATTERNS:
            if pat.match(q):
                return True
        return False

    # ------------------------------------------------------------------ upsert

    def upsert_pair(
        self,
        user_id: str,
        session_id: str,
        turn_idx: int,
        user_text: str,
        assistant_text: str,
        domain: str = "mac_dinh",
    ) -> bool:
        """Embed pair và upsert với 3 lớp chống bloat.

        Trả True nếu thực sự ghi point mới vào Qdrant.
        Trả False nếu bị 1 trong 3 guard chặn (heuristic / hash dup / semantic dup).
        """
        try:
            pair_text = _format_pair_text(user_text, assistant_text)
            if len(pair_text) < 10:
                return False

            # Guard 1 — heuristic filter (zero-cost, chặn câu xã giao)
            worth, reason = _is_worth_storing(user_text, assistant_text)
            if not worth:
                logger.info(
                    "conv_memory skip upsert (heuristic): user=%s turn=%d — %s",
                    user_id, turn_idx, reason,
                )
                return False

            # Guard 2 — exact hash dup (zero-cost, chặn trước khi tốn embed)
            pair_hash = _hash_pair(user_text, assistant_text)
            if self._hash_seen(user_id, pair_hash):
                logger.info(
                    "conv_memory skip upsert (hash dup): user=%s turn=%d hash=%s",
                    user_id, turn_idx, pair_hash[:8],
                )
                return False

            # Embed 1 lần, reuse cho cả dedup search và upsert
            vec = self._embed(pair_text)
            now = int(time.time())

            # Guard 3 — semantic dup (cosine >= threshold với pair cũ cùng user)
            near = self._find_near_duplicate(user_id, vec, CONV_DEDUP_THRESHOLD)
            if near is not None:
                near_id = near.get("id")
                self._touch_last_seen(near_id, now)
                logger.info(
                    "conv_memory skip upsert (semantic dup): user=%s turn=%d "
                    "score=%.3f existing=%s",
                    user_id, turn_idx, near.get("score", 0.0), near_id,
                )
                return False

            # Thực sự ghi point mới
            conv_id = f"conv_{uuid.uuid4().hex[:12]}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, conv_id))

            payload = {
                "conv_id": conv_id,
                "user_id": user_id,
                "session_id": session_id,
                "turn_idx": turn_idx,
                "text": pair_text,
                "created_at": now,
                "last_seen_at": now,
                "domain": domain or "mac_dinh",
                "kind": "conversation_pair",
            }

            point = {
                "id": point_id,
                "vector": {self._vector_name: vec},
                "payload": payload,
            }
            self._req(
                "PUT",
                f"/collections/{self.collection}/points?wait=false",
                {"points": [point]},
            )
            logger.info(
                "conv_memory upsert: user=%s session=%s turn=%d len=%d",
                user_id, session_id, turn_idx, len(pair_text),
            )
            return True
        except Exception:
            logger.error("conv_memory upsert failed", exc_info=True)
            return False

    # ---------------------------------------------------------------- retrieve

    def retrieve(
        self,
        user_id: str,
        query: str,
        current_session_id: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        """Vector search trong scope user_id. Trả về list payload đã sort.

        Luôn thành công ngay cả khi Qdrant lỗi (trả [] ).

        ``query_vec`` (optional): nếu caller đã embed query rồi (vd chain.py
        embed 1 lần dùng cho cả retriever + recall) thì truyền vào để khỏi
        embed lại — tiết kiệm 1 Voyage call/turn.
        """
        if not user_id or not query:
            return []

        k = top_k or CONV_RECALL_TOP_K
        threshold = score_threshold if score_threshold is not None else CONV_RECALL_MIN_SCORE

        try:
            vec = query_vec if query_vec is not None else self._embed(query)

            must: list[dict] = [
                {"key": "user_id", "match": {"value": user_id}},
            ]
            # Chỉ recall record kiểu conversation_pair (không ăn nhầm record khác
            # nếu collection ttt_memory có dữ liệu loại khác trong tương lai).
            # Dùng should để coexist cả point cũ không có field kind.
            body = {
                "vector": {"name": self._vector_name, "vector": vec},
                "limit": k * 2,  # lấy dư để lọc session & kind
                "with_payload": True,
                "filter": {"must": must},
                "score_threshold": threshold,
            }

            result = self._req(
                "POST", f"/collections/{self.collection}/points/search", body
            )
            hits = result.get("result", [])

            out: list[dict] = []
            seen_texts: set[str] = set()
            for h in hits:
                payload = h.get("payload", {}) or {}
                # Bỏ record không phải conversation_pair
                if payload.get("kind") not in (None, "conversation_pair"):
                    continue
                # Exclude pairs từ session hiện tại (tránh recall lại chính sliding window)
                if current_session_id and payload.get("session_id") == current_session_id:
                    continue
                # Dedup text gần giống nhau
                text_key = (payload.get("text") or "")[:120]
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                out.append({
                    "score": h.get("score", 0.0),
                    "text": payload.get("text", ""),
                    "session_id": payload.get("session_id", ""),
                    "turn_idx": payload.get("turn_idx", 0),
                    "created_at": payload.get("created_at", 0),
                    "domain": payload.get("domain", ""),
                })
                if len(out) >= k:
                    break

            # Sort theo created_at tăng dần (cũ → mới) để LLM đọc dễ ưu tiên mới
            out.sort(key=lambda p: p.get("created_at", 0))
            logger.info(
                "conv_memory retrieve: user=%s query_len=%d → %d hits (threshold=%.2f)",
                user_id, len(query), len(out), threshold,
            )
            return out
        except Exception:
            logger.error("conv_memory retrieve failed", exc_info=True)
            return []

    # ------------------------------------------------------------------ delete

    def delete_by_user(self, user_id: str) -> int:
        """GDPR: xoá mọi pair của user. Trả số points bị xoá (ước tính)."""
        if not user_id:
            return 0
        try:
            self._req(
                "POST",
                f"/collections/{self.collection}/points/delete?wait=true",
                {
                    "filter": {
                        "must": [{"key": "user_id", "match": {"value": user_id}}]
                    }
                },
            )
            logger.info("conv_memory deleted all pairs for user=%s", user_id)
            return 1
        except Exception:
            logger.error("conv_memory delete_by_user failed", exc_info=True)
            return 0

    def delete_by_session(self, session_id: str) -> int:
        if not session_id:
            return 0
        try:
            self._req(
                "POST",
                f"/collections/{self.collection}/points/delete?wait=true",
                {
                    "filter": {
                        "must": [{"key": "session_id", "match": {"value": session_id}}]
                    }
                },
            )
            logger.info("conv_memory deleted pairs for session=%s", session_id)
            return 1
        except Exception:
            logger.error("conv_memory delete_by_session failed", exc_info=True)
            return 0
