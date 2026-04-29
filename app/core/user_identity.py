"""In-memory store cho identity user TỰ KHAI BÁO — keyed by user_id (cross-session).

Vai trò: cầu nối giữa chế độ "no auth" (test, dev) và "có auth" (backend wire xong).

  * No-auth: user nói "Tôi là Quốc Tuấn" trong session 1 → AI lưu theo user_id.
    Session 2/3/... cùng user_id đọc lại → "Tôi là ai?" trả "Bạn là Quốc Tuấn".
  * Có auth: ChatRequest.user_name từ JWT/IUser.fullName là source of truth, store
    này thành fallback (chỉ dùng khi auth name rỗng).

Key = user_id để CROSS-SESSION cho cùng 1 user. Trước đây key theo session_id
gây bug "đổi session là quên tên" — không phù hợp model auth-per-account.

KHÔNG persist qua restart — đúng spirit "AI module agnostic, backend own user data".
TTL 7 ngày + LRU 5k users để chống memory leak (TTL dài hơn session vì user
quay lại sau nhiều ngày vẫn nên nhớ).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict


_DEFAULT_TTL_SECONDS = 7 * 24 * 3600   # 7 ngày
_DEFAULT_MAX_USERS = 5000
_NAME_MAX_LEN = 60


class UserIdentityStore:
    """LRU dict[user_id, {name, role, introduced_at}] — thread-safe, in-memory.

    `introduced_at` là epoch seconds. Entry hết hạn khi `time.time() - introduced_at > ttl`.
    """

    def __init__(
        self,
        max_users: int = _DEFAULT_MAX_USERS,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ):
        self._data: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_users
        self._ttl = ttl_seconds

    def get(self, user_id: str) -> dict:
        """Trả shallow copy của entry. {} nếu không có hoặc đã hết TTL."""
        if not user_id:
            return {}
        with self._lock:
            entry = self._data.get(user_id)
            if not entry:
                return {}
            if time.time() - entry.get("introduced_at", 0) > self._ttl:
                del self._data[user_id]
                return {}
            self._data.move_to_end(user_id)
            return dict(entry)

    def get_name(self, user_id: str) -> str:
        return self.get(user_id).get("name", "")

    def set_name(self, user_id: str, name: str) -> None:
        """Idempotent — cùng tên 2 lần chỉ refresh TTL, không bump LRU thrash."""
        if not user_id or not name:
            return
        clean = name.strip()[:_NAME_MAX_LEN]
        if not clean:
            return
        with self._lock:
            entry = self._data.get(user_id, {})
            if entry.get("name") == clean:
                entry["introduced_at"] = int(time.time())
                self._data.move_to_end(user_id)
                return
            entry["name"] = clean
            entry["introduced_at"] = int(time.time())
            self._data[user_id] = entry
            self._data.move_to_end(user_id)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self, user_id: str) -> None:
        if not user_id:
            return
        with self._lock:
            self._data.pop(user_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
