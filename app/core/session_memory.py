"""Session memory with file-backed persistence.

Mỗi session được lưu ở data/sessions/{session_id}.json gồm:
  - history: danh sách message (sliding window)
  - summary: rolling summary của các turn đã rớt khỏi window

Storage layer thuần — không trực tiếp gọi LLM. Caller (chain.py) chịu
trách nhiệm tính summary khi pop_overflow trả về turns.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

SESSIONS_DIR = Path(__file__).parent.parent.parent / "data" / "sessions"


def _session_path(session_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return SESSIONS_DIR / f"{safe}.json"


def _load_from_disk(session_id: str) -> dict:
    path = _session_path(session_id)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "history": data.get("history", []),
                "summary": data.get("summary", ""),
                "turn_count": data.get("turn_count", 0),
            }
        except Exception:
            return {"history": [], "summary": "", "turn_count": 0}
    return {"history": [], "summary": "", "turn_count": 0}


def _save_to_disk(session_id: str, state: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(session_id)
    payload = {
        "session_id": session_id,
        "updated_at": time.time(),
        "history": state.get("history", []),
        "summary": state.get("summary", ""),
        "turn_count": state.get("turn_count", 0),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SessionMemory:
    """Thread-safe (GIL) in-process session store with file persistence."""

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}

    def _state(self, session_id: str) -> dict:
        if session_id not in self._cache:
            self._cache[session_id] = _load_from_disk(session_id)
        return self._cache[session_id]

    # ------------------------------------------------------------------ read

    def get_history(self, session_id: str) -> list[dict]:
        return list(self._state(session_id).get("history", []))

    def get_summary(self, session_id: str) -> str:
        return self._state(session_id).get("summary", "")

    def get_turn_count(self, session_id: str) -> int:
        """Tổng số pair đã xử lý từ đầu session (dùng làm turn_idx)."""
        return int(self._state(session_id).get("turn_count", 0))

    # ------------------------------------------------------------------ write

    def append(self, session_id: str, role: str, content: str) -> None:
        state = self._state(session_id)
        state["history"].append({"role": role, "content": content})
        _save_to_disk(session_id, state)

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> int:
        """Thêm 1 pair user+bot. Trả về turn_idx của pair vừa thêm (1-based).

        KHÔNG tự trim — caller dùng pop_overflow() để quản lý sliding window.
        """
        state = self._state(session_id)
        state["history"].append({"role": "user", "content": user_msg})
        state["history"].append({"role": "assistant", "content": assistant_msg})
        state["turn_count"] = int(state.get("turn_count", 0)) + 1
        _save_to_disk(session_id, state)
        return state["turn_count"]

    def set_summary(self, session_id: str, summary: str) -> None:
        state = self._state(session_id)
        state["summary"] = summary or ""
        _save_to_disk(session_id, state)

    def pop_overflow(self, session_id: str, window_pairs: int) -> list[dict]:
        """Nếu history vượt window_pairs*2 message → cắt phần thừa ở đầu,
        trả về danh sách message bị cắt (để caller gọi summarizer).

        Ví dụ: window_pairs=3, history=10 message (5 pair).
          → giữ 6 message cuối (3 pair gần nhất), trả về 4 message đầu (2 pair).
        """
        state = self._state(session_id)
        history: list[dict] = state.get("history", [])
        max_msgs = window_pairs * 2
        if len(history) <= max_msgs:
            return []
        rolled = history[:-max_msgs]
        state["history"] = history[-max_msgs:]
        _save_to_disk(session_id, state)
        return rolled

    # ---------------------------------------------------------------- manage

    def clear(self, session_id: str) -> None:
        self._cache[session_id] = {"history": [], "summary": "", "turn_count": 0}
        path = _session_path(session_id)
        if path.exists():
            path.unlink()

    def list_sessions(self) -> list[str]:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        return [p.stem for p in SESSIONS_DIR.glob("*.json")]


# Singleton
memory = SessionMemory()
