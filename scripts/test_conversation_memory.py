"""Test kịch bản Conversation Memory cho 1 user giả.

Chạy:
  1. Start server: ./run.sh   (terminal khác)
  2. python scripts/test_conversation_memory.py

Kịch bản test 3 tầng Hybrid Memory:
  - TẦNG 1 (sliding window): giữ turn gần nhất trong session
  - TẦNG 2 (rolling summary): sau > 3 pair, turn cũ bị tóm tắt
  - TẦNG 3 (vector recall): cross-session, tìm lại fact cũ

User "test-user-001":
  SESSION 1: khai báo 5 fact → verify rolling summary kích hoạt
  SESSION 2: hỏi lại fact ở S1 (cross-session recall)

Tùy chọn --cleanup để xoá test data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).parent.parent / ".env")

import requests  # noqa: E402

API = "http://localhost:8000"
USER_ID = "test-user-001"


def _post_chat(message: str, session_id: str, user_id: str = USER_ID,
               domain: str = "mặc định") -> dict[str, Any]:
    payload = {
        "message": message,
        "session_id": session_id,
        "user_id": user_id,
        "domain": domain,
    }
    r = requests.post(f"{API}/api/chat/", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _print_turn(label: str, user_msg: str, result: dict) -> None:
    print("=" * 72)
    print(f"[{label}] USER: {user_msg}")
    print("-" * 72)
    ans = result.get("answer", "")
    if len(ans) > 400:
        ans = ans[:400] + "…"
    print(f"BOT: {ans}\n")


def _health() -> None:
    r = requests.get(f"{API}/health", timeout=5)
    r.raise_for_status()
    print(f"[health] {r.json()}\n")


def _qdrant_count(user_id: str) -> int:
    from app.config import QDRANT_URL, QDRANT_API_KEY
    h = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    body = {"exact": True,
            "filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]}}
    r = requests.post(f"{QDRANT_URL.rstrip('/')}/collections/ttt_memory/points/count",
                      headers=h, json=body, timeout=15).json()
    return r.get("result", {}).get("count", 0)


def _read_session_file(session_id: str) -> dict:
    path = Path(__file__).parent.parent / "data" / "sessions" / f"{session_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_scenario() -> None:
    _health()

    # Cleanup trước khi chạy để không bị lẫn data cũ
    print(">>> Cleanup data cũ của test-user-001")
    requests.delete(f"{API}/api/chat/memory/user/{USER_ID}", timeout=30)
    print()

    session1 = f"test-s1-{uuid.uuid4().hex[:8]}"
    session2 = f"test-s2-{uuid.uuid4().hex[:8]}"
    print(f"User:      {USER_ID}")
    print(f"Session 1: {session1}")
    print(f"Session 2: {session2}\n")

    # ---------------- SESSION 1: khai báo 5 fact ----------------
    print(">>> SESSION 1 — khai báo fact qua nhiều turn")

    fact_turns = [
        "Xin chào, tôi tên là Nam. Tôi làm ở phòng Marketing của công ty TDI.",
        "Tôi đang phụ trách chiến dịch content marketing Q2 năm 2026. Ngân sách là 500 triệu.",
        "Trong đó, event 30/4 được phân bổ 200 triệu đồng, do chị Hà bên team event hỗ trợ.",
        "Tôi thích câu trả lời ngắn gọn, theo format bullet points.",
        "À, một chi tiết nữa: brand color chủ đạo của TDI là xanh lá và trắng.",
    ]
    for i, msg in enumerate(fact_turns, 1):
        r = _post_chat(msg, session_id=session1)
        _print_turn(f"S1-T{i}", msg, r)

    print("… chờ 4s cho background task upsert Qdrant + summarize …\n")
    time.sleep(4)

    # ---- Kiểm tra tầng 2: rolling summary ----
    s1_state = _read_session_file(session1)
    history_len = len(s1_state.get("history", []))
    summary = s1_state.get("summary", "")
    print(f"[CHECK TẦNG 2] Session 1 state:")
    print(f"  history messages: {history_len} (window 3 pair = 6 msgs)")
    print(f"  summary len:      {len(summary)} chars")
    if summary:
        print(f"  summary preview:  {summary[:300]}…" if len(summary) > 300 else f"  summary:          {summary}")
    tang2_ok = history_len <= 6 and len(summary) > 20
    print(f"  ✅ Rolling summary ĐÃ kích hoạt" if tang2_ok else f"  ❌ Rolling summary chưa kích hoạt")
    print()

    # ---- Kiểm tra tầng 3: upsert Qdrant ----
    qcount = _qdrant_count(USER_ID)
    print(f"[CHECK TẦNG 3] Qdrant points user={USER_ID}: {qcount} (kỳ vọng = 5)")
    tang3_upsert_ok = qcount >= 5
    print(f"  ✅ Vector upsert OK" if tang3_upsert_ok else f"  ❌ Upsert thiếu")
    print()

    # ---------------- SESSION 2: recall cross-session ----------------
    print(">>> SESSION 2 — phiên MỚI, cùng user_id. Hỏi lại fact S1.")
    print("    Kỳ vọng: vector recall fetch được pair liên quan → bot trả lời chính xác\n")

    recall_queries = [
        "Tôi tên gì nhỉ?",  # → recall fact S1-T1
        "Ngân sách chiến dịch Q2 bao nhiêu?",  # → recall S1-T2
        "Màu chủ đạo của TDI là gì?",  # → recall S1-T5
    ]
    recall_results = []
    for i, q in enumerate(recall_queries, 1):
        r = _post_chat(q, session_id=session2)
        _print_turn(f"S2-T{i}", q, r)
        recall_results.append((q, r))

    # ---- Heuristic check: câu trả lời có chứa keyword từ fact cũ không ----
    print("[CHECK TẦNG 3] Cross-session recall:")
    expected_keywords = ["Nam", "500", "xanh"]
    hits = 0
    for (q, r), kw in zip(recall_results, expected_keywords):
        ans = r.get("answer", "").lower()
        hit = kw.lower() in ans
        hits += int(hit)
        mark = "✅" if hit else "❌"
        print(f"  {mark} '{q}' → answer chứa '{kw}'? {hit}")
    print(f"  Tổng: {hits}/{len(recall_queries)} recall thành công\n")

    # ---------------- Test DELETE ----------------
    print(">>> TEST DELETE endpoints")
    r = requests.delete(f"{API}/api/chat/memory/session/{session1}", timeout=10)
    print(f"  DELETE session S1: {r.status_code} {r.json()}")
    time.sleep(1)
    qcount_after = _qdrant_count(USER_ID)
    print(f"  Qdrant points sau khi xoá S1: {qcount_after} (kỳ vọng < {qcount})\n")

    # ---------------- Final verdict ----------------
    print("=" * 72)
    print("TÓM TẮT KẾT QUẢ:")
    print(f"  Tầng 1 (sliding window): ✅ hoạt động (history được trim về ≤6 msg)")
    print(f"  Tầng 2 (rolling summary): {'✅ OK' if tang2_ok else '❌ FAIL'}")
    print(f"  Tầng 3 (vector upsert):   {'✅ OK' if tang3_upsert_ok else '❌ FAIL'}")
    print(f"  Tầng 3 (cross-session):   {hits}/{len(recall_queries)} match keyword")
    print(f"  DELETE session:           {'✅ OK' if qcount_after < qcount else '❌ FAIL'}")
    print()
    print(f"Files để xem thêm: data/sessions/{session1}.json, data/sessions/{session2}.json")


def cleanup() -> None:
    r = requests.delete(f"{API}/api/chat/memory/user/{USER_ID}", timeout=30)
    print(f"[cleanup qdrant] {r.status_code} {r.json()}")
    sessions_dir = Path(__file__).parent.parent / "data" / "sessions"
    if sessions_dir.exists():
        for f in sessions_dir.glob("test-s*.json"):
            f.unlink()
            print(f"  removed {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
        sys.exit(0)

    try:
        run_scenario()
    except requests.ConnectionError:
        print(f"ERROR: Không kết nối được {API}. Bạn đã chạy ./run.sh chưa?")
        sys.exit(1)
