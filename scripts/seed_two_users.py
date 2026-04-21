"""Seed 2 user giả với vài turn hội thoại để test lưu lịch sử Qdrant.

Chạy:
  1. Start server ở terminal khác:  ./run.sh
  2. python scripts/seed_two_users.py

Flow:
  - Mỗi user có 1 session, gửi vài message qua /api/chat/
  - Background task của server sẽ upsert pair vào ttt_memory
  - Script chờ 4s rồi đếm points theo user_id để verify

Tùy chọn:
  --cleanup  xoá toàn bộ pair của 2 user khỏi Qdrant
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).parent.parent / ".env")

import requests  # noqa: E402

from app.config import QDRANT_URL, QDRANT_API_KEY  # noqa: E402


API = "http://localhost:8000"
COLLECTION = "ttt_memory"

USERS: list[dict] = [
    {
        "user_id": "test-user-alpha",
        "domain": "marketing",
        "turns": [
            "Xin chào, tôi tên là Linh, làm ở phòng Marketing công ty TDI.",
            "Tôi đang phụ trách chiến dịch truyền thông nội bộ Q2 2026, ngân sách 300 triệu.",
            "Màu brand của TDI là xanh dương và trắng, slogan 'Tri thức dẫn lối'.",
        ],
    },
    {
        "user_id": "test-user-beta",
        "domain": "kỹ thuật",
        "turns": [
            "Chào bạn, tôi là Minh, kỹ sư backend team Data Platform.",
            "Dự án hiện tại của tôi là pipeline ingest video từ YouTube vào Qdrant.",
            "Tôi thích câu trả lời có code example và format markdown.",
        ],
    },
]


def _health() -> None:
    r = requests.get(f"{API}/health", timeout=5)
    r.raise_for_status()
    print(f"[health] {r.json()}\n")


def _post_chat(message: str, session_id: str, user_id: str, domain: str) -> dict:
    payload = {
        "message": message,
        "session_id": session_id,
        "user_id": user_id,
        "domain": domain,
    }
    r = requests.post(f"{API}/api/chat/", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _qdrant_count(user_id: str) -> int:
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    body = {
        "exact": True,
        "filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]},
    }
    r = requests.post(
        f"{QDRANT_URL.rstrip('/')}/collections/{COLLECTION}/points/count",
        headers=headers, json=body, timeout=15,
    )
    r.raise_for_status()
    return r.json().get("result", {}).get("count", 0)


def _delete_user(user_id: str) -> None:
    r = requests.delete(f"{API}/api/chat/memory/user/{user_id}", timeout=30)
    print(f"  DELETE {user_id}: {r.status_code} {r.json()}")


def run_seed() -> None:
    _health()

    print(">>> Cleanup data cũ của 2 test user\n")
    for u in USERS:
        _delete_user(u["user_id"])
    print()

    summary: list[tuple[str, int, int]] = []

    for u in USERS:
        user_id = u["user_id"]
        domain = u["domain"]
        session_id = f"seed-{user_id}-{uuid.uuid4().hex[:6]}"

        print("=" * 72)
        print(f"USER: {user_id}   SESSION: {session_id}   DOMAIN: {domain}")
        print("=" * 72)

        for i, msg in enumerate(u["turns"], 1):
            r = _post_chat(msg, session_id=session_id, user_id=user_id, domain=domain)
            ans = r.get("answer", "")
            if len(ans) > 300:
                ans = ans[:300] + "…"
            print(f"[T{i}] USER: {msg}")
            print(f"      BOT : {ans}\n")

        summary.append((user_id, session_id, len(u["turns"])))

    print("… chờ 5s cho background upsert Qdrant …\n")
    time.sleep(5)

    print(">>> VERIFY: đếm points theo user_id trong Qdrant")
    for user_id, session_id, expected in summary:
        count = _qdrant_count(user_id)
        mark = "✅" if count >= expected else "❌"
        print(f"  {mark} {user_id}: {count} points (kỳ vọng ≥ {expected})")

    print()
    print("Hoàn tất. Có thể:")
    print(f"  - Xem payload:  python scripts/check_memory_collection.py")
    print(f"  - Chat tiếp với các user_id trên ở phiên mới để test cross-session recall")
    print(f"  - Dọn data:     python scripts/seed_two_users.py --cleanup")


def cleanup() -> None:
    print(">>> Xoá data 2 test user khỏi Qdrant\n")
    for u in USERS:
        _delete_user(u["user_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true", help="Xoá data 2 test user")
    args = parser.parse_args()

    try:
        if args.cleanup:
            cleanup()
        else:
            run_seed()
    except requests.ConnectionError:
        print(f"ERROR: Không kết nối được {API}. Bạn đã chạy ./run.sh chưa?")
        sys.exit(1)
