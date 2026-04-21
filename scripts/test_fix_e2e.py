"""Reproduce lại kịch bản từng fail và verify fix hoạt động.

Bối cảnh: trong Qdrant đã có fact của test-user-1 (session fd1c89c5...):
  - "Team tôi gồm 6 người, ngân sách content tháng này là 80 triệu"
  - Plan video: 6 video về BKVN

Trước fix: hỏi "tính toán ngân sách cho 6 video" → bot nói "không tìm thấy".
Sau fix:   bot phải tính được ~13 triệu/video hoặc nêu đủ dữ kiện.

Chạy:
  source venv/bin/activate
  python scripts/test_fix_e2e.py
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).parent.parent / ".env")

import requests  # noqa: E402


API = "http://localhost:8088"
USER_ID = "test-user-1"


def chat(message: str, session_id: str) -> dict:
    r = requests.post(
        f"{API}/api/chat/",
        json={
            "message": message,
            "session_id": session_id,
            "user_id": USER_ID,
            "domain": "mặc định",
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def check(label: str, answer: str, must_contain: list[str],
          must_not_contain: list[str] = None) -> bool:
    ans_lower = answer.lower()
    missing = [kw for kw in must_contain if kw.lower() not in ans_lower]
    leaked = [kw for kw in (must_not_contain or []) if kw.lower() in ans_lower]
    ok = not missing and not leaked
    mark = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {mark} {label}")
    if missing:
        print(f"     thiếu keyword: {missing}")
    if leaked:
        print(f"     leak keyword cấm: {leaked}")
    return ok


def main() -> None:
    # Session MỚI — test cross-session recall
    session_id = f"e2e-{uuid.uuid4().hex[:8]}"
    print(f"User: {USER_ID}")
    print(f"Session mới: {session_id}\n")

    tests = [
        {
            "label": "Q1: Recall đơn giản 'ngân sách bao nhiêu'",
            "msg": "Ngân sách content của team tôi là bao nhiêu?",
            "must_contain": ["80", "triệu"],
            "must_not_contain": ["không tìm thấy", "không có thông tin"],
        },
        {
            "label": "Q2: Suy luận tính toán - ước lượng ngân sách/video",
            "msg": "Dựa vào plan video tháng và ngân sách team tôi, hãy ước lượng ngân sách cho từng video.",
            "must_contain": ["triệu"],  # phải nêu số tiền/video
            "must_not_contain": ["không tìm thấy thông tin về ngân sách"],
        },
        {
            "label": "Q3: Recall tên nhóm nhân sự (RAG doc)",
            "msg": "Nhóm nhân sự nào của TDI làm tại công trường?",
            "must_contain": ["ban chỉ huy"],
            "must_not_contain": [],
        },
    ]

    passed = 0
    for t in tests:
        print(f"\n>>> {t['label']}")
        print(f"    USER: {t['msg']}")
        r = chat(t["msg"], session_id)
        ans = r.get("answer", "")
        print(f"    BOT : {ans[:400]}{'…' if len(ans) > 400 else ''}")
        if check(t["label"], ans,
                 t["must_contain"], t.get("must_not_contain")):
            passed += 1
        time.sleep(1)  # tránh rate limit

    print("\n" + "=" * 72)
    print(f"KẾT QUẢ: {passed}/{len(tests)} pass")


if __name__ == "__main__":
    try:
        main()
    except requests.ConnectionError:
        print(f"ERROR: server không chạy ở {API}")
        sys.exit(1)
