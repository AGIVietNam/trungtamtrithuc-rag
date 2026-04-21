"""Xoá các pair "không tìm thấy / không biết" khỏi ttt_memory.

Các pair này gây feedback loop: bot recall câu trả lời cũ của chính nó
và lặp lại "không tìm thấy" thay vì dùng fact thật từ user.

Chạy:
  source venv/bin/activate
  python scripts/clean_poisoned_pairs.py           # dry-run: liệt kê
  python scripts/clean_poisoned_pairs.py --apply   # thực sự xoá
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).parent.parent / ".env")

import requests  # noqa: E402

from app.config import QDRANT_URL, QDRANT_API_KEY, CONV_COLLECTION  # noqa: E402
from app.api.chat import _is_no_info_answer, _NO_INFO_MARKERS  # noqa: E402


def _headers() -> dict:
    return {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}


def scroll_all() -> list[dict]:
    url = f"{QDRANT_URL.rstrip('/')}/collections/{CONV_COLLECTION}/points/scroll"
    offset = None
    all_pts: list[dict] = []
    while True:
        body = {"limit": 200, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(url, headers=_headers(), json=body, timeout=30)
        r.raise_for_status()
        res = r.json().get("result", {})
        all_pts.extend(res.get("points", []))
        offset = res.get("next_page_offset")
        if not offset:
            break
    return all_pts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Thực sự xoá (mặc định chỉ dry-run)")
    args = parser.parse_args()

    print(f"Markers dùng để detect poisoned pair:")
    for m in _NO_INFO_MARKERS:
        print(f"  - {m!r}")
    print()

    points = scroll_all()
    print(f"Tổng points trong {CONV_COLLECTION}: {len(points)}\n")

    poisoned: list[tuple[str, dict]] = []
    for p in points:
        payload = p.get("payload") or {}
        text = payload.get("text", "")
        # Chỉ check phần sau "BOT:"
        parts = text.split("BOT:", 1)
        bot_part = parts[1] if len(parts) == 2 else text
        if _is_no_info_answer(bot_part):
            poisoned.append((str(p["id"]), payload))

    print(f"Poisoned pairs phát hiện: {len(poisoned)}")
    for pid, pl in poisoned:
        text = (pl.get("text") or "")[:120].replace("\n", " | ")
        print(f"  id={pid}")
        print(f"    user={pl.get('user_id')} session={pl.get('session_id')} turn={pl.get('turn_idx')}")
        print(f"    text={text!r}")

    if not poisoned:
        print("\nKhông có gì để dọn. ✅")
        return

    if not args.apply:
        print("\n(dry-run) Chạy lại với --apply để xoá thực sự.")
        return

    ids = [pid for pid, _ in poisoned]
    r = requests.post(
        f"{QDRANT_URL.rstrip('/')}/collections/{CONV_COLLECTION}/points/delete?wait=true",
        headers=_headers(),
        json={"points": ids},
        timeout=30,
    )
    r.raise_for_status()
    print(f"\n✅ Đã xoá {len(ids)} poisoned pairs khỏi Qdrant.")


if __name__ == "__main__":
    main()
