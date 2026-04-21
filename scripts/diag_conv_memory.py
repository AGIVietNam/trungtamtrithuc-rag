"""Diagnostic cho conversation memory: đếm points theo user_id, sample payload,
và test retrieve() xem pair nào bị filter.

Chạy:
  source venv/bin/activate
  python scripts/diag_conv_memory.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).parent.parent / ".env")

import requests  # noqa: E402

from app.config import (  # noqa: E402
    QDRANT_URL, QDRANT_API_KEY, CONV_COLLECTION,
    VOYAGE_API_KEY, VOYAGE_MODEL,
    CONV_RECALL_TOP_K, CONV_RECALL_MIN_SCORE,
)
from app.core.voyage_embed import VoyageEmbedder  # noqa: E402
from app.core.conv_memory import ConversationMemory  # noqa: E402


def _headers() -> dict:
    return {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}


def scroll_all() -> list[dict]:
    """Scroll TẤT CẢ points trong ttt_memory."""
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
        pts = res.get("points", [])
        all_pts.extend(pts)
        offset = res.get("next_page_offset")
        if not offset:
            break
    return all_pts


def group_by_user(points: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        payload = p.get("payload") or {}
        uid = payload.get("user_id", "<no_user_id>")
        grouped[uid].append(p)
    return grouped


def print_summary(grouped: dict[str, list[dict]]) -> None:
    print("=" * 72)
    print(f"TOTAL points trong {CONV_COLLECTION}: {sum(len(v) for v in grouped.values())}")
    print(f"Unique user_id: {len(grouped)}")
    print("=" * 72)

    for uid, pts in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        sess_ids = {(p.get("payload") or {}).get("session_id") for p in pts}
        print(f"\n[user_id={uid!r}]  count={len(pts)}  sessions={len(sess_ids)}")
        for sid in sorted(sess_ids, key=lambda s: str(s)):
            sess_pts = [p for p in pts
                        if (p.get("payload") or {}).get("session_id") == sid]
            print(f"    session={sid!r}  pairs={len(sess_pts)}")
            for p in sess_pts[:3]:
                pl = p.get("payload") or {}
                text = (pl.get("text") or "")[:100].replace("\n", " | ")
                print(f"      turn={pl.get('turn_idx')}  text={text!r}")
            if len(sess_pts) > 3:
                print(f"      ... +{len(sess_pts) - 3} pair khác")


def test_retrieve(conv_mem: ConversationMemory, user_id: str,
                  query: str, current_session_id: str | None = None) -> None:
    print("\n" + "=" * 72)
    print(f"TEST retrieve(user_id={user_id!r}, query={query!r}, "
          f"current_session_id={current_session_id!r})")
    print(f"top_k={CONV_RECALL_TOP_K}, min_score={CONV_RECALL_MIN_SCORE}")
    print("=" * 72)

    # Gọi raw search không threshold để thấy hết candidate
    vec = conv_mem._embed(query)
    body = {
        "vector": {"name": "", "vector": vec},
        "limit": 10,
        "with_payload": True,
        "filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]},
    }
    r = requests.post(
        f"{QDRANT_URL.rstrip('/')}/collections/{CONV_COLLECTION}/points/search",
        headers=_headers(), json=body, timeout=30,
    )
    if not r.ok:
        print(f"ERROR search: {r.status_code} {r.text[:200]}")
        return
    hits = r.json().get("result", [])

    print(f"\n>>> Raw candidates (no threshold, no session filter): {len(hits)}")
    for h in hits:
        pl = h.get("payload") or {}
        text = (pl.get("text") or "")[:80].replace("\n", " | ")
        excluded_reason = []
        if h["score"] < CONV_RECALL_MIN_SCORE:
            excluded_reason.append(f"score<{CONV_RECALL_MIN_SCORE}")
        if current_session_id and pl.get("session_id") == current_session_id:
            excluded_reason.append("same-session")
        mark = "✅" if not excluded_reason else f"❌ {','.join(excluded_reason)}"
        print(f"  score={h['score']:.3f} sess={pl.get('session_id')} turn={pl.get('turn_idx')}")
        print(f"     text={text!r}  {mark}")

    # Gọi retrieve() thực tế
    out = conv_mem.retrieve(user_id, query, current_session_id=current_session_id)
    print(f"\n>>> conv_memory.retrieve() trả về: {len(out)} pairs")
    for p in out:
        text = (p.get("text") or "")[:80].replace("\n", " | ")
        print(f"  score={p['score']:.3f} sess={p['session_id']} text={text!r}")


def main() -> None:
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {CONV_COLLECTION}\n")

    points = scroll_all()
    grouped = group_by_user(points)
    print_summary(grouped)

    if not points:
        print("\n(Collection rỗng — chưa có data để test)")
        return

    # Khởi tạo ConversationMemory để test retrieve
    embedder = VoyageEmbedder(api_key=VOYAGE_API_KEY, model=VOYAGE_MODEL)
    conv_mem = ConversationMemory(embedder=embedder)

    # Thử retrieve với 2 query: 1 trực tiếp hỏi fact, 1 gián tiếp (kiểu suy luận)
    # Chọn user_id có nhiều data nhất
    top_uid = max(grouped.keys(), key=lambda k: len(grouped[k]))
    print(f"\n\n### Test retrieve cho user_id có nhiều data nhất: {top_uid!r}")

    test_queries = [
        "Ngân sách content của team tôi là bao nhiêu?",
        "tính toán ngân sách hợp lý đi",
        "dựa vào plan quay và ngân sách hãy ước lượng ngân sách cho từng video",
        "Tôi tên gì?",
    ]
    for q in test_queries:
        test_retrieve(conv_mem, top_uid, q, current_session_id=None)


if __name__ == "__main__":
    main()
