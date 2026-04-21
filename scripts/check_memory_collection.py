"""Check current state of ttt_memory collection in Qdrant.

Prints:
  - collection exists?
  - vector dim + distance
  - payload indexes
  - point count
  - sample payload (1 point)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from app.config import QDRANT_URL, QDRANT_API_KEY, VOYAGE_DIM


COLLECTION = "ttt_memory"


def _h() -> dict:
    return {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}


def main() -> None:
    base = QDRANT_URL.rstrip("/")

    print(f"Target: {base}/collections/{COLLECTION}\n")

    r = requests.get(f"{base}/collections/{COLLECTION}", headers=_h(), timeout=30)
    if r.status_code == 404:
        print(f"[STATUS] Collection '{COLLECTION}' KHÔNG TỒN TẠI.")
        print("        → Sẽ cần tạo mới khi bắt đầu Phase 2.")
        return
    r.raise_for_status()
    info = r.json().get("result", {})

    vectors = info.get("config", {}).get("params", {}).get("vectors", {})
    print(f"[STATUS] Collection TỒN TẠI")
    print(f"[VECTORS]")
    if isinstance(vectors, dict):
        for name, cfg in vectors.items():
            disp_name = name if name else "(default/unnamed)"
            print(f"  name={disp_name!r} size={cfg.get('size')} "
                  f"distance={cfg.get('distance')}")
    else:
        print(f"  {vectors}")

    print(f"\n[EXPECTED] size={VOYAGE_DIM} distance=Cosine")

    payload_schema = info.get("payload_schema", {})
    print(f"\n[PAYLOAD INDEXES] ({len(payload_schema)})")
    for k, v in payload_schema.items():
        dtype = v.get("data_type") if isinstance(v, dict) else v
        print(f"  {k}: {dtype}")

    # Count
    r = requests.post(f"{base}/collections/{COLLECTION}/points/count",
                      headers=_h(), json={"exact": True}, timeout=30)
    count = r.json().get("result", {}).get("count", 0) if r.ok else "?"
    print(f"\n[POINT COUNT] {count}")

    # Sample 1 point
    r = requests.post(f"{base}/collections/{COLLECTION}/points/scroll",
                      headers=_h(),
                      json={"limit": 3, "with_payload": True, "with_vector": False},
                      timeout=30)
    if r.ok:
        points = r.json().get("result", {}).get("points", [])
        print(f"\n[SAMPLE PAYLOADS] (up to 3)")
        for i, p in enumerate(points, 1):
            payload = p.get("payload", {})
            print(f"  --- point {i} (id={p.get('id')}) ---")
            print(f"  {json.dumps(payload, ensure_ascii=False, indent=2)[:500]}")

    # Summary of fields seen in sample (to detect legacy entity schema)
    legacy_fields = {"category", "confidence", "status", "supersedes", "superseded_by",
                     "memory_id", "access_count", "last_accessed"}
    new_fields = {"conv_id", "turn_idx"}
    if r.ok and points:
        seen = set()
        for p in points:
            seen.update(p.get("payload", {}).keys())
        is_legacy = bool(seen & legacy_fields)
        is_new = bool(seen & new_fields)
        print(f"\n[SCHEMA DETECT]")
        print(f"  Legacy entity fields present: {sorted(seen & legacy_fields) or 'none'}")
        print(f"  New conversation fields present: {sorted(seen & new_fields) or 'none'}")
        if is_legacy and not is_new:
            print("  → Đang chứa DATA CŨ (entity memory). Cần decision: xoá hay coexist.")
        elif is_new and not is_legacy:
            print("  → Đã có schema conversation mới.")
        elif is_legacy and is_new:
            print("  → Mix cả 2 — cần dọn.")


if __name__ == "__main__":
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("ERROR: QDRANT_URL / QDRANT_API_KEY chưa cấu hình trong .env")
        sys.exit(1)
    main()
