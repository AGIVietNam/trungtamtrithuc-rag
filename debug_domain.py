"""Debug script: kiểm tra domain values thực tế trong Qdrant."""
import unicodedata
import requests
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTIONS = [
    os.getenv("COLLECTION_DOCS", "ttt_documents"),
    os.getenv("COLLECTION_VIDEOS", "ttt_videos"),
]

headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}


def scroll_all_domains(collection: str) -> dict:
    """Scroll qua collection, thu thập tất cả domain values."""
    domains: dict[str, int] = {}
    offset = None
    total = 0

    while True:
        body = {"limit": 100, "with_payload": ["domain"]}
        if offset is not None:
            body["offset"] = offset

        resp = requests.post(
            f"{QDRANT_URL}/collections/{collection}/points/scroll",
            headers=headers, json=body, timeout=30,
        )
        if not resp.ok:
            print(f"  ERROR scroll {collection}: {resp.status_code} {resp.text[:200]}")
            break

        data = resp.json().get("result", {})
        points = data.get("points", [])
        next_offset = data.get("next_page_offset")

        for p in points:
            total += 1
            payload = p.get("payload", {}) or {}
            domain = payload.get("domain")
            if domain is None:
                domains["<NONE>"] = domains.get("<NONE>", 0) + 1
            elif domain == "":
                domains["<EMPTY>"] = domains.get("<EMPTY>", 0) + 1
            else:
                domains[domain] = domains.get(domain, 0) + 1

        if not next_offset or not points:
            break
        offset = next_offset

    return {"total_points": total, "domains": domains}


def analyze_unicode(value: str) -> str:
    """Phân tích Unicode encoding của một chuỗi."""
    nfc = unicodedata.normalize("NFC", value)
    nfd = unicodedata.normalize("NFD", value)
    is_nfc = value == nfc
    is_nfd = value == nfd
    hex_bytes = value.encode("utf-8").hex()
    return f"NFC={is_nfc} NFD={is_nfd} hex={hex_bytes} len={len(value)}"


def test_filter(collection: str, domain_value: str) -> int:
    """Thử filter trực tiếp với 1 domain value, trả về số hits."""
    body = {
        "limit": 5,
        "with_payload": ["domain"],
        "filter": {
            "must": [
                {"key": "domain", "match": {"value": domain_value}}
            ]
        }
    }
    resp = requests.post(
        f"{QDRANT_URL}/collections/{collection}/points/scroll",
        headers=headers, json=body, timeout=30,
    )
    if not resp.ok:
        print(f"  FILTER ERROR: {resp.status_code} {resp.text[:200]}")
        return -1
    points = resp.json().get("result", {}).get("points", [])
    return len(points)


def test_filter_any(collection: str, domain_values: list[str]) -> int:
    """Thử filter 'any' với danh sách domain values."""
    body = {
        "limit": 5,
        "with_payload": ["domain"],
        "filter": {
            "must": [
                {"key": "domain", "match": {"any": domain_values}}
            ]
        }
    }
    resp = requests.post(
        f"{QDRANT_URL}/collections/{collection}/points/scroll",
        headers=headers, json=body, timeout=30,
    )
    if not resp.ok:
        print(f"  FILTER ANY ERROR: {resp.status_code} {resp.text[:200]}")
        return -1
    points = resp.json().get("result", {}).get("points", [])
    return len(points)


if __name__ == "__main__":
    print("=" * 70)
    print("DEBUG QDRANT DOMAIN FILTER")
    print("=" * 70)

    for col in COLLECTIONS:
        print(f"\n{'─' * 50}")
        print(f"Collection: {col}")
        print(f"{'─' * 50}")

        result = scroll_all_domains(col)
        print(f"  Tổng points: {result['total_points']}")
        print(f"  Số domain khác nhau: {len(result['domains'])}")

        if not result["domains"]:
            print("  (Không có dữ liệu)")
            continue

        print(f"\n  Domain values thực tế trong Qdrant:")
        for domain, count in sorted(result["domains"].items(), key=lambda x: -x[1]):
            unicode_info = analyze_unicode(domain) if domain not in ("<NONE>", "<EMPTY>") else ""
            print(f"    '{domain}' → {count} points  {unicode_info}")

        # Test filter cho từng domain
        print(f"\n  Test filter trực tiếp (match exact):")
        for domain in result["domains"]:
            if domain in ("<NONE>", "<EMPTY>"):
                continue
            hits = test_filter(col, domain)
            nfc_domain = unicodedata.normalize("NFC", domain)
            hits_nfc = test_filter(col, nfc_domain) if nfc_domain != domain else hits
            print(f"    filter('{domain}') → {hits} hits")
            if nfc_domain != domain:
                print(f"    filter(NFC '{nfc_domain}') → {hits_nfc} hits  ← NFC khác gốc!")

        # Test filter "Sản xuất" cụ thể
        test_domains = ["Sản xuất", "sản xuất", "Pháp lý", "pháp lý", "Công nghệ thông tin"]
        print(f"\n  Test filter các domain phổ biến:")
        for d in test_domains:
            hits = test_filter(col, d)
            nfc_d = unicodedata.normalize("NFC", d)
            hits_nfc = test_filter(col, nfc_d) if nfc_d != d else hits
            status = "✓" if hits > 0 else "✗"
            print(f"    {status} filter('{d}') → {hits} hits")
            if nfc_d != d:
                nfc_status = "✓" if hits_nfc > 0 else "✗"
                print(f"    {nfc_status} filter(NFC '{nfc_d}') → {hits_nfc} hits")

        # Test match any
        print(f"\n  Test filter 'any' (cách code hiện tại dùng):")
        any_list = ["Sản xuất", "sản xuất", "Mặc định", "mặc định", "General", "general"]
        hits_any = test_filter_any(col, any_list)
        print(f"    match.any={any_list} → {hits_any} hits")

    # Check payload index
    print(f"\n{'─' * 50}")
    print("Check payload indexes:")
    print(f"{'─' * 50}")
    for col in COLLECTIONS:
        resp = requests.get(
            f"{QDRANT_URL}/collections/{col}",
            headers=headers, timeout=30,
        )
        if resp.ok:
            info = resp.json().get("result", {})
            indexes = info.get("payload_schema", {})
            domain_idx = indexes.get("domain")
            print(f"  {col}:")
            print(f"    payload_schema.domain = {domain_idx}")
            if not domain_idx:
                print(f"    ⚠️  THIẾU INDEX 'domain'! Filter sẽ KHÔNG hoạt động!")
        else:
            print(f"  {col}: ERROR {resp.status_code}")

    print("\n" + "=" * 70)
    print("DONE")
