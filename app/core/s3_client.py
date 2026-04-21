"""S3-compatible storage client (Viettel IDC / AWS S3 / MinIO...).

Dùng boto3 với `endpoint_url` để hỗ trợ mọi provider S3-compatible.
Upload file gốc → trả về public URL cố định (deterministic theo SHA-256).

Quy ước key: `<prefix>/<sha256>.<ext>`
  - deterministic: cùng content → cùng key → upsert không duplicate storage.
  - prefix theo loại: `docs/`, `videos/` để dễ thống kê/phân quyền.
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Final

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from app.core import config

logger = logging.getLogger(__name__)


PREFIX_DOCS: Final[str] = "docs"
PREFIX_VIDEOS: Final[str] = "videos"
UNSORTED_DOMAIN: Final[str] = "unsorted"

_CHUNK_SIZE_FOR_HASH = 64 * 1024  # 64KB đọc từng chunk khi hash file lớn


def slugify_domain(domain: str | None) -> str:
    """Chuyển tên domain tiếng Việt → subfolder S3 ASCII-safe.

    'công nghệ thông tin' → 'cong-nghe-thong-tin'
    'Pháp lý'             → 'phap-ly'
    '' / None             → UNSORTED_DOMAIN

    Tránh ký tự lạ trong URL (dù S3 hỗ trợ UTF-8, URL percent-encode rắc rối).
    """
    if not domain:
        return UNSORTED_DOMAIN
    # Thay đ/Đ bằng d TRƯỚC khi normalize (đ không decompose NFKD)
    src = domain.strip().lower().replace("đ", "d")
    # Bỏ dấu tiếng Việt (NFKD → strip combining marks)
    normalized = unicodedata.normalize("NFKD", src)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    # collapse mọi ký tự không phải a-z0-9 thành dấu gạch
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug or UNSORTED_DOMAIN


@lru_cache(maxsize=1)
def _get_client():
    """Singleton boto3 S3 client. Raise nếu credentials thiếu."""
    if not all([
        config.S3_ENDPOINT, config.S3_ACCESS_KEY_ID,
        config.S3_SECRET_ACCESS_KEY, config.S3_BUCKET_NAME,
    ]):
        raise RuntimeError(
            "S3 config incomplete. Cần set S3_ENDPOINT, S3_ACCESS_KEY_ID, "
            "S3_SECRET_ACCESS_KEY, S3_BUCKET_NAME trong .env"
        )

    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        aws_access_key_id=config.S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
        region_name=config.S3_REGION,
        # Path-style URL (https://host/bucket/key) tương thích nhiều provider hơn
        # virtual-hosted style (https://bucket.host/key). Viettel IDC dùng path-style.
        config=BotoConfig(s3={"addressing_style": "path"}),
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE_FOR_HASH), b""):
            h.update(chunk)
    return h.hexdigest()


def _public_url(key: str) -> str:
    base = (config.S3_PUBLIC_ENDPOINT or config.S3_ENDPOINT).rstrip("/")
    return f"{base}/{config.S3_BUCKET_NAME}/{key}"


def upload_file(
    local_path: str | Path,
    prefix: str,
    original_name: str | None = None,
    content_type: str | None = None,
) -> str:
    """Upload file lên S3. Trả public URL.

    Args:
      local_path: đường dẫn file local.
      prefix: "docs" / "videos" / ... → thư mục con trong bucket.
      original_name: tên file hiển thị (dùng để lấy ext + metadata). Fallback: local_path.
      content_type: MIME type, tự detect nếu None.

    Key format: `<prefix>/<sha256>.<ext>` → cùng content → cùng key (idempotent).

    Raises RuntimeError nếu S3 config thiếu hoặc upload fail.
    """
    path = Path(local_path)
    if not path.is_file():
        raise RuntimeError(f"File không tồn tại: {path}")

    name_for_ext = original_name or path.name
    ext = Path(name_for_ext).suffix.lower().lstrip(".")
    sha = _sha256_file(path)
    key = f"{prefix.strip('/')}/{sha}{('.' + ext) if ext else ''}"

    ct = content_type or mimetypes.guess_type(name_for_ext)[0] or "application/octet-stream"

    client = _get_client()
    try:
        # S3 metadata chỉ cho phép ASCII → bỏ qua tên gốc tiếng Việt ở đây.
        # Tên gốc đã được lưu trong Qdrant payload (extra_metadata.original_filename).
        extra = {
            "ContentType": ct,
            "ACL": "public-read",  # bucket policy không public → cần ACL per-object
        }
        client.upload_file(
            Filename=str(path),
            Bucket=config.S3_BUCKET_NAME,
            Key=key,
            ExtraArgs=extra,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.exception("S3 upload failed: key=%s", key)
        raise RuntimeError(f"S3 upload failed: {exc}") from exc

    url = _public_url(key)
    logger.info("S3 upload OK: %s → %s (%d bytes, ct=%s)", name_for_ext, key, path.stat().st_size, ct)
    return url


def is_configured() -> bool:
    """True nếu đủ 4 env var bắt buộc."""
    return all([
        config.S3_ENDPOINT, config.S3_ACCESS_KEY_ID,
        config.S3_SECRET_ACCESS_KEY, config.S3_BUCKET_NAME,
    ])
