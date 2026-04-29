"""S3 image store — upload ảnh extract từ tài liệu lên bucket team.

Pattern:
  - BE upload file gốc (PDF/DOCX) lên S3 → tạo presigned URL → đẩy AI service.
  - AI service trong process ingest extract ảnh → upload vào cùng bucket dưới
    prefix `images/{doc_id}/{image_id}.png` → trả full public URL về payload
    Qdrant để FE render.

Bucket được team setup public-read → URL public trực tiếp dùng được, không cần
endpoint API ở AI side, không phải presign mỗi response.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.core import config

logger = logging.getLogger(__name__)


_IMAGE_KEY_PREFIX = "images"  # bucket-level prefix cho ảnh extract
_PUBLIC_CONTENT_TYPE = "image/png"


def _build_public_url(key: str) -> str:
    """Build public URL theo path-style (S3-compatible Viettel IDC).

    Format: {S3_PUBLIC_ENDPOINT}/{bucket}/{key}
    Lý do path-style thay vì virtual-hosted: S3-compat providers (Viettel IDC,
    Wasabi, MinIO) thường dùng path-style; AWS native S3 dùng virtual-hosted.
    """
    endpoint = (config.S3_PUBLIC_ENDPOINT or config.S3_ENDPOINT).rstrip("/")
    bucket = config.S3_BUCKET_NAME
    return f"{endpoint}/{bucket}/{key}"


def _image_key(doc_id: str, image_id: str) -> str:
    """Key trên S3: images/{doc_id}/{image_id}.png. Independent prefix với
    PDF gốc — không assume BE lưu PDF ở prefix nào, không nest theo doc của BE.
    """
    return f"{_IMAGE_KEY_PREFIX}/{doc_id}/{image_id}.png"


_s3_client = None


def _get_client():
    """Lazy boto3 client — config từ env, dùng path-style addressing."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        logger.warning("boto3 chưa cài — không upload S3 được. Pipeline sẽ skip ảnh.")
        return None

    if not config.S3_ENDPOINT or not config.S3_BUCKET_NAME:
        logger.warning("S3 chưa config (S3_ENDPOINT/S3_BUCKET_NAME rỗng) — skip upload")
        return None

    _s3_client = boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        aws_access_key_id=config.S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
        region_name=config.S3_REGION,
        # Path-style cho S3-compat (Viettel IDC). Virtual-hosted style không
        # work với endpoint custom domain.
        config=Config(s3={"addressing_style": "path"}),
    )
    logger.info("S3 client initialized: endpoint=%s bucket=%s",
                config.S3_ENDPOINT, config.S3_BUCKET_NAME)
    return _s3_client


def upload_image(image_bytes: bytes, doc_id: str, image_id: str) -> Optional[str]:
    """Upload PNG bytes lên S3 và trả public URL. None nếu fail / không config.

    ContentType=image/png để browser render đúng. ACL public-read để FE load
    thẳng URL không cần presign (bucket đã setup public-read theo team).
    """
    client = _get_client()
    if client is None:
        return None

    key = _image_key(doc_id, image_id)
    try:
        client.put_object(
            Bucket=config.S3_BUCKET_NAME,
            Key=key,
            Body=image_bytes,
            ContentType=_PUBLIC_CONTENT_TYPE,
            ACL="public-read",
        )
    except Exception:
        logger.exception("S3 upload failed: key=%s size=%d", key, len(image_bytes))
        return None

    url = _build_public_url(key)
    logger.info("S3 uploaded: key=%s size=%d url=%s", key, len(image_bytes), url)
    return url


def delete_doc_images(doc_id: str) -> int:
    """Xoá toàn bộ object có prefix images/{doc_id}/. Trả số object đã xoá.

    Gọi đầu mỗi lần ingest cùng doc_id (giống pattern delete-then-upsert ở
    Qdrant) — tránh ảnh stale từ ingest cũ nếu user reup file đã sửa.

    Best-effort: list (max 1000) → batch delete. Doc thường có <100 ảnh nên
    1 vòng list là đủ; pagination chỉ cần khi doc quá lớn (chưa gặp).
    """
    client = _get_client()
    if client is None:
        return 0

    prefix = f"{_IMAGE_KEY_PREFIX}/{doc_id}/"
    try:
        resp = client.list_objects_v2(Bucket=config.S3_BUCKET_NAME, Prefix=prefix)
        contents = resp.get("Contents", [])
        if not contents:
            return 0
        client.delete_objects(
            Bucket=config.S3_BUCKET_NAME,
            Delete={"Objects": [{"Key": obj["Key"]} for obj in contents]},
        )
        logger.info("S3 deleted %d objects under %s", len(contents), prefix)
        return len(contents)
    except Exception:
        logger.exception("S3 delete prefix failed: %s", prefix)
        return 0


def is_configured() -> bool:
    """True nếu S3 đã setup đủ (config + boto3 cài). Pipeline check trước khi
    cố upload — nếu False, pipeline sẽ skip extract ảnh."""
    return _get_client() is not None
