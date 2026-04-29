from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# Voyage
VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL: str = os.getenv("VOYAGE_MODEL", "voyage-3")
VOYAGE_DIM: int = int(os.getenv("VOYAGE_DIM", "1024"))

# Qdrant (main cluster — ttt_*)
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
QDRANT_VECTOR_NAME: str = os.getenv("QDRANT_VECTOR_NAME", "")

# Qdrant (vmedia cluster — READ ONLY)
QDRANT_VMEDIA_URL: str = os.getenv("QDRANT_VMEDIA_URL", "")
QDRANT_VMEDIA_API_KEY: str = os.getenv("QDRANT_VMEDIA_API_KEY", "")

VMEDIA_COLLECTIONS: list[str] = os.getenv(
    "VMEDIA_COLLECTIONS",
    "vmedia_content,vmedia_design,vmedia_digital,vmedia_documents,vmedia_fonts,vmedia_image,vmedia_media,vmedia_qa,vmedia_ttnb",
).split(",")

# Claude
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
CLAUDE_HAIKU_MODEL: str = os.getenv("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5-20251001")

# Chunking
CHUNK_MAX_TOKENS: int = int(os.getenv("CHUNK_MAX_TOKENS", "700"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))

# Retrieval
TOP_K: int = int(os.getenv("TOP_K", "7"))
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))

# Hybrid retrieval (Anthropic Contextual Retrieval recipe).
# - HYBRID_RETRIEVAL: bật dual vector dense (Voyage) + sparse (BM25 + underthesea)
#   trong Qdrant, fuse RRF server-side qua /points/query API. Mặc định ON;
#   tắt (=0) để rollback về dense-only nếu hybrid lỗi.
# - CONTEXTUAL_CHUNKING: bật Haiku-generated context prefix cho mỗi chunk
#   trước khi embed/sparse-encode. Anthropic recipe: -49% fail rate kèm BM25.
# - BM25_HASH_BUCKETS: số buckets cho stable hash token → vocab id. 2^24 = 16M
#   đủ cho corpus < 1M unique tokens với rate collision < 1%.
HYBRID_RETRIEVAL: bool = os.getenv("HYBRID_RETRIEVAL", "1").strip().lower() not in ("0", "false", "no", "off")
CONTEXTUAL_CHUNKING: bool = os.getenv("CONTEXTUAL_CHUNKING", "1").strip().lower() not in ("0", "false", "no", "off")
BM25_HASH_BUCKETS: int = int(os.getenv("BM25_HASH_BUCKETS", str(1 << 24)))
# RRF prefetch limit per branch — mỗi nhánh dense/sparse pull tối đa N candidate
# rồi Qdrant fuse. 30 đủ rộng để chunk đúng lọt qua nhưng không quá nhiều noise.
HYBRID_PREFETCH_LIMIT: int = int(os.getenv("HYBRID_PREFETCH_LIMIT", "30"))

# API
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# GPU Requirement
REQUIRE_GPU: bool = os.getenv("REQUIRE_GPU", "0").strip().lower() not in ("0", "false", "no")

# YouTube transcript proxy (để vượt qua IP block).
# 3 cách cấu hình, ưu tiên theo thứ tự:
# 1) YOUTUBE_PROXY_LIST: chuỗi các proxy (ngăn cách dấu phẩy hoặc xuống dòng),
#    mỗi dòng "http://user:pass@host:port" — sẽ xoay vòng mỗi lần retry
# 2) WEBSHARE_PROXY_USERNAME + WEBSHARE_PROXY_PASSWORD: rotating endpoint của Webshare
# 3) YOUTUBE_PROXY_HTTP / YOUTUBE_PROXY_HTTPS: 1 proxy cố định
WEBSHARE_PROXY_USERNAME: str = os.getenv("WEBSHARE_PROXY_USERNAME", "")
WEBSHARE_PROXY_PASSWORD: str = os.getenv("WEBSHARE_PROXY_PASSWORD", "")
YOUTUBE_PROXY_HTTP: str = os.getenv("YOUTUBE_PROXY_HTTP", "")
YOUTUBE_PROXY_HTTPS: str = os.getenv("YOUTUBE_PROXY_HTTPS", "")
YOUTUBE_PROXY_LIST: str = os.getenv("YOUTUBE_PROXY_LIST", "")
YOUTUBE_TRANSCRIPT_MAX_RETRIES: int = int(os.getenv("YOUTUBE_TRANSCRIPT_MAX_RETRIES", "10"))
YOUTUBE_TRANSCRIPT_RETRY_DELAY: float = float(os.getenv("YOUTUBE_TRANSCRIPT_RETRY_DELAY", "1.5"))

# Conversation Memory (Hybrid: sliding window + rolling summary + vector recall)
CONV_COLLECTION: str = os.getenv("CONV_COLLECTION", "ttt_memory")
CONV_WINDOW_TURNS: int = int(os.getenv("CONV_WINDOW_TURNS", "3"))
CONV_SUMMARY_TRIGGER_EXTRA: int = int(os.getenv("CONV_SUMMARY_TRIGGER_EXTRA", "2"))
CONV_SUMMARY_MAX_TOKENS: int = int(os.getenv("CONV_SUMMARY_MAX_TOKENS", "400"))
CONV_RECALL_TOP_K: int = int(os.getenv("CONV_RECALL_TOP_K", "5"))
CONV_RECALL_MIN_SCORE: float = float(os.getenv("CONV_RECALL_MIN_SCORE", "0.3"))
CONV_REWRITE_MIN_LEN: int = int(os.getenv("CONV_REWRITE_MIN_LEN", "40"))

# Conv memory anti-bloat guards
# - CONV_MIN_USER_CHARS: user_msg ngắn hơn → skip upsert (câu xã giao)
# - CONV_MIN_BOT_CHARS: bot_msg ngắn hơn + không có "Nguồn:" → skip (chitchat)
# - CONV_DEDUP_THRESHOLD: cosine score với pair cũ cùng user, vượt ngưỡng → skip upsert
#   (0.90 EMem synonym, 0.95 Mem0 entity merge; 0.92 = trung dung Voyage-3)
# - CONV_HASH_CACHE_SIZE: số hash gần nhất giữ trong RAM để chặn exact dup
CONV_MIN_USER_CHARS: int = int(os.getenv("CONV_MIN_USER_CHARS", "20"))
CONV_MIN_BOT_CHARS: int = int(os.getenv("CONV_MIN_BOT_CHARS", "40"))
CONV_DEDUP_THRESHOLD: float = float(os.getenv("CONV_DEDUP_THRESHOLD", "0.92"))
CONV_HASH_CACHE_SIZE: int = int(os.getenv("CONV_HASH_CACHE_SIZE", "2000"))

# Data dirs
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
LOG_DIR = BASE_DIR / "data" / "logs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Async ingest / batch upload ---
# Cap kích thước 1 file (MB) — vượt sẽ reject ngay khi streaming write
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "5120"))         # 5GB default
# Số job xử lý song song trong worker pool
INGEST_WORKER_CONCURRENCY: int = int(os.getenv("INGEST_WORKER_CONCURRENCY", "2"))
# Cap số file/batch (multipart array hoặc from-urls)
INGEST_MAX_BATCH_SIZE: int = int(os.getenv("INGEST_MAX_BATCH_SIZE", "50"))
# Job giữ trong RAM bao lâu sau khi done/failed (giây)
INGEST_JOB_TTL_SEC: int = int(os.getenv("INGEST_JOB_TTL_SEC", "3600"))
# Kích thước chunk khi stream-to-disk (bytes)
UPLOAD_STREAM_CHUNK: int = int(os.getenv("UPLOAD_STREAM_CHUNK", str(1 << 20)))   # 1MB

# --- Preview optimization ---
# Số trang/slide đầu tiên parse để gen metadata preview
PREVIEW_MAX_PAGES: int = int(os.getenv("PREVIEW_MAX_PAGES", "5"))
# Số giây đầu video transcribe để gen metadata preview
PREVIEW_VIDEO_CLIP_SEC: int = int(os.getenv("PREVIEW_VIDEO_CLIP_SEC", "180"))

# --- Backend document webhook ---
# AI POST {document_id, status} sang BE khi ingest job đạt terminal (done/failed).
# BE truyền `document_id` xuống cùng `from-url` payload; nếu trống thì bỏ qua call.
# Cấu hình BẮT BUỘC ở .env (BACKEND_DOCUMENT_WEBHOOK_URL + BACKEND_WEBHOOK_API_KEY).
BACKEND_DOCUMENT_WEBHOOK_URL: str = os.getenv("BACKEND_DOCUMENT_WEBHOOK_URL", "")
BACKEND_WEBHOOK_API_KEY: str = os.getenv("BACKEND_WEBHOOK_API_KEY", "")
