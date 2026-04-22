"""Central config loaded from .env / environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)


def _req(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


def _opt(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# --- LLM ---
ANTHROPIC_API_KEY: str = _req("ANTHROPIC_API_KEY")
CLAUDE_MODEL: str = _opt("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_HAIKU_MODEL: str = _opt("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5-20251001")

# --- Groq (Whisper API) ---
GROQ_API_KEY: str = _opt("GROQ_API_KEY", "")

# --- Embeddings (tất cả dùng voyage-3, 1024-dim) ---
VOYAGE_API_KEY: str = _req("VOYAGE_API_KEY")
VOYAGE_MODEL: str = _opt("VOYAGE_MODEL", "voyage-3")
VOYAGE_DIM: int = int(_opt("VOYAGE_DIM", "1024"))

# --- Qdrant (main cluster — ttt_*) ---
QDRANT_URL: str = _req("QDRANT_URL")
QDRANT_API_KEY: str = _req("QDRANT_API_KEY")

# --- Qdrant (vmedia cluster — READ ONLY) ---
QDRANT_VMEDIA_URL: str = _opt("QDRANT_VMEDIA_URL", "https://dd2f49bd-a20a-49b6-abf3-a4805b544ff2.us-east4-0.gcp.cloud.qdrant.io:6333")
QDRANT_VMEDIA_API_KEY: str = _opt("QDRANT_VMEDIA_API_KEY", "")

# --- Collections ---
COLLECTION_DOCS: str = _opt("COLLECTION_DOCS", "ttt_documents")
COLLECTION_VIDEOS: str = _opt("COLLECTION_VIDEOS", "ttt_videos")
VMEDIA_COLLECTIONS: list[str] = _opt("VMEDIA_COLLECTIONS", "vmedia_content,vmedia_design,vmedia_digital,vmedia_documents,vmedia_fonts,vmedia_image,vmedia_media,vmedia_qa,vmedia_ttnb").split(",")


# --- Chunking ---
CHUNK_MAX_TOKENS: int = int(_opt("CHUNK_MAX_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS: int = int(_opt("CHUNK_OVERLAP_TOKENS", "50"))

# --- Retrieval ---
TOP_K: int = int(_opt("TOP_K", "7"))
RERANK_TOP_K: int = int(_opt("RERANK_TOP_K", "5"))
MIN_SIMILARITY: float = float(_opt("MIN_SIMILARITY", "0.3"))
ENABLE_RERANKING: bool = _opt("ENABLE_RERANKING", "true").lower() == "true"
RERANK_SKIP_THRESHOLD: float = float(_opt("RERANK_SKIP_THRESHOLD", "0.85"))
ENABLE_QUERY_REWRITE: bool = _opt("ENABLE_QUERY_REWRITE", "true").lower() == "true"

# --- API server ---
API_HOST: str = _opt("API_HOST", "0.0.0.0")
API_PORT: int = int(_opt("API_PORT", "8000"))

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
UPLOADS_DIR = DATA_DIR / "uploads"
