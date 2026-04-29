from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core import config
from app.core import sparse_encoder
from app.core.chunker import chunk_text
from app.core.claude_client import ClaudeClient
from app.core.contextual_chunker import add_contexts
from app.core.voyage_embed import VoyageEmbedder
from app.core.qdrant_store import QdrantStore, QdrantRegistry, PERSONA_TO_DOMAIN, DOMAINS
from app.ingestion.doc_parser import parse, _fix_common_typos

logger = logging.getLogger(__name__)

_LOG_DIR = config.DATA_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_file_handler = logging.FileHandler(
    _LOG_DIR / f"ingest-{datetime.now().strftime('%Y%m%d')}.log",
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logging.getLogger().addHandler(_file_handler)


# Prompt: short table summary (2-3 sentences, for search)
TABLE_DESCRIBE_PROMPT = (
    "Viết TÓM TẮT NGẮN (2-3 câu) về bảng Markdown dưới đây.\n"
    "Chỉ nêu bảng chứa gì, bao nhiêu mục, thông tin chính.\n"
    "VD: 'Bảng phân nhóm nhân sự gồm 3 nhóm: Khối Văn phòng 30%, "
    "Ban chỉ huy công trình 40%, Công nhân cơ hữu 30%.'\n"
    "Quy tắc:\n"
    "- Tối đa 3 câu.\n"
    "- Giữ chính xác tên riêng, số liệu.\n"
    "- KHÔNG dùng Markdown (**, ##, `, |).\n"
    "- Chỉ trả về plain text.\n"
)

# Prompt: Vision with document context
VISION_WITH_CONTEXT_PROMPT = (
    "Bạn đang đọc MỘT TRANG trong tài liệu. Dưới đây là NGỮ CẢNH từ các trang khác:\n"
    "---\n{context}\n---\n\n"
    "Hãy đọc hình ảnh trang này và viết lại NỘI DUNG thành plain text có cấu trúc.\n"
    "Yêu cầu:\n"
    "- Hiểu trang này trong ngữ cảnh tài liệu tổng thể ở trên.\n"
    "- Dòng đầu: mô tả ngắn trang này chứa gì, thuộc phần nào của tài liệu.\n"
    "- Nếu có bảng: mỗi dòng dữ liệu viết thành 1 dòng text riêng.\n"
    "- Nếu ô có màu mang ý nghĩa (xanh=hoàn thành, cam=đang làm, đỏ=quan trọng):\n"
    "  → Ghi gắn với dữ liệu. VD: 'Tất niên (T1-T2): đã hoàn thành.'\n"
    "- Bỏ qua cột/ô trống.\n"
    "- Giữ chính xác tên riêng, số liệu, tiếng Việt có dấu.\n"
    "- KHÔNG dùng Markdown (**, ##, `, |, ---).\n"
    "- KHÔNG mô tả format, font, kiểu bảng.\n"
    "- Chỉ trả về plain text.\n"
)


@dataclass
class IngestResult:
    doc_id: str
    num_chunks: int
    num_pages: int
    source_name: str


_registry: QdrantRegistry | None = None


def _get_registry() -> QdrantRegistry:
    """Lazy singleton — tránh tạo lại 20 store dict mỗi lần ingest."""
    global _registry
    if _registry is None:
        _registry = QdrantRegistry(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            vector_size=config.VOYAGE_DIM,
        )
    return _registry


_claude_client: ClaudeClient | None = None


def _get_claude_for_context() -> ClaudeClient:
    """Singleton ClaudeClient cho contextual chunking — Haiku model."""
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient(
            api_key=config.ANTHROPIC_API_KEY,
            model=config.CLAUDE_HAIKU_MODEL,
        )
    return _claude_client


def _resolve_domain_store(metadata: dict | None, source: str) -> QdrantStore:
    """Pick đúng tdi_{source}_{slug} theo metadata['domain'].

    Raise ValueError nếu metadata thiếu hoặc domain không hợp lệ — upload form
    bắt buộc gửi 1 trong 10 persona key.
    """
    domain_key = (metadata or {}).get("domain", "").strip()
    if not domain_key:
        raise ValueError(
            "metadata['domain'] là bắt buộc khi ingest — hãy chọn lĩnh vực ở form upload."
        )
    if domain_key not in PERSONA_TO_DOMAIN and domain_key not in DOMAINS:
        raise ValueError(
            f"domain={domain_key!r} không hợp lệ. "
            f"Chọn 1 trong: {DOMAINS} (slug) hoặc {list(PERSONA_TO_DOMAIN.keys())} (persona)."
        )
    return _get_registry().get_by_persona(domain_key, source)


# --- Table detection & processing ---

def _is_table_line(line: str) -> bool:
    s = line.strip()
    return (s.startswith("|") and s.endswith("|")) or bool(re.match(r"^\s*\|[\s\-:|]+\|\s*$", s))


def _has_markdown_tables(text: str) -> bool:
    return bool(re.search(r"\|-{2,}", text))


def _extract_tables_and_text(markdown: str) -> tuple[str, list[str]]:
    """Split markdown into text parts and individual tables."""
    lines = markdown.split("\n")
    text_lines: list[str] = []
    tables: list[str] = []
    current_table: list[str] = []
    in_table = False

    for line in lines:
        if _is_table_line(line):
            in_table = True
            current_table.append(line)
        else:
            if in_table:
                tables.append("\n".join(current_table))
                current_table = []
                in_table = False
            text_lines.append(line)

    if current_table:
        tables.append("\n".join(current_table))

    return "\n".join(text_lines).strip(), tables


def _table_empty_cell_ratio(table_md: str) -> float:
    lines = [l for l in table_md.strip().split("\n")
             if l.strip().startswith("|") and l.strip().endswith("|")
             and not re.match(r"^\s*\|[\s\-:|]+\|\s*$", l)]
    total = 0
    empty = 0
    for line in lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        total += len(cells)
        empty += sum(1 for c in cells if not c.strip())
    return empty / total if total > 0 else 0


def _strip_markdown_formatting(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    # Remove Docling image placeholders entirely (before general image handling)
    text = re.sub(r"!\[(?:image|Image\s*#?\d*|Figure\s*#?\d*|Picture\s*#?\d*|Photo\s*#?\d*)\]\([^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(?:Image|Figure|Picture|Photo)\s*#?\d+\](?!\()", "", text, flags=re.IGNORECASE)
    # Remaining images: keep meaningful alt text
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _llm_describe_table(table_md: str) -> str:
    """Short LLM description of table (2-3 sentences)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.CLAUDE_HAIKU_MODEL, max_tokens=512,
            system=TABLE_DESCRIBE_PROMPT,
            messages=[{"role": "user", "content": table_md}],
            temperature=0.1,
        )
        result = response.content[0].text.strip()
        return _fix_common_typos(result)
    except Exception:
        logger.exception("LLM table describe failed")
        return ""


def _vision_with_context(pdf_path: Path, context_text: str) -> str:
    """Vision for PDF pages with document context."""
    from app.ingestion.doc_parser import (
        _get_anthropic_client, _pdf_page_to_base64,
        _count_pdf_pages, _fix_common_typos,
    )
    from app.config import CLAUDE_HAIKU_MODEL

    client = _get_anthropic_client()
    if client is None:
        return ""

    num_pages = _count_pdf_pages(pdf_path)
    short_context = context_text[:2000] if len(context_text) > 2000 else context_text
    prompt = VISION_WITH_CONTEXT_PROMPT.format(context=short_context)

    all_text: list[str] = []
    for page_num in range(1, num_pages + 1):
        image_b64 = _pdf_page_to_base64(pdf_path, page_num)
        if not image_b64:
            continue
        try:
            response = client.messages.create(
                model=CLAUDE_HAIKU_MODEL, max_tokens=4096, temperature=0.1,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            result = response.content[0].text.strip()
            result = _fix_common_typos(result)
            if result:
                all_text.append(result)
                logger.info("Vision+context page %d: %d chars", page_num, len(result))
        except Exception:
            logger.exception("Vision+context failed for page %d", page_num)

    return "\n\n".join(all_text)


def _process_content(text: str) -> tuple[str, str, bool]:
    """Process content into (embed_text, table_data, needs_vision).

    - Text only → strip formatting, no LLM
    - Table with data (<50% empty) → keep in table_data, LLM summary for embed
    - Table with empty cells (>50%) → discard, needs Vision
    """
    if not _has_markdown_tables(text):
        clean = _strip_markdown_formatting(text)
        return clean, "", False

    text_part, tables = _extract_tables_and_text(text)
    text_part = _strip_markdown_formatting(text_part)

    good_tables: list[str] = []
    needs_vision = False

    for table_md in tables:
        empty_ratio = _table_empty_cell_ratio(table_md)
        if empty_ratio > 0.5:
            logger.info("Table discarded: %.0f%% empty cells (needs Vision)", empty_ratio * 100)
            needs_vision = True
        else:
            good_tables.append(table_md)

    table_data = "\n\n".join(good_tables)
    table_desc = _llm_describe_table(table_data) if good_tables else ""

    parts = [p for p in [text_part, table_desc] if p]
    embed_text = "\n\n".join(parts)

    return embed_text, table_data, needs_vision


# --- Main ingest ---

def ingest_document(
    file_path: str,
    original_name: str,
    metadata: dict | None = None,
    document_id: str | None = None,
) -> IngestResult:
    path = Path(file_path)
    logger.info("Ingesting document: %s", original_name)

    parsed = parse(path)
    doc_id = parsed["doc_id"]
    uploaded_at = parsed["uploaded_at"]
    content = parsed["content"]

    # Route theo domain — validate metadata trước khi parse nội dung nặng là lý tưởng,
    # nhưng parse đã xong ở trên (không thể rollback), nên validate ở đây vẫn OK
    # vì delete_by_filter/upsert mới là phần tốn tiền (vector embed + Qdrant write).
    store = _resolve_domain_store(metadata, "docs")

    # Delete old chunks theo doc_id trong chính collection domain đó — nếu user
    # đổi domain cho cùng file, bản cũ ở domain khác sẽ vẫn còn: đó là hành vi
    # có chủ đích (user tự reup/xoá thủ công nếu cần dời).
    try:
        store.delete_by_filter({
            "must": [{"key": "doc_id", "match": {"value": doc_id}}]
        })
        logger.info(
            "Deleted old chunks for doc_id=%s in %s", doc_id, store.collection,
        )
    except Exception:
        logger.warning(
            "Could not delete old chunks for doc_id=%s in %s",
            doc_id, store.collection,
        )

    # Flatten pages — kéo theo images metadata per page (PDF/parser mới trả về).
    # Page chỉ có ảnh không text vẫn được giữ lại để ảnh không "biến mất".
    # `sheet_name` chỉ XLSX có (parse_xlsx set), PDF/DOCX là "" → display logic
    # ở payload chỉ thêm field này khi non-empty.
    if isinstance(content, list):
        page_data: list[tuple[int, str, list[dict], str]] = [
            (item["page"], item["text"], item.get("images", []) or [],
             item.get("sheet_name", "") or "")
            for item in content
            if item["text"].strip() or item.get("images")
        ]
    else:
        page_data = [(1, content, [], "")]

    num_pages = len(page_data)
    embedder = VoyageEmbedder(api_key=config.VOYAGE_API_KEY, model=config.VOYAGE_MODEL)

    # Full doc text: dùng cho contextual chunking (Anthropic recipe). Concat tất
    # cả page text — Haiku sẽ thấy toàn cảnh tài liệu khi sinh ngữ cảnh cho
    # mỗi chunk. Prompt caching giữ doc cache trong 5 phút TTL → các chunk
    # sau hit cache.
    full_doc_text = "\n\n".join(t for _, t, _, _ in page_data if t and t.strip())
    claude_for_ctx = _get_claude_for_context() if config.CONTEXTUAL_CHUNKING else None

    all_points: list[dict] = []
    chunk_index = 0

    for page_num, text, page_images, sheet_name in page_data:
        embed_text, table_data, needs_vision = _process_content(text)

        # Vision with context for tables with empty cells (colors)
        if needs_vision:
            logger.info("Page %d: tables with empty cells, calling Vision with context", page_num)
            vision_text = _vision_with_context(path, embed_text)
            if vision_text:
                embed_text = embed_text + "\n\n" + vision_text if embed_text else vision_text

        # Caption KHÔNG gộp vào embed_text. Caption sẽ được search qua synthetic
        # image-caption chunks (sinh sau vòng for-loop này) — embed sạch tiếng
        # Việt, score cao khi query match caption. Text chunk giữ images[]
        # đầy đủ trong payload để recall theo text-relevant query.
        chunks = chunk_text(embed_text, max_tokens=config.CHUNK_MAX_TOKENS, overlap_tokens=config.CHUNK_OVERLAP_TOKENS)
        chunks = [c for c in chunks if c.text.strip() and c.token_count >= 10]
        if not chunks:
            # Trang chỉ có ảnh, không có text/caption đủ dài để chunk → ảnh không
            # vào store. v1 chấp nhận; v2 có thể emit chunk placeholder để giữ ảnh.
            continue

        # Contextual chunking — sinh prefix ngữ cảnh cho mỗi chunk dựa trên
        # toàn doc. Nếu disabled → trả ContextualChunk(text=t, context="", embed_text=t).
        chunk_texts_raw = [c.text for c in chunks]
        if claude_for_ctx is not None:
            ctxs = add_contexts(claude_for_ctx, full_doc_text, chunk_texts_raw)
        else:
            from app.core.contextual_chunker import ContextualChunk
            ctxs = [ContextualChunk(text=t, context="", embed_text=t) for t in chunk_texts_raw]

        # Slim images payload — duplicate vào MỌI chunk của trang để recall không
        # phụ thuộc chunk #1 phải win score (caption có khi nằm cuối embed_text).
        # `url` = full public URL S3, FE load thẳng. Không có local filename nữa.
        images_payload = [
            {
                "image_id": img["image_id"],
                "url": img.get("url", ""),
                "caption": img.get("caption", ""),
                "page": img.get("page", page_num),
                "ord": img.get("ord", 0),
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            }
            for img in page_images
            if img.get("url")  # Bỏ ảnh upload S3 fail (url rỗng) khỏi payload
        ]

        # Embed dùng augmented text (context + chunk). Voyage không phân biệt —
        # embedding sẽ phản ánh cả ngữ cảnh và nội dung chunk.
        embed_inputs = [c.embed_text for c in ctxs]
        vectors = embedder.embed_documents(embed_inputs)

        # Sparse encode CHỈ nếu hybrid bật. Encode local nên rẻ.
        if config.HYBRID_RETRIEVAL:
            sparse_vecs = sparse_encoder.encode_batch(embed_inputs)
        else:
            sparse_vecs = [None] * len(ctxs)

        for i, (chunk, ctx, vector, sparse_vec) in enumerate(zip(chunks, ctxs, vectors, sparse_vecs)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}-{chunk_index}"))
            payload: dict = {
                "source_type": "document",
                "doc_id": doc_id,
                "source_name": original_name,
                "page": page_num,
                "chunk_index": chunk_index,
                # ``text``: chunk gốc — vào Citations API. KHÔNG gồm context_prefix
                # để Claude cite đúng câu thật trong tài liệu, không cite text
                # tổng hợp do Haiku sinh ra.
                "text": chunk.text,
                "heading_path": chunk.heading_path,
                "uploaded_at": uploaded_at,
            }
            if sheet_name:
                payload["sheet_name"] = sheet_name
            if document_id:
                payload["document_id"] = document_id
            # Lưu context riêng — debug only, không ảnh hưởng search/citations.
            if ctx.context:
                payload["context"] = ctx.context
            if table_data and i == 0:
                payload["table_data"] = table_data
            if images_payload:
                payload["images"] = images_payload
            # Store searchable metadata fields at top level
            if metadata:
                if metadata.get("domain"):
                    payload["domain"] = metadata["domain"]
                if metadata.get("title"):
                    payload["title"] = metadata["title"]
                if metadata.get("description"):
                    payload["description"] = metadata["description"]
                if metadata.get("tags"):
                    payload["tags"] = metadata["tags"]
                if metadata.get("url"):
                    payload["url"] = metadata["url"]
                payload["extra_metadata"] = metadata
            all_points.append({
                "id": point_id,
                "vector": vector,
                "sparse": sparse_vec,
                "payload": payload,
            })
            chunk_index += 1

        logger.info(
            "Page %d: %d chunks, table_data=%s, vision=%s, images=%d, contextual=%s, sparse=%s",
            page_num, len(chunks), bool(table_data), needs_vision,
            len(images_payload), config.CONTEXTUAL_CHUNKING, config.HYBRID_RETRIEVAL,
        )

    # === Synthetic image-caption chunks ===
    # Mỗi ảnh duy nhất (theo image_id) → 1 chunk synthetic với text=caption.
    # Mục đích: query tiếng Việt "có sơ đồ X không?" match trực tiếp caption
    # → embed thuần tiếng Việt → score cao + chỉ 1 ảnh trong payload (precision).
    # Text chunks gốc vẫn giữ images[] đầy đủ → recall fallback khi query
    # match nội dung doc mà không match caption nào.
    #
    # chunk_type="image_caption" cho debug và để team có thể tune sau (vd
    # boost/penalty score) ở reranker mà không phải re-ingest.
    seen_image_ids: set[str] = set()
    synthetic_units: list[tuple[dict, str, str]] = []  # (image_meta, caption, sheet_name)
    for page_num, _, page_images, sheet_name in page_data:
        for img in page_images:
            image_id = img["image_id"]
            caption = (img.get("caption") or "").strip()
            # Skip nếu thiếu caption hoặc URL S3 (upload fail) — synthetic chunk
            # không có URL thì FE không hiển thị được.
            if not caption or not img.get("url") or image_id in seen_image_ids:
                continue
            seen_image_ids.add(image_id)
            synthetic_units.append(
                ({**img, "page": img.get("page", page_num)}, caption, sheet_name)
            )

    if synthetic_units:
        captions = [c for _, c, _ in synthetic_units]
        # 1 batch embed cho tất cả captions — Voyage chấp nhận multi-input.
        # Caption chunks dùng plain dense embed (không context augmentation,
        # không sparse) — mục tiêu là cross-lingual semantic match cho query
        # tiếng Việt → caption ngắn, đủ rồi.
        caption_vectors = embedder.embed_documents(captions)

        for (img, caption, sheet_name), vector in zip(synthetic_units, caption_vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}-{chunk_index}"))
            slim_image = {
                "image_id": img["image_id"],
                "url": img.get("url", ""),
                "caption": caption,
                "page": img["page"],
                "ord": img.get("ord", 0),
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            }
            payload: dict = {
                "source_type": "document",
                "chunk_type": "image_caption",
                "doc_id": doc_id,
                "source_name": original_name,
                "page": img["page"],
                "chunk_index": chunk_index,
                "text": caption,
                "heading_path": [],
                "uploaded_at": uploaded_at,
                "images": [slim_image],
            }
            if sheet_name:
                payload["sheet_name"] = sheet_name
            if document_id:
                payload["document_id"] = document_id
            if metadata:
                if metadata.get("domain"):
                    payload["domain"] = metadata["domain"]
                if metadata.get("title"):
                    payload["title"] = metadata["title"]
                if metadata.get("description"):
                    payload["description"] = metadata["description"]
                if metadata.get("tags"):
                    payload["tags"] = metadata["tags"]
                if metadata.get("url"):
                    payload["url"] = metadata["url"]
                payload["extra_metadata"] = metadata
            all_points.append({
                "id": point_id,
                "vector": vector,
                "sparse": None,  # caption chunks không hybrid
                "payload": payload,
            })
            chunk_index += 1

        logger.info("Synthetic image-caption chunks: %d (deduped from %d images)",
                    len(synthetic_units),
                    sum(len(p[2]) for p in page_data))

    store.upsert(all_points)
    logger.info("Ingested %s: doc_id=%s pages=%d chunks=%d (incl. %d image-captions)",
                original_name, doc_id, num_pages, chunk_index, len(synthetic_units))
    return IngestResult(doc_id=doc_id, num_chunks=chunk_index,
                        num_pages=num_pages, source_name=original_name)
