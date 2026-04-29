"""Demo server: hỏi → answer + ảnh thumbnails (load từ S3 public URL).

Mục đích: thấy ngay UX trả ảnh trước khi BE/FE wire `build_sources_mapping`.
Demo này TỰ aggregate ảnh từ payload.images[] của hits — chính là logic Surface
3 mà BE sẽ implement chính thức.

Run:
    cd /Users/russia/Documents/Projects/trungtamtrithuc-rag
    venv/bin/python scratch/demo_qa_with_images.py [domain]

→ Mở browser http://localhost:8088 → form input câu hỏi → submit → answer + grid ảnh.

Default domain: 'bim'. Có thể đổi ở argv: `python ... bim` / `marketing` / ...

Cần keys trong .env: ANTHROPIC, VOYAGE, QDRANT, S3.
"""
from __future__ import annotations

import logging
import sys
from html import escape
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Form  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
import uvicorn  # noqa: E402

from app.core import config  # noqa: E402
from app.core.claude_client import ClaudeClient  # noqa: E402
from app.core.qdrant_store import QdrantRegistry, VMediaReadOnlyStore  # noqa: E402
from app.core.voyage_embed import VoyageEmbedder  # noqa: E402
from app.rag.image_markers import (  # noqa: E402
    build_available_images_section, resolve_image_markers,
)
from app.rag.prompt_builder import (  # noqa: E402
    build_documents_block, build_system_prompt, build_user_turn,
)
from app.rag.reranker import CrossEncoderReranker  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402


VALID_DOMAINS = {"bim", "mep", "marketing", "phap_ly", "san_xuat",
                 "cntt", "nhan_su", "tai_chinh", "kinh_doanh", "thiet_ke"}
DEFAULT_DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "bim"
if DEFAULT_DOMAIN not in VALID_DOMAINS:
    print(f"❌ domain={DEFAULT_DOMAIN!r} không hợp lệ. Chọn 1 trong: {sorted(VALID_DOMAINS)}")
    sys.exit(1)

# Init dependencies — 1 lần khi import module để tránh load lại mỗi request.
print(f"Initializing demo dependencies (domain={DEFAULT_DOMAIN})...")
_voyage = VoyageEmbedder(api_key=config.VOYAGE_API_KEY, model=config.VOYAGE_MODEL)
_registry = QdrantRegistry(
    url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY,
    vector_size=config.VOYAGE_DIM,
    vector_name=config.QDRANT_VECTOR_NAME,
)
_vmedia = VMediaReadOnlyStore(
    url=config.QDRANT_VMEDIA_URL,
    vmedia_api_key=config.QDRANT_VMEDIA_API_KEY,
    collections=config.VMEDIA_COLLECTIONS,
)
_retriever = Retriever(voyage=_voyage, registry=_registry, vmedia_store=_vmedia)
_reranker = CrossEncoderReranker(min_score=0.0)
_claude = ClaudeClient(api_key=config.ANTHROPIC_API_KEY, model=config.CLAUDE_MODEL)
print("Ready.\n")


_MAX_IMAGES = 6


def _collect_images(hits) -> list[dict]:
    """Aggregate unique images từ hits, dedupe theo image_id, giữ rerank score
    cao nhất. Đây là logic mẫu cho `build_sources_mapping` mà BE sẽ implement."""
    seen: dict[str, dict] = {}
    for hit in hits:
        for img in hit.payload.get("images", []) or []:
            iid = img.get("image_id")
            url = img.get("url")
            if not iid or not url:
                continue
            existing = seen.get(iid)
            if existing is None or hit.score > existing["score"]:
                seen[iid] = {
                    "image_id": iid,
                    "url": url,
                    "caption": img.get("caption", ""),
                    "page": img.get("page"),
                    "score": hit.score,
                    "source_name": hit.payload.get("source_name", ""),
                }
    images = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return images[:_MAX_IMAGES]


def _ask(query: str, domain: str) -> tuple[str, list[dict], int]:
    """Run retrieve → rerank → Claude (với inline image markers) → return
    (answer_with_resolved_markers, leftover_images, num_hits).

    `answer_with_resolved_markers` là markdown text có `![](url)` ngay tại chỗ
    Claude đặt — FE render markdown sẽ ra ảnh inline.
    `leftover_images` = ảnh top hits MÀ Claude KHÔNG chèn marker → vẫn hiển thị
    cuối câu trả lời như fallback (tránh ảnh quan trọng bị bỏ sót).
    """
    hits = _retriever.retrieve(
        query=query, top_k=10, domain=domain, sources=["documents"],
    )
    if not hits:
        return ("Tài liệu TDI hiện chưa có thông tin về câu hỏi này.", [], 0)

    hits = _reranker.rerank(query, hits, top_k=5)

    # Ghép image section vào docs_block để Claude thấy danh sách image_id+caption
    # ngay trong context. Claude sẽ chèn `[IMG:image_id]` vào answer khi viết
    # tới câu liên quan.
    docs_block = build_documents_block(hits)
    images_section = build_available_images_section(hits)
    if images_section:
        docs_block = docs_block + "\n\n" + images_section

    system_prompt = build_system_prompt(domain)
    user_turn = build_user_turn(query, docs_block, "")
    messages = [{"role": "user", "content": user_turn}]
    answer_raw = _claude.generate(system_prompt=system_prompt, messages=messages)

    # Post-process: replace `[IMG:abc123]` thành markdown `![cap](url)`.
    answer_resolved = resolve_image_markers(answer_raw, hits)

    # Tính leftover: ảnh top hits không được Claude chèn — fallback hiển thị
    # cuối answer để user vẫn thấy (giảm risk Claude bỏ sót ảnh quan trọng).
    used_ids = set()
    import re as _re
    for m in _re.finditer(r"\[IMG:([a-f0-9]{16})\]", answer_raw):
        used_ids.add(m.group(1))
    all_imgs = _collect_images(hits)
    leftover = [img for img in all_imgs if img["image_id"] not in used_ids]
    return answer_resolved, leftover, len(hits)


# --- HTML rendering (inline, không cần Jinja) ---

_PAGE_TPL = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Demo RAG kèm ảnh inline</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px;
          margin: 30px auto; padding: 0 20px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #777; font-size: 13px; }}
  textarea {{ width: 100%; height: 80px; font: inherit; font-size: 16px;
              padding: 10px; border: 1px solid #ccc; border-radius: 6px;
              box-sizing: border-box; }}
  button {{ padding: 10px 22px; font-size: 16px; cursor: pointer;
            background: #2563eb; color: #fff; border: 0; border-radius: 6px;
            margin-top: 8px; }}
  button:hover {{ background: #1e4ec0; }}
  .answer {{ background: #f6f8fa; padding: 16px 20px;
             border-radius: 8px; margin-top: 20px; line-height: 1.6; }}
  .answer p {{ margin: 0.5em 0; }}
  .answer img {{ max-width: 100%; max-height: 360px; display: block;
                  margin: 12px auto; border: 1px solid #ddd;
                  border-radius: 6px; background: #fff; }}
  .leftover {{ display: grid;
               grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
               gap: 12px; margin-top: 12px; }}
  .leftover .card {{ border: 1px solid #ddd; border-radius: 6px;
                      overflow: hidden; background: #fff; }}
  .leftover .card img {{ width: 100%; max-height: 180px;
                          object-fit: contain; background: #f9f9f9; }}
  .leftover .card .cap {{ padding: 6px 8px; font-size: 12px; color: #555; }}
  .empty {{ color: #999; font-style: italic; margin-top: 12px; }}
</style></head>
<body>
<h1>Demo RAG — ảnh inline trong câu trả lời</h1>
<p class="meta">Domain: <code>{domain}</code> · Strategy B: Claude tự chèn marker [IMG:id] → BE replace bằng markdown ![](url) → FE render inline.</p>
<form method="POST" action="/ask">
  <textarea name="query" placeholder="VD: Có sơ đồ kiến trúc YOLOv8 nào không?">{query}</textarea>
  <br><button type="submit">Hỏi</button>
</form>
{result}
</body></html>
"""


def _markdown_answer_to_html(answer_md: str) -> str:
    """Render markdown answer (có `![cap](url)` cho ảnh inline) thành HTML.

    Dùng `markdown` library — đã có trong requirements.txt. Auto-escape HTML
    nguy hiểm; chỉ output `<p>`, `<img>`, `<strong>`, `<em>`, `<code>` etc.
    """
    import markdown as _md
    return _md.markdown(answer_md, extensions=["nl2br"])


def _render_leftover_card(img: dict) -> str:
    cap = escape(img.get("caption", "") or "(no caption)")
    cap_short = cap[:120] + ("…" if len(cap) > 120 else "")
    url = escape(img["url"])
    page = img.get("page", "?")
    return f'''<div class="card">
  <a href="{url}" target="_blank" rel="noopener"><img src="{url}" alt="{cap}" loading="lazy"></a>
  <div class="cap">page {page}: {cap_short}</div>
</div>'''


def _render_result(query: str, answer_md: str, leftover: list[dict], num_hits: int) -> str:
    if not query:
        return ""
    answer_html = _markdown_answer_to_html(answer_md)
    parts = [
        '<h2>Câu trả lời</h2>',
        f'<div class="answer">{answer_html}</div>',
        f'<p class="meta">Dựa trên {num_hits} hit (sau rerank). '
        f'Ảnh inline được Claude chèn trực tiếp tại đoạn liên quan.</p>',
    ]
    if leftover:
        parts.append(
            f'<h3>Ảnh khác trong top hits ({len(leftover)} — Claude không chèn inline)</h3>'
        )
        parts.append('<div class="leftover">')
        parts.extend(_render_leftover_card(img) for img in leftover)
        parts.append('</div>')
    return "\n".join(parts)


# --- Routes ---

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def home():
    return _PAGE_TPL.format(domain=DEFAULT_DOMAIN, query="", result="")


@app.post("/ask", response_class=HTMLResponse)
async def ask(query: str = Form(...)):
    q = query.strip()
    if not q:
        return _PAGE_TPL.format(domain=DEFAULT_DOMAIN, query="", result="")
    answer, images, num_hits = _ask(q, DEFAULT_DOMAIN)
    return _PAGE_TPL.format(
        domain=DEFAULT_DOMAIN,
        query=escape(q),
        result=_render_result(q, answer, images, num_hits),
    )


if __name__ == "__main__":
    # Port 8089 — để không đụng API_PORT=8088 của run.sh (FastAPI production)
    PORT = 8089
    print(f"→ Mở browser: http://localhost:{PORT}  (domain={DEFAULT_DOMAIN})\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
