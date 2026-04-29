# Trung Tâm Tri Thức — RAG Chatbot

Hệ thống hỏi đáp doanh nghiệp dựa trên RAG (Retrieval-Augmented Generation): nạp tài liệu (PDF/DOCX/XLSX/TXT/MD), video (MP4/YouTube/Playlist), trả lời tiếng Việt kèm trích dẫn nguồn và gợi ý câu hỏi tiếp theo.

**Hybrid Retrieval (Anthropic Contextual Retrieval recipe)** — Voyage `voyage-3` dense + BM25 sparse (underthesea VN tokenizer) qua Qdrant Query API + RRF fusion + Contextual chunking với Haiku prompt caching. Fix triệt để brittleness: query "form order thiết kế" và "hướng dẫn điền form nội dung order thiết kế" cùng ra top chunk.

**AI Auto Metadata** — khi upload, hệ thống tự sinh `title`, `description`, `domain` (phân loại 10 lĩnh vực), `tags` từ nội dung. FE prefill form với badge ✦ AI / ▶ YT để user review trước khi commit.

## Kiến trúc tổng thể

```
  Upload (file / video / YouTube URL / Playlist)
            │
            ▼
  ┌──────────────────────────────┐
  │ Ingestion pipeline           │
  │  PDF: 3-tier                 │
  │   ├ Docling (local)          │
  │   ├ Claude Vision (fallback) │
  │   └ pdfplumber (last resort) │
  │  DOCX: Docling / python-docx │
  │  XLSX: openpyxl              │
  │  Video: yt-dlp + Whisper     │
  │  YouTube: transcript-api     │
  └──────────────┬───────────────┘
                 │ table detect + LLM describe + Vision+context
                 ▼
  ┌──────────────────────────────┐
  │ Chunker (heading-aware)      │
  │   max 700 tok, overlap 80    │
  └──────────────┬───────────────┘
                 ▼
  ┌──────────────────────────────┐
  │ Contextual Chunker (Haiku)   │
  │   prepend ngữ cảnh tài liệu  │
  │   + prompt caching ephemeral │
  └──────────────┬───────────────┘
                 ▼
  ┌──────────────────────────────┐    upsert    ┌──────────────────────────────────┐
  │ Voyage AI dense (voyage-3)   │ ───────────► │ Qdrant Cloud (hybrid schema)     │
  │   1024-dim                   │              │  20 collections:                 │
  │ + BM25 sparse (underthesea)  │              │   tdi_docs_{10 domain}           │
  │   FNV-1a hash, TF only       │              │   tdi_videos_{10 domain}         │
  │   IDF do Qdrant tự nhân      │              │  vmedia_*  (READ ONLY, legacy)   │
  └──────────────────────────────┘              │  ttt_memory (conv recall, dense) │
                                                └──────────────┬───────────────────┘
                                                               │ /points/query +
  Câu hỏi người dùng                                           │ prefetch[dense,sparse]
            │                                                  │ + RRF fusion
            ├──────────────► ┌──────────────────────────────────────────┐
            │                │  Retriever (parallel, 20+vmedia)         │
            │                └──────────────────────┬───────────────────┘
            │                                       ▼
            │                ┌──────────────────────────────────────────┐
            │                │  Reranker (cross-encoder)                │
            │                └──────────────────────┬───────────────────┘
            │                                       ▼
            │                ┌──────────────────────────────────────────┐
            │                │  Prompt builder                          │
            │                │   system + domain preset + context       │
            │                │   + table_data + history                 │
            │                └──────────────────────┬───────────────────┘
            │                                       ▼
            │                ┌──────────────────────────────────────────┐
            │                │  Claude Sonnet 4 (Anthropic)             │
            │                └──────────────────────┬───────────────────┘
            │                                       ▼
            └──► Answer + Sources + Suggested Questions (JSON)
```

## Yêu cầu hệ thống

- Python 3.12+ (macOS Apple Silicon M1-M4 hoặc Linux)
- RAM ≥ 16GB (Docling + Vision)
- `poppler` cho pdf2image: `brew install poppler` / `apt install poppler-utils`
- `ffmpeg` cho Whisper (tuỳ chọn, cho video local): `brew install ffmpeg`
- Tài khoản: **Anthropic API**, **Voyage AI**, **Qdrant Cloud**

## Cài đặt

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
brew install poppler            # macOS
# apt install poppler-utils     # Linux

cp .env.example .env            # điền API keys (xem bảng bên dưới)
```

### Migration Qdrant schema (chỉ chạy 1 lần khi switch sang hybrid)

Code hybrid yêu cầu Qdrant collection có schema `dense + sparse` (không tương thích schema cũ single dense `vector_name=""`).

```bash
# Trường hợp Qdrant rỗng → KHÔNG cần làm gì, server start tự tạo
# Trường hợp Qdrant có collection schema CŨ → chạy reset 1 lần:

QDRANT_URL=<prod-url> QDRANT_API_KEY=<prod-key> \
  ./venv/bin/python scripts/reset_qdrant_collections.py
```

⚠️ Script DELETE toàn bộ chunk data trong 20 collection `tdi_*`. Backup nếu có data quan trọng. KHÔNG đụng `ttt_memory` (conv pair).

Check schema hiện tại:

```bash
QDRANT_URL=<prod> QDRANT_API_KEY=<key> ./venv/bin/python -c "
import os, requests
url, key = os.environ['QDRANT_URL'].rstrip('/'), os.environ['QDRANT_API_KEY']
r = requests.get(f'{url}/collections/tdi_docs_marketing', headers={'api-key': key}, timeout=10)
if r.status_code == 404: print('EMPTY: chưa có — không cần reset')
else:
    p = r.json()['result']['config']['params']
    has_dense, has_sparse = 'dense' in p.get('vectors', {}), 'sparse' in p.get('sparse_vectors', {})
    print('HYBRID OK' if has_dense and has_sparse else 'LEGACY: PHẢI reset')
"
```

## Cấu hình `.env`

### Bắt buộc

| Biến | Mô tả |
|------|-------|
| `ANTHROPIC_API_KEY` | Key Claude API |
| `VOYAGE_API_KEY` | Key Voyage AI |
| `VOYAGE_MODEL` | Mặc định `voyage-3` |
| `VOYAGE_DIM` | Mặc định `1024` |
| `QDRANT_URL` | URL Qdrant cluster chính |
| `QDRANT_API_KEY` | Key R/W cho 20 collection `tdi_{docs,videos}_{slug}` + `ttt_memory` |

### Tuỳ chọn

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Model trả lời chính |
| `CLAUDE_HAIKU_MODEL` | `claude-haiku-4-5-20251001` | Vision + describe table + contextual chunking |
| `CHUNK_MAX_TOKENS` | `700` | Kích thước chunk tối đa |
| `CHUNK_OVERLAP_TOKENS` | `80` | Overlap giữa các chunk |
| `TOP_K` | `7` | Số hit trước rerank |
| `RERANK_TOP_K` | `5` | Số hit sau rerank |
| `RERANKER_DEVICE` | auto | Ép device cho cross-encoder: `cpu`/`cuda`/`mps`. Mặc định auto-detect (CUDA > MPS > CPU). |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | |

### Hybrid Retrieval (Anthropic Contextual Retrieval recipe)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `HYBRID_RETRIEVAL` | `1` | Bật BM25 sparse + RRF fusion. Tắt → fallback dense-only |
| `CONTEXTUAL_CHUNKING` | `1` | Bật Haiku context prefix lúc ingest (Anthropic recipe). Tắt → tiết kiệm ~$0.075/doc |
| `BM25_HASH_BUCKETS` | `16777216` | FNV-1a hash buckets cho BM25 vocab (2^24). Ít khi cần đổi |
| `HYBRID_PREFETCH_LIMIT` | `30` | Số candidate mỗi nhánh dense/sparse pull trước RRF fusion |

### Conversation Memory (Hybrid 3 tầng)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `CONV_COLLECTION` | `ttt_memory` | Qdrant collection lưu conversation pairs |
| `CONV_WINDOW_TURNS` | `3` | Số pair (user+bot) giữ trong sliding window |
| `CONV_SUMMARY_MAX_TOKENS` | `400` | Giới hạn rolling summary |
| `CONV_RECALL_TOP_K` | `5` | Số pair Qdrant recall mỗi lần |
| `CONV_RECALL_MIN_SCORE` | `0.3` | Score tối thiểu để recall (Voyage cosine) |
| `CONV_REWRITE_MIN_LEN` | `40` | Query ngắn hơn ngưỡng này sẽ được rewrite |
| `CONV_MIN_USER_CHARS` | `20` | Guard 1 — user_msg ngắn hơn → skip upsert |
| `CONV_MIN_BOT_CHARS` | `40` | Guard 1 — bot_msg ngắn hơn + không có `Nguồn:` → skip |
| `CONV_DEDUP_THRESHOLD` | `0.92` | Guard 3 — cosine với pair cũ vượt ngưỡng → skip, chỉ update `last_seen_at` |
| `CONV_HASH_CACHE_SIZE` | `2000` | Guard 2 — số MD5 hash gần nhất giữ trong RAM để chặn exact dup |

### Collection READ-ONLY (vmedia)

| Biến | Mô tả |
|------|-------|
| `QDRANT_VMEDIA_URL` | URL cluster vmedia (riêng) |
| `QDRANT_VMEDIA_API_KEY` | Key **CHỈ READ** — tuyệt đối không dùng để upsert/delete |
| `VMEDIA_COLLECTIONS` | CSV: `vmedia_content,vmedia_design,...` |

### YouTube proxy (vượt IP block)

Ưu tiên theo thứ tự:

| Biến | Mô tả |
|------|-------|
| `YOUTUBE_PROXY_LIST` | CSV hoặc xuống dòng: `http://user:pass@host:port` (xoay vòng) |
| `WEBSHARE_PROXY_USERNAME` + `WEBSHARE_PROXY_PASSWORD` | Webshare rotating endpoint |
| `YOUTUBE_PROXY_HTTP` / `YOUTUBE_PROXY_HTTPS` | Proxy cố định |
| `YOUTUBE_TRANSCRIPT_MAX_RETRIES` | Mặc định `10` |
| `YOUTUBE_TRANSCRIPT_RETRY_DELAY` | Mặc định `1.5` giây |

## Chạy

```bash
# Dev (auto-reload)
./run.sh

# Trực tiếp
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Docker
docker build -t ttt-chatbot .
docker run --env-file .env -p 8000:8000 ttt-chatbot
```

Truy cập:

- Trang chủ: `http://localhost:8000/`
- Chat: `http://localhost:8000/chat.html`
- Nạp tài liệu: `http://localhost:8000/ingest.html`
- Tri thức: `http://localhost:8000/knowledge.html`
- Swagger: `http://localhost:8000/docs`

## Cấu trúc thư mục

```
trungtamtrithuc/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Biến môi trường
│   ├── schemas.py              # Pydantic I/O
│   ├── api/
│   │   ├── chat.py             # POST /api/chat/ (JSON) + /api/chat/stream (SSE)
│   │   └── ingest.py           # POST /api/ingest/{file,video/file,youtube,youtube-playlist}
│   ├── core/
│   │   ├── chunker.py          # Heading-aware chunking (tiktoken)
│   │   ├── contextual_chunker.py  # Anthropic Contextual Retrieval (Haiku + cache)
│   │   ├── sparse_encoder.py   # BM25 sparse vector (underthesea + FNV-1a hash)
│   │   ├── claude_client.py    # Anthropic client wrapper
│   │   ├── voyage_embed.py     # Voyage AI dense embedder
│   │   ├── qdrant_store.py     # Hybrid R/W (dense+sparse) + VMediaReadOnlyStore
│   │   ├── session_memory.py   # Sliding window + rolling summary (file-backed)
│   │   ├── conv_memory.py      # Vector recall cross-session (Qdrant ttt_memory)
│   │   ├── conv_summarizer.py  # Rolling summary updater (Claude Haiku)
│   │   └── conv_query_rewriter.py  # Zero-anaphora query rewrite (Haiku)
│   ├── ingestion/
│   │   ├── doc_parser.py       # 3-tier PDF parsing + typo fix
│   │   ├── doc_pipeline.py     # Table detect/process + embed + store
│   │   ├── metadata_generator.py # AI auto-gen metadata (Haiku + tool use, Pydantic schema)
│   │   ├── video_pipeline.py   # Video ingest (YouTube + local + playlist)
│   │   ├── video_transcriber.py# Whisper transcription
│   │   └── youtube_fetcher.py  # YouTube transcript + full metadata (yt-dlp)
│   └── rag/
│       ├── chain.py            # Retrieve → rerank → generate → parse suggestions
│       ├── retriever.py        # Multi-source parallel search
│       ├── reranker.py         # Cross-encoder reranker
│       └── prompt_builder.py   # System prompt + context + table_data
├── web/                        # Static frontend (HTML/CSS/JS)
├── data/{uploads,logs,sessions}/   # Runtime (auto-created)
├── scripts/
│   ├── diag_conv_memory.py     # Diag Qdrant ttt_memory: count theo user, test retrieve
│   ├── clean_poisoned_pairs.py # Dọn pair "không tìm thấy" gây feedback loop
│   ├── check_memory_collection.py  # Show schema + sample payload ttt_memory
│   ├── seed_two_users.py       # Seed 2 test user vào Qdrant
│   ├── test_fix_e2e.py         # E2E test cross-session recall + tính toán
│   ├── test_conversation_memory.py  # Kịch bản test 3 tầng hybrid memory
│   ├── reset_qdrant_collections.py  # Drop + recreate 20 coll với hybrid schema
│   ├── test_hybrid_e2e.py       # E2E ingest 2 file + test query brittleness
│   └── test_hybrid_advanced.py  # 10 stress test groups (diacritic, mixed VN+EN, ...)
├── docs/{PROJECT_OVERVIEW,INTEGRATION}.md
├── requirements.txt
├── run.sh
└── Dockerfile
```

## API Reference

Base URL: `http://localhost:8000`

### `GET /health`

```json
{"status": "ok"}
```

### AI Auto-Metadata (preview before ingest)

3 endpoint `POST .../preview` **không lưu Qdrant** — chỉ parse + gọi Haiku để sinh metadata (title, description, domain, tags). FE gọi preview khi user chọn file/dán URL → prefill form → user review → submit endpoint ingest chính.

| Endpoint | Input | Sinh field gì | Thời gian |
|----------|-------|---------------|-----------|
| `POST /api/ingest/file/preview` | multipart file | tất cả 4 field từ nội dung doc | ~2-5s |
| `POST /api/ingest/video/file/preview` | multipart file video | tất cả 4 field từ transcript Whisper | 30s-2 phút (tuỳ Groq/local) |
| `POST /api/ingest/youtube/preview?url=…` | query `url` | Video: `title/description` từ YouTube, `domain/tags` từ AI. Playlist: `playlist_*` fields (title, uploader, video_count…) — không chạy LLM. | 1-3s |

Response schema chung:
```json
{
  "status": "ok" | "partial" | "error" | "skip",
  "message": "...",
  "metadata": {
    "title": "...", "description": "...", "domain": "bim", "tags": ["...", "..."],
    "thumbnail": "...", "channel": "...", "duration_sec": 1234    // chỉ YouTube
  }
}
```

Domain nằm trong enum cố định 10 slug: `bim | mep | marketing | phap_ly | san_xuat | cntt | nhan_su | tai_chinh | kinh_doanh | thiet_ke`. Anthropic tool use ép LLM không hallucinate label ngoài list. Persona VN có dấu (vd `"công nghệ thông tin"`) cũng được accept qua `PERSONA_TO_DOMAIN` backward compat.

### `POST /api/ingest/file` — Nạp tài liệu

`Content-Type: multipart/form-data`

| Field | Bắt buộc | Mô tả |
|-------|----------|-------|
| `file` | ✓ | PDF / DOCX / DOC / TXT / MD / XLSX |
| `domain` | ✓ | Slug 1 trong 10 lĩnh vực — route đúng `tdi_docs_{slug}` |
| `title`, `description`, `tags`, `url` | | Metadata (tuỳ chọn — AI auto-gen nếu trống) |

```bash
curl -X POST http://localhost:8000/api/ingest/file \
  -F "file=@report.pdf" \
  -F "title=Báo cáo Q1" \
  -F "domain=marketing" \
  -F "tags=2026,Q1"
```

Response:

```json
{"status": "ok", "chunks_added": 42, "message": "Nạp thành công 'report.pdf': 42 đoạn từ 12 trang."}
```

### `POST /api/ingest/video/file` — Nạp video local

File: MP4, MKV, AVI, MOV, WEBM, FLV, WMV. Yêu cầu `openai-whisper` (cài thủ công — xem `requirements.txt`).

### `POST /api/ingest/youtube` — Nạp URL YouTube (tự nhận playlist)

```bash
curl -X POST "http://localhost:8000/api/ingest/youtube?url=https://www.youtube.com/watch?v=XXX"
curl -X POST "http://localhost:8000/api/ingest/youtube?url=https://www.youtube.com/playlist?list=YYY"
```

### `POST /api/ingest/youtube-playlist` — Nạp playlist (chi tiết từng video)

Mỗi chunk con được gắn thêm nhóm field `playlist_*` (id, title, description, uploader, thumbnail, url) lấy từ `yt-dlp --flat-playlist -J` — không LLM, không tốn chi phí. Dùng để filter/group ở `knowledge.html` hoặc query RAG theo playlist.

```json
{
  "status": "ok",
  "message": "Playlist 'Khoá BIM cơ bản': 8/10 video thành công, tổng 312 đoạn.",
  "playlist_info": {
    "playlist_id": "PLxxxxxxxx",
    "playlist_title": "Khoá BIM cơ bản",
    "playlist_description": "...",
    "playlist_uploader": "Tên kênh",
    "playlist_thumbnail": "https://...",
    "playlist_url": "https://www.youtube.com/playlist?list=PLxxxxxxxx",
    "video_count": 10
  },
  "total_videos": 10,
  "success_count": 8,
  "total_chunks": 312,
  "results": [{"video_id": "...", "status": "ok", "chunks_added": 45}]
}
```

`POST /api/ingest/youtube/preview` với URL playlist trả `status: ok` cùng block `metadata.is_playlist=true` + các field `playlist_*` để FE prefill card review trước khi submit.

### `POST /api/chat/` — Hỏi đáp

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tóm tắt kế hoạch truyền thông nội bộ",
    "session_id": "uuid-v4",
    "user_id": "u-001",
    "domain": "marketing",
    "history": []
  }'
```

`user_id` định danh người dùng cho conversation memory cross-session. Nếu bỏ trống, hệ thống dùng `session_id` làm fallback (memory không chia sẻ giữa các session).

`domain` hỗ trợ 10 slug: `bim`, `mep`, `marketing`, `phap_ly`, `san_xuat`, `cntt`, `nhan_su`, `tai_chinh`, `kinh_doanh`, `thiet_ke`. Bỏ trống / null → chat chung (fanout 20 collection).

Response:

```json
{
  "answer": "Kế hoạch gồm 15 hoạt động...\n\nNguồn:\n- KH truyền thông nội bộ 2026 (trang 3)",
  "sources": [
    {"index": 1, "source_type": "document", "title": "KH truyền thông nội bộ 2026",
     "url": "...", "page": 3, "score": 0.89, "positions": [{"page": 3}]}
  ],
  "session_id": "uuid-v4",
  "suggested_questions": [
    "Timeline cụ thể từng tháng ra sao?",
    "Có bao nhiêu sự kiện nội bộ trong Q2?",
    "Ngân sách dự kiến cho từng hoạt động?"
  ]
}
```

**Ví dụ JS:**

```js
const sessionId = localStorage.getItem('sid') ?? crypto.randomUUID();
localStorage.setItem('sid', sessionId);

const res = await fetch('/api/chat/', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message: 'câu hỏi',
    session_id: sessionId,
    user_id: 'u-001',
    domain: 'mặc định',
    history: [],
  }),
});
const {answer, sources, suggested_questions} = await res.json();
```

### `POST /api/chat/stream` — Hỏi đáp (SSE streaming)

Cùng schema `ChatRequest` như `/api/chat/`, nhưng response là `text/event-stream`. Mỗi event theo format `data: <json>\n\n` với `type` thuộc một trong:

| Event | Khi nào | Payload |
|-------|---------|---------|
| `meta` | Ngay sau khi retrieval + rerank xong | `confidence`, `rewritten_query` (nếu khác query gốc), `recall_count` |
| `delta` | Mỗi token Claude sinh ra | `text` — chunk text cộng dồn phía client |
| `done` | Kết thúc generation | `answer` (đã tách `---GỢI Ý---`), `sources` (nếu bot trích "Nguồn:"), `suggested_questions` |
| `error` | Chain raise exception | `message` |

Background memory update (`session_memory` + `ttt_memory` upsert) chạy sau khi stream đóng, dùng `answer` clean từ event `done`.

**Ví dụ JS:**

```js
const res = await fetch('/api/chat/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({message, session_id, user_id, domain, history: []}),
});
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = '';
while (true) {
  const {value, done} = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, {stream: true});
  let idx;
  while ((idx = buffer.indexOf('\n\n')) >= 0) {
    const raw = buffer.slice(0, idx);
    buffer = buffer.slice(idx + 2);
    for (const line of raw.split('\n')) {
      if (line.startsWith('data:')) {
        const evt = JSON.parse(line.slice(5).trim());
        if (evt.type === 'delta') appendToken(evt.text);
        else if (evt.type === 'done') renderFinal(evt.answer, evt.sources, evt.suggested_questions);
      }
    }
  }
}
```

Nginx proxy lưu ý: header `X-Accel-Buffering: no` đã được gắn, nhưng nếu dùng proxy khác cần tắt buffering/compression cho route này.

### `DELETE /api/chat/memory/user/{user_id}` — Xoá toàn bộ conv memory của 1 user

Dọn sạch mọi pair của user trong Qdrant `ttt_memory`. Dùng cho GDPR / reset test data.

### `DELETE /api/chat/memory/session/{session_id}` — Xoá 1 session

Xoá file session JSON + tất cả pair cùng `session_id` trong Qdrant.

## Conversation Memory (Hybrid 3 tầng)

Mỗi lượt chat đi qua 3 tầng memory, mỗi tầng bắt 1 scope khác nhau:

| Tầng | Lưu ở đâu | Scope | Chứa gì |
|------|-----------|-------|---------|
| 1 — Sliding window | File JSON `data/sessions/{sid}.json` | Trong cùng session | 3 pair gần nhất (config `CONV_WINDOW_TURNS`) |
| 2 — Rolling summary | Cùng file JSON, field `summary` | Trong cùng session | Haiku tóm tắt các pair đã rớt khỏi window |
| 3 — Vector recall | Qdrant `ttt_memory` | Cross-session theo `user_id` | Mọi pair (user+bot) đã upsert, search bằng Voyage embed |

Khi có query mới:

1. Lấy sliding window + summary của session hiện tại.
2. Song song: RAG doc retrieval + Qdrant recall pair cùng `user_id` (score ≥ `CONV_RECALL_MIN_SCORE`).
3. Build prompt XML: `<retrieved_documents>` + `<session_summary>` + `<user_context>` (pairs).
4. Sau khi trả lời: upsert pair mới vào Qdrant, pop overflow + update summary.

### Guards chống bloat Qdrant (4 lớp)

Mỗi turn, trước khi upsert pair vào `ttt_memory`, hệ thống chạy 4 lớp guard liên tiếp để tránh phình storage và nhiễu recall:

| # | Guard | Module | Chặn cái gì | Chi phí |
|---|-------|--------|-------------|---------|
| 0 | **No-info filter** | `app/api/chat.py::_is_no_info_answer` | Pair mà bot trả "không tìm thấy / không biết" → tránh feedback loop | 0 (regex in-RAM) |
| 1 | **Heuristic filter** | `conv_memory._is_worth_storing` | Câu xã giao (length < 20/40, regex "xin chào/cảm ơn/ok/vâng...", density thấp) | 0 (regex in-RAM) |
| 2 | **Hash dup LRU** | `conv_memory._hash_seen` | Exact duplicate theo MD5 pair đã normalize, cache per-user `CONV_HASH_CACHE_SIZE` | 0 (trước cả embed) |
| 3 | **Semantic dedup** | `conv_memory._find_near_duplicate` | Pair cùng `user_id` có cosine ≥ `CONV_DEDUP_THRESHOLD` (0.92) → chỉ update `last_seen_at`, không ghi point mới | +1 Qdrant search (~5ms) |

Tổng hiệu ứng thực tế: loại ~60-80% turn rác (greetings, ack, câu lặp) mà không tốn thêm LLM call. Chi tiết threshold dựa trên research Mem0 + Zep + EMem (xem `docs/PROJECT_OVERVIEW.md`).

### Test nhanh

```bash
source venv/bin/activate

# Xem collection ttt_memory
python scripts/check_memory_collection.py

# Diag: count theo user + test retrieve
python scripts/diag_conv_memory.py

# Seed 2 user giả
python scripts/seed_two_users.py

# Dọn pair "không tìm thấy" đã ô nhiễm
python scripts/clean_poisoned_pairs.py --apply
```

Trên UI `chat.html`: dropdown "User đang test" cho phép switch giữa `test-user-1` / `test-user-2`. Mỗi user có sessions riêng lưu ở localStorage (`ttt_sessions_<userId>`).

---

## Hybrid Retrieval (Anthropic Contextual Retrieval recipe)

Pipeline retrieval gồm **3 component bổ sung** cho dense embedding truyền thống:

| Component | Module | Vai trò |
|---|---|---|
| **Contextual chunking** | `app/core/contextual_chunker.py` | Mỗi chunk → Haiku sinh 50-100 token mô tả ngữ cảnh tài liệu, prepend vào trước embed. Prompt caching ephemeral → cost cache hit ~85% |
| **BM25 sparse encoder** | `app/core/sparse_encoder.py` | underthesea word_tokenize (VN, 80% accuracy) → split unigram → FNV-1a 64-bit hash → SparseVector{indices, values=TF}. IDF do Qdrant tự nhân (`modifier="idf"`) |
| **Qdrant hybrid Query API** | `app/core/qdrant_store.py` | `/points/query` với prefetch[dense, sparse] + RRF fusion server-side, k=60. 1 HTTP call/store |

**Kết quả test E2E (10 nhóm stress test, 24 query)**: PASS 10/10. Δ sigmoid giữa các biến thể cùng intent < 0.05 (trừ synonym English-Vietnamese acronym 0.188 — corpus terminology limitation, vẫn pass `_HARD_REFUSE_SCORE=0.4`).

**Test scripts:**

```bash
./venv/bin/python scripts/test_hybrid_e2e.py        # E2E ingest + 4 nhóm
./venv/bin/python scripts/test_hybrid_advanced.py   # 10 stress test groups
```

Chi tiết kiến trúc + benchmark: `docs/PROJECT_OVERVIEW.md` section "Hybrid Retrieval".

---

## Prompt Engineering

`app/rag/prompt_builder.py` áp dụng best practices Anthropic cho Claude 4+:

- **XML tags** (`<retrieved_documents>`, `<user_context>`, `<session_summary>`) thay cho markdown headers → parse boundary chắc chắn.
- **Strict grounding** — cấm Claude dùng training data cho định nghĩa/khái niệm/best practice ngoài tài liệu. Chỉ cho tính toán số học thuần trên số user/tài liệu đã cung cấp.
- **`<refusal_protocol>`** — template cứng khi không có context liên quan: bot trả đúng 1 câu refusal, không bịa "bù" bằng kiến thức chung.
- **Pre-LLM guard** (`chain.py`) — top rerank sigmoid < `_HARD_REFUSE_SCORE` (0.4) hoặc không có hit → trả refusal cứng, **không gọi Claude** (tiết kiệm $ + chặn hallucination ở gốc).
- **Quote-first grounding** — yêu cầu Claude ngầm xác định đoạn tài liệu liên quan trước khi tổng hợp.
- **Vietnamese tone** — xưng "tôi", gọi "bạn", giữ thuật ngữ EN, format số theo VN (`1.000.000 đồng`, `13,3 triệu`).
- **Follow-up suggestion** — rút trực tiếp từ `<retrieved_documents>` vừa dùng, không generic theo domain.

## Performance — Reranker GPU

Cross-encoder `BAAI/bge-reranker-v2-m3` (~560M params) chạy local để rerank hit Qdrant. Auto-detect device theo thứ tự `cuda` > `mps` > `cpu`:

| Môi trường | Device | Latency (5 pairs, steady-state) |
|------------|--------|---------------------------------|
| Mac Apple Silicon M1-M4 | `mps` (Metal) | ~35ms |
| Cloud NVIDIA GPU (T4/A10G/...) | `cuda` | ~20-40ms |
| Cloud CPU thường | `cpu` | ~1-2s |

Warmup eager-load model lúc FastAPI startup (qua `lifespan` event) → request chat đầu tiên sau restart không còn gánh 3-5s cold load. Override bằng env `RERANKER_DEVICE=cpu|cuda|mps`.

Ref: [Anthropic XML tags guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags) · [Claude 4 best practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)

---

## Chi phí tham khảo

| Thành phần | Khi nào | Chi phí |
|------------|---------|---------|
| Docling | Luôn chạy | $0 (local) |
| Claude Vision | Docling fail / bảng có màu | ~$0.002/trang |
| LLM mô tả bảng | Bảng có dữ liệu | ~$0.001/bảng |
| **Contextual chunking** | Mỗi chunk khi `CONTEXTUAL_CHUNKING=1` | ~$0.0015/chunk (cache hit) |
| Voyage embed | Luôn chạy | ~$0.0001/chunk |
| BM25 sparse encode | Luôn chạy khi `HYBRID_RETRIEVAL=1` | $0 (local underthesea) |
| Claude Sonnet 4 answer | Mỗi câu trả lời | ~$0.003-0.015/lần |
| **File text thuần (50 chunks)** | Docling OK + contextual + sparse | **~$0.08/file** |
| **File phức tạp + Vision** | Vision + LLM + contextual + sparse | **~$0.10-0.15/file** |

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `401` từ Qdrant | Kiểm tra `QDRANT_API_KEY` |
| `400 "Not existing vector name: dense"` từ Qdrant | Collection có schema CŨ → chạy `scripts/reset_qdrant_collections.py` 1 lần |
| `ModuleNotFoundError: underthesea` | `pip install -r requirements.txt` (lib mới cho BM25 sparse VN) |
| Search trả 0 hits dù đã upload | Check `HYBRID_RETRIEVAL` env, schema collection (xem section "Migration Qdrant schema"), domain match metadata |
| Anthropic cost cao bất thường khi ingest | Tắt `CONTEXTUAL_CHUNKING=0` (mỗi chunk = 1 Haiku call) |
| `poppler not found` | `brew install poppler` / `apt install poppler-utils` |
| `ffmpeg not found` | `brew install ffmpeg` (chỉ cần khi ingest video local) |
| YouTube `IP blocked` | Cấu hình `YOUTUBE_PROXY_LIST` hoặc Webshare |
| `openai-whisper not installed` | Bật dòng `openai-whisper` trong `requirements.txt` và `pip install` lại |
| `collection not found` | Nạp ít nhất 1 tài liệu để tạo collection |
| Server khởi động chậm lần đầu | Docling + underthesea tải model — chờ 1-2 phút |

## Tài liệu chi tiết

- `docs/PROJECT_OVERVIEW.md` — kiến trúc, pipeline, quy tắc xử lý bảng
- `docs/INTEGRATION.md` — hướng dẫn tích hợp FE/BE
