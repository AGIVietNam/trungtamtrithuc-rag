# Trung Tâm Tri Thức — Tổng Quan Dự Án

## Giới thiệu

**Trung Tâm Tri Thức** (Knowledge Center) là hệ thống **RAG (Retrieval-Augmented Generation)** xây dựng trên FastAPI, phục vụ hỏi đáp tri thức doanh nghiệp bằng tiếng Việt với trích dẫn nguồn.

Tính năng chính:

1. **Nạp tài liệu** đa định dạng (PDF, DOCX, XLSX, TXT, MD) qua pipeline 3 tier.
2. **Nạp video** (YouTube URL, YouTube Playlist, file MP4/MKV/AVI/MOV) — phiên âm + embed theo timestamp.
3. **AI auto-metadata** khi upload: tự sinh `title`, `description`, `domain` (7 label cố định), `tags` — Haiku tool use + Pydantic schema, FE prefill form để user review.
4. **Hỏi đáp** với chuyên gia theo domain (BIM, MEP, kết cấu, marketing, pháp lý, sản xuất, hoặc tự do).
5. **Gợi ý câu hỏi tiếp theo** tự động sau mỗi câu trả lời.
6. **Session history** file-backed — nhớ hội thoại trong phiên để tiếp tục ngữ cảnh.

Bộ công nghệ lõi: **Claude Sonnet 4** (trả lời) + **Claude Haiku 4.5** (Vision + describe table), **Voyage AI** (`voyage-3`, 1024-dim) cho embedding, **Qdrant Cloud** làm vector store.

---

## Kiến trúc tổng thể

```
                    +-------------------------+
                    |   Static Frontend        |
                    |   (HTML/CSS/JS)          |
                    |   index / chat           |
                    |   ingest / knowledge     |
                    +------------+-------------+
                                 |
                            HTTP REST
                                 |
                    +------------v-------------+
                    |   FastAPI Server         |
                    |   (app/main.py)          |
                    +------------+-------------+
                                 |
            +--------------------+---------------------+
            |                                          |
  +---------v---------+                    +-----------v-----------+
  |  /api/chat        |                    |  /api/ingest          |
  |   (RAG answer)    |                    |   file / video / yt   |
  +---------+---------+                    +-----------+-----------+
            |                                          |
  +---------v---------+                    +-----------v-----------+
  |   RAG Chain       |                    |  Ingestion Pipelines  |
  |  - Retriever      |                    |  - doc_parser (3-tier)|
  |  - Reranker       |                    |  - doc_pipeline       |
  |  - Prompt builder |                    |    (table + Vision    |
  |  - Claude Sonnet4 |                    |     with context)     |
  |  - Suggestions    |                    |  - video_pipeline     |
  +---------+---------+                    +-----------+-----------+
            |                                          |
  +---------v------------------------------------------v----------+
  |                    Qdrant Vector Database                    |
  |  +----------------+  +----------------+                      |
  |  | ttt_documents  |  | ttt_videos     |                      |
  |  +----------------+  +----------------+                      |
  |                                                              |
  |  +---------------------------------------------------------+ |
  |  | vmedia_* (READ ONLY — cluster riêng)                    | |
  |  | content, design, digital, documents, fonts, image,      | |
  |  | media, qa, ttnb                                         | |
  |  +---------------------------------------------------------+ |
  +--------------------------------------------------------------+
```

---

## Pipeline xử lý tài liệu

### Tổng quan flow

```
FILE UPLOAD
    │
    ▼
[PARSE] app/ingestion/doc_parser.py — 3 tier:
    │
    ├─ Tier 1: Docling (local, miễn phí)
    │    OCR + layout + table → Markdown
    │    Quality check 4 điều kiện:
    │      • broken table (>20 pipes/1-2 dòng)
    │      • repeated header (≥5 lần)
    │      • no line breaks (>500 chars/<3 dòng)
    │      • duplicate blocks (>40% paragraph trùng)
    │    ├─ OK  → Markdown
    │    └─ FAIL → Tier 2
    │
    ├─ Tier 2: Claude Haiku Vision (~$0.002/trang)
    │    PDF → pdf2image → base64 → Vision
    │    Hiểu: bảng phức tạp, ô màu, merged cells
    │    ├─ OK  → structured text
    │    └─ FAIL → Tier 3
    │
    └─ Tier 3: pdfplumber (miễn phí)
         Text thuần, không cấu trúc — last resort
    │
    ▼
[PROCESS] app/ingestion/doc_pipeline.py — xử lý từng bảng riêng:
    │
    ├─ Không có bảng → strip ## ** → giữ plain text
    │
    ├─ Bảng có dữ liệu (<50% ô trống)
    │    → LLM Haiku tóm tắt 2-3 câu (cho search)
    │    → Giữ bảng gốc Markdown (cho LLM đọc chính xác)
    │    → payload: text = tóm tắt, table_data = bảng Markdown
    │
    └─ Bảng có nhiều ô trống (>50%, biểu thị bằng màu)
         → Vision với context từ các trang khác
         → Output: structured plain text
         → payload: text = structured text
    │
    ▼
[TYPO FIX] _fix_common_typos (Vietnamese OCR errors)
    │
    ▼
[CHUNK] app/core/chunker.py — heading-aware:
    Chia theo heading (##) → paragraph → sentence
    Max 700 tokens, overlap 80 tokens
    Lọc chunk < 10 tokens
    Tokenizer: tiktoken cl100k_base
    │
    ▼
[EMBED] Voyage AI (voyage-3) → vector 1024-dim
    │
    ▼
[STORE] Qdrant payload:
    text: plain text (cho search + LLM đọc)
    table_data: bảng gốc Markdown (chỉ khi có bảng dữ liệu)
    heading_path: ["Chương", "Mục"]
    source_name, page, doc_id, uploaded_at
    domain, title, description, tags, url (nếu có từ form)
```

### Ma trận xử lý theo dạng tài liệu

| Dạng | Parse | Process | `text` | `table_data` |
|------|-------|---------|--------|--------------|
| Văn bản thuần (báo cáo, tài liệu thường) | Docling → Markdown | Strip formatting | Plain text | (trống) |
| Bảng có dữ liệu (nhân sự, kênh, chỉ số) | Docling → Markdown table | LLM Haiku tóm tắt 2-3 câu | Tóm tắt ngắn | Bảng Markdown gốc |
| Bảng có màu / ô trống (timeline, Gantt) | Docling (bỏ) → Vision+context | Vision plain text | Structured text | (trống) |
| Docling fail (Excel merged, bảng vỡ) | Vision per-page | Giữ nguyên | Structured text | (trống) |
| XLSX | openpyxl → structured text | Giữ nguyên | Structured text | (trống) |
| DOCX | Docling / python-docx | Như PDF | Như PDF | Như PDF |
| TXT / MD | Đọc trực tiếp | Strip formatting | Plain text | (trống) |

### Vì sao có 2 field (`text` + `table_data`)?

```
User hỏi: "Khối Văn phòng có bao nhiêu nhân sự?"

1) SEARCH: embed(query) match với `text` = "Bảng phân nhóm 3 nhóm:
   VP 30%, BCHCT 40%, CN 30%"  → tìm đúng chunk (plain text search tốt)

2) LLM TRẢ LỜI: Claude đọc cả `text` + `table_data`
   text:       "Bảng phân nhóm gồm 3 nhóm: VP 30%, BCHCT 40%, CN 30%"
   table_data: "| Nhóm | Tỷ trọng | Chi tiết |
                |---|---|---|
                | VP | ~30% | ~200 nhân sự |"
   → Trả lời chính xác: "Khối Văn phòng có ~200 nhân sự"
```

### Vision với Context

Khi Vision đọc một trang có bảng màu, nó nhận **context từ các trang khác** (do Docling đã parse) để hiểu trang này thuộc phần nào của tài liệu:

```
Vision nhận:
  Context: "Kế hoạch truyền thông nội bộ 2026 của TDI.
            Mục tiêu: tăng tham gia nội bộ, xây dựng văn hoá..."
  + Hình: trang timeline với các ô màu xanh/cam

Vision trả về:
  "Timeline truyền thông nội bộ 2026 của TDI gồm 15 hoạt động:
   Tất niên, du xuân (T1-T2): đã hoàn thành.
   30/4 - 1/5 Hoạt động nội bộ (T4): sự kiện quan trọng.
   ..."
```

Màu sắc được diễn giải: xanh = hoàn thành, cam = đang làm, đỏ = quan trọng.

### Vietnamese Typo Auto-Fix

Vision/OCR thường sai dấu tiếng Việt. Hệ thống có bảng `_TYPO_FIXES` trong `app/ingestion/doc_parser.py`:

| Sai | Đúng |
|-----|------|
| Chông Cháy / chông chảy | Chống Cháy / chống cháy |
| Của chống | Cửa chống |
| Dã quay / Dã dạng | Đã quay / Đã đăng |
| Dạng edit | Đang edit |
| Tính trạng | Tình trạng |
| KỀ HOẠCH | KẾ HOẠCH |
| ban nhạt định | bạn nhất định |

Thêm lỗi mới: chỉnh sửa list `_TYPO_FIXES` và reload.

---

## Pipeline xử lý video

```
YouTube URL / Playlist URL / File MP4
    │
    ▼
[TRANSCRIBE]
    ├─ YouTube: youtube-transcript-api (proxy rotation)
    │    → segments [{start, text}, ...]
    │    Retry tối đa 10 lần, xoay proxy mỗi lần
    │
    └─ File local: Whisper (openai-whisper)
         → segments [{start, text}, ...]
    │
    ▼
[CHUNK] chunk_transcript_with_timestamps (max 500 tokens/chunk)
    Mỗi chunk giữ start_sec đầu đoạn → deep-link YouTube
    │
    ▼
[EMBED] Voyage AI
    │
    ▼
[STORE] Qdrant ttt_videos:
    text, video_id, title, start_sec, file_source (youtube|local),
    url (https://www.youtube.com/watch?v=XXX&t=YYs)
```

Playlist: lấy danh sách video → ingest tuần tự → trả về tổng hợp
`total_videos / success_count / total_chunks`.

---

## AI Auto-Metadata (preview before ingest)

Khi user upload, FE gọi endpoint **preview** để AI sinh metadata trước khi commit ingest. User review + sửa → submit endpoint chính với metadata đã duyệt.

### 3 endpoint preview

| Endpoint | Input | Cơ chế | Sinh field gì |
|----------|-------|--------|---------------|
| `POST /api/ingest/file/preview` | multipart file | Docling parse → lấy 5 trang đầu (~3000 token) → Haiku tool use | title, description, domain, tags |
| `POST /api/ingest/video/file/preview` | multipart file video | `get_transcriber()` (Groq nếu có, fallback Whisper local) → ghép segment text → Haiku tool use | title, description, domain, tags |
| `POST /api/ingest/youtube/preview?url=…` | query url | `yt-dlp --dump-single-json` (title/description/thumbnail/channel/duration) + transcript best-effort → Haiku chỉ classify domain/tags | title+description (từ YouTube) + domain+tags (từ AI) |

**Playlist URL** → `/youtube/preview` trả `status=skip` (mỗi video playlist có metadata khác nhau, không gen cho cả list).

### Structured output — Anthropic tool use + Pydantic

`app/ingestion/metadata_generator.py::generate_document_metadata()` dùng **tool use** ép Claude gọi virtual tool `save_document_metadata` với JSON schema:

```python
{
  "title":       {"type": "string", "minLength": 3, "maxLength": 200},
  "description": {"type": "string", "minLength": 10, "maxLength": 500},
  "domain":      {"type": "string", "enum": [
                   "bim", "mep", "kết cấu", "marketing",
                   "pháp lý", "sản xuất", "mặc định"]},
  "tags":        {"type": "array", "items": {"type": "string"},
                  "minItems": 0, "maxItems": 10}
}
```

Domain là `Literal` enum trong Pydantic → LLM **không thể sinh label ngoài 7 giá trị** (constrained decoding ở API level). Tags được normalize sau (lowercase, dedup, strip punctuation, cap 8).

### Filename & heading hint (doc)

Trước khi gọi LLM, helper:

- `_clean_filename_hint()` strip `Copy of`, `_v2`, `_final`, `_draft` → `"Báo cáo Q1"` thay vì `"Copy of Báo_cáo_Q1_final_v2.pdf"`
- `_extract_first_heading()` bắt `# Heading` Markdown đầu tiên từ Docling output

Cả hai gắn vào prompt dưới XML tag `<filename>` `<heading>` để hint cho LLM, không ép buộc.

### UX frontend (web/ingest.html)

- 2 loại badge: **✦ AI** (tím — source AI Haiku) và **▶ YT** (đỏ — source YouTube metadata)
- Field AI-filled có class `ai-filled` (bg tím nhạt), YT-filled có `yt-filled` (bg hồng nhạt)
- User chỉnh tay field nào → class + badge tự clear (→ user-confirmed)
- Đổi file/URL → reset chỉ field AI-filled/YT-filled (giữ field user nhập tay)
- Tab YouTube: hiển thị thumbnail + channel + duration dưới URL

### Chi phí & latency

| Tab | Input | LLM call | Chi phí | Latency user chờ |
|-----|-------|----------|---------|------------------|
| Doc | ~3000 token text | 1 Haiku | ~$0.004/file | ~2-5s |
| Video | ~3000 token transcript | 1 Haiku (sau Whisper) | ~$0.005 + Whisper cost | Groq ~10s, Whisper local 30s-2 phút |
| YouTube | title+desc+transcript | 1 Haiku (sau yt-dlp) | ~$0.004/URL | ~1-3s |

### Respect user input

Trong toàn bộ flow: field nào user **đã nhập tay** (không có class `ai-filled`/`yt-filled`) sẽ **không bị override** khi preview trả kết quả. Chỉ prefill field đang trống.

---

## RAG Pipeline (hỏi đáp)

### Sơ đồ

```
User query
    │
    ▼
[EMBED] Voyage embed_query
    │
    ▼
[RETRIEVE] Parallel search (ThreadPoolExecutor, 3 nguồn)
    │   Qdrant filter: domain = X OR is_empty(domain)
    │
    ├─ ttt_documents
    ├─ ttt_videos
    └─ vmedia_* (nếu có QDRANT_VMEDIA_API_KEY)
    │
    ▼
[DEDUP] theo prefix text (80 ký tự) + sort theo score
    │
    ▼
[RERANK] CrossEncoderReranker (sentence-transformers)
    Trả về top-K (mặc định 3-5)
    │
    ▼
[CONTEXT BUILD] prompt_builder.build_context_block
    XML: <retrieved_documents>
           <document index="1">
             <source>title — url — trang/timestamp</source>
             <content>chunk text + table_data (nếu có)</content>
           </document>
           ...
         </retrieved_documents>
    Source mapping (dedup theo title+url, merge positions)
    │
    ▼
[CONV BUILD] prompt_builder.build_conversation_block(summary, recall_pairs)
    XML: <session_summary>...</session_summary>
         <user_context>
           [#1 — 2 ngày trước]
           USER: ... | BOT: ...
           [#2 ...]
         </user_context>
    │
    ▼
[PROMPT] system = DOMAIN_PERSONAS[domain] + _BASE_RULES + conv_block
    DOMAIN_PERSONAS: bim | mep | kết cấu | marketing | pháp lý | sản xuất | mặc định
    _BASE_RULES (positive framing, XML sections):
      <language_style>   tiếng Việt, xưng "tôi"/gọi "bạn", format số VN
      <reasoning_process> quote-first grounding (ngầm)
      <grounding_rules>  cho phép suy luận tài liệu + dữ kiện user
      <citation_rules>   trích TÊN, mục "Nguồn:" hoặc bỏ nếu xã giao
      <followup_suggestions> 3 câu rút từ <retrieved_documents>
    │
    ▼
[LLM] Claude Sonnet 4
    messages = history + [{role: user, content: query}]
    │
    ▼
[PARSE]
    - Split tại "---GỢI Ý---" → (clean_answer, suggested_questions[3])
    - Ẩn sources nếu user chỉ chào hỏi (không có "Nguồn:" trong answer)
    - Confidence = high/medium/low theo top score
```

### Domain chuyên gia

| Domain | Phong cách |
|--------|-----------|
| `mặc định` | Trợ lý tri thức chung |
| `bim` | Chuyên gia BIM — LOD, clash, Revit/Navisworks, IFC, BEP |
| `mep` | Kỹ sư MEP — HVAC, PCCC, sprinkler, ELV, BMS, TCVN/ASHRAE/NFPA |
| `kết cấu` | Structural — BTCT, thép, móng, ETABS/SAP, TCVN/Eurocode/ACI |
| `marketing` | Brand, 4P/7P, digital, funnel, KPI, ROI, A/B test |
| `pháp lý` | Luật Xây dựng/Đầu tư/Doanh nghiệp/Lao động/Dân sự — trích Điều/Khoản |
| `sản xuất` | Lean, 5S, Kaizen, OEE, Six Sigma, SOP, BOM, MRP |

Domain khác → prompt tổng quát: `"Bạn là chuyên gia về '<domain>'..."`.

### Suggested Questions

Mỗi câu trả lời kèm 3 câu hỏi gợi ý ở cuối, bắt đầu bằng `---GỢI Ý---`:

```
1. Timeline cụ thể từng tháng ra sao?
2. Có bao nhiêu sự kiện nội bộ trong Q2?
3. Ngân sách dự kiến?
```

Parser regex tách ra trường `suggested_questions[]` trong response JSON.

---

## Conversation Memory (Hybrid 3 tầng)

Memory của bot được chia 3 tầng, mỗi tầng bắt 1 scope khác nhau:

| Tầng | Module | Lưu ở | Scope | Role |
|------|--------|-------|-------|------|
| 1 — Sliding window | `app/core/session_memory.py` | File `data/sessions/{sid}.json` | Cùng session | 3 pair gần nhất (direct context) |
| 2 — Rolling summary | `app/core/conv_summarizer.py` | Cùng file, field `summary` | Cùng session | Haiku tóm tắt các pair đã rớt khỏi window |
| 3 — Vector recall | `app/core/conv_memory.py` | Qdrant `ttt_memory` | Cross-session theo `user_id` | Mọi pair đã upsert, search bằng Voyage |

### Flow mỗi lượt chat

```
POST /api/chat/ {message, session_id, user_id, domain}
  │
  ├─► session_memory.get_history(sid)      # tầng 1
  ├─► session_memory.get_summary(sid)      # tầng 2
  │
  ├─► RAGChain.answer():
  │     ├─ conv_query_rewriter.rewrite()   # giải đại từ "nó / cái đó / ngân sách đó"
  │     ├─ parallel:
  │     │   ├─ retriever.retrieve()        # doc RAG (ttt_documents + ttt_videos + vmedia)
  │     │   └─ conv_memory.retrieve()      # tầng 3 — filter user_id, same-session exclude
  │     ├─ reranker.rerank()
  │     ├─ build_system_prompt(domain)     # XML: rules + language_style + reasoning
  │     ├─ build_conversation_block()      # XML: <session_summary> + <user_context>
  │     ├─ build_context_block()           # XML: <retrieved_documents>
  │     └─ claude.generate()
  │
  └─► background task:
        ├─ session_memory.add_turn()
        ├─ pop_overflow → conv_summarizer.summarize → set_summary
        └─ conv_memory.upsert_pair() — SKIP nếu bot trả "không tìm thấy"
                                       (tránh feedback loop)
```

### Guards chống bloat + feedback loop

Mỗi turn, trước khi upsert pair vào `ttt_memory`, pipeline chạy 4 lớp guard liên tiếp. Pair chỉ thực sự được ghi Qdrant khi pass CẢ 4.

| # | Guard | Ở đâu | Chặn cái gì | Chi phí |
|---|-------|-------|-------------|---------|
| 0 | **No-info filter** | `app/api/chat.py::_is_no_info_answer` | Bot trả "không tìm thấy / tôi không biết / không có trong cơ sở tri thức" → skip upsert. Tránh feedback loop: hỏi tương tự về sau → recall lại câu "không biết" cũ → tự khẳng định lại "không biết" thay vì dùng fact thật. | 0 (regex) |
| 1 | **Heuristic filter** | `conv_memory._is_worth_storing` | 3 sublayer tuần tự: (A) length gate — `len(user) < 20` hoặc `len(bot) < 40 AND không có "Nguồn:"`; (B) regex câu xã giao VN+EN với anchor `\W*$` (`xin chào, cảm ơn, ok, vâng, dạ, tạm biệt, tuyệt, hay quá...`); (C) density — câu < 6 từ + không có số + không có từ > 5 ký tự. | 0 (regex) |
| 2 | **Hash dup LRU** | `conv_memory._hash_seen` | MD5 của pair đã normalize (lowercase + collapse whitespace + strip punctuation) — LRU `OrderedDict` per-user, size `CONV_HASH_CACHE_SIZE=2000`. Trùng exact → skip **trước cả embed**. | 0 |
| 3 | **Semantic dedup** | `conv_memory._find_near_duplicate` | Embed pair (1 lần, reuse cho upsert), search Qdrant filter `user_id` với `score_threshold=CONV_DEDUP_THRESHOLD` (0.92). Có hit → gọi `_touch_last_seen()` update payload `last_seen_at` của pair cũ thay vì insert point mới. Giữ cluster collection compact, biết pair nào "hot". | +1 Qdrant search (~5ms) |

**Same-session filter** (read side, không liên quan upsert): `conv_memory.retrieve()` loại pair cùng `session_id` hiện tại — vì đã có trong L1 sliding window + L2 summary, recall lại sẽ trùng context.

Threshold chọn dựa trên: Mem0 paper dùng 0.95 cho entity merge, EMem paper dùng 0.90 cho synonym edge. 0.92 = trung dung cho Voyage-3 1024-dim. Log mỗi lần skip để audit volume ("conv_memory skip upsert (heuristic/hash/semantic): user=... reason=...").

### Endpoints quản lý

| Endpoint | Hành vi |
|----------|---------|
| `DELETE /api/chat/memory/user/{user_id}` | Xoá toàn bộ pair của user trong Qdrant (GDPR / reset test) |
| `DELETE /api/chat/memory/session/{session_id}` | Xoá file JSON + pair cùng session_id trong Qdrant |

---

## Cấu trúc thư mục

```
trungtamtrithuc/
├── app/
│   ├── main.py                  # FastAPI app + CORS + static mount
│   ├── config.py                # Env vars (Voyage, Qdrant, Claude, Proxy)
│   ├── schemas.py               # ChatRequest/Response, Ingest, KnowledgeSearch
│   ├── api/
│   │   ├── chat.py              # POST /api/chat/
│   │   └── ingest.py            # POST /api/ingest/{file,video/file,youtube,youtube-playlist}
│   ├── core/
│   │   ├── chunker.py           # Heading-aware chunking (tiktoken cl100k_base)
│   │   ├── claude_client.py     # Anthropic messages client (sync + stream)
│   │   ├── voyage_embed.py      # Voyage embedder (query vs document)
│   │   ├── qdrant_store.py     # QdrantStore (R/W) + VMediaReadOnlyStore
│   │   ├── session_memory.py    # Sliding window + rolling summary (file JSON)
│   │   ├── conv_memory.py       # Vector recall Qdrant ttt_memory (cross-session)
│   │   ├── conv_summarizer.py   # Haiku summarize rolled turns
│   │   └── conv_query_rewriter.py  # Haiku rewrite đại từ → query độc lập
│   ├── ingestion/
│   │   ├── doc_parser.py        # 3-tier parser + typo fix
│   │   ├── doc_pipeline.py      # Table detect + LLM describe + Vision+context
│   │   ├── video_pipeline.py    # YouTube + local + playlist
│   │   ├── video_transcriber.py # Whisper wrapper
│   │   └── youtube_fetcher.py   # youtube-transcript-api + proxy rotation
│   └── rag/
│       ├── chain.py             # Retrieve → rerank → generate → parse suggestions
│       ├── retriever.py         # Multi-source parallel search
│       ├── reranker.py          # Cross-encoder reranker
│       └── prompt_builder.py    # Domain presets + context + table_data
├── web/                         # Static frontend
│   ├── index.html
│   ├── chat.html
│   ├── ingest.html
│   └── knowledge.html
├── data/{uploads,logs,sessions}/    # Runtime (sessions = file JSON session_memory)
├── scripts/
│   ├── check_memory_collection.py   # Schema + sample payload ttt_memory
│   ├── diag_conv_memory.py          # Count theo user + test retrieve()
│   ├── clean_poisoned_pairs.py      # Xoá pair "không tìm thấy" khỏi Qdrant
│   ├── seed_two_users.py            # Seed 2 test user + conv turns
│   ├── test_conversation_memory.py  # Test 3 tầng hybrid memory
│   └── test_fix_e2e.py              # E2E test cross-session recall
├── docs/
│   ├── PROJECT_OVERVIEW.md      # File này
│   └── INTEGRATION.md           # Tích hợp FE/BE
├── requirements.txt
├── run.sh
└── Dockerfile
```

---

## Cài đặt & chạy

### Yêu cầu

- Python 3.12+
- macOS Apple Silicon (M1-M4) hoặc Linux
- RAM ≥ 16GB (Docling + embedder nạp model)
- `poppler` (pdf2image): `brew install poppler` / `apt install poppler-utils`
- `ffmpeg` (nếu ingest video local với Whisper): `brew install ffmpeg`

### Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install poppler

cp .env.example .env
# Điền: ANTHROPIC_API_KEY, VOYAGE_API_KEY, QDRANT_URL, QDRANT_API_KEY

./run.sh
```

### Truy cập

| URL | Mô tả |
|-----|-------|
| `http://localhost:8000/` | Trang chủ |
| `http://localhost:8000/chat.html` | Chat |
| `http://localhost:8000/ingest.html` | Nạp tài liệu / video |
| `http://localhost:8000/knowledge.html` | Quản lý tri thức |
| `http://localhost:8000/docs` | Swagger API |
| `http://localhost:8000/health` | Healthcheck |

---

## Chi phí vận hành

| Thành phần | Khi nào | Ước tính |
|------------|---------|---------|
| Docling | Luôn chạy | $0 (local) |
| Claude Vision | Docling fail / bảng có màu | ~$0.002/trang |
| LLM describe table | Bảng có dữ liệu | ~$0.001/bảng |
| Voyage embed | Luôn chạy | ~$0.0001/chunk |
| Claude Sonnet 4 answer | Mỗi câu trả lời | ~$0.003-0.015/lần |
| **File text thuần** (Docling OK) | | **~$0.001** |
| **File phức tạp** (Vision + LLM) | | **~$0.005-0.01** |
| **Chat 1 turn** | | **~$0.005-0.02** |

---

## Bảng công nghệ

| Thành phần | Công nghệ |
|------------|-----------|
| Backend | Python 3.12, FastAPI, Uvicorn |
| LLM trả lời | Claude Sonnet 4 (`claude-sonnet-4-20250514`) |
| LLM Vision / Describe table | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) |
| Embedding | Voyage AI `voyage-3` — 1024-dim |
| Vector DB | Qdrant Cloud (2 cluster: main + vmedia read-only) |
| PDF parser | Docling (IBM) → Claude Vision → pdfplumber |
| PDF → image | pdf2image + poppler |
| DOCX | Docling / python-docx |
| XLSX | openpyxl |
| Video transcribe | yt-dlp, youtube-transcript-api, Whisper |
| Reranker | sentence-transformers (cross-encoder) |
| Tokenizer | tiktoken (`cl100k_base`) |
| Frontend | HTML/CSS/JS thuần (static) |

---

## Các dòng chảy dữ liệu chính

### 1. Ingest tài liệu

```
Upload → tempfile → parse (3-tier) → doc_pipeline
  → detect tables → process (strip / describe / vision+context)
  → typo fix → chunk → embed → Qdrant.upsert (ttt_documents)
  → xoá chunks cũ cùng doc_id trước khi upsert mới
```

### 2. Ingest video

```
URL / File → transcribe → segments
  → chunk_transcript_with_timestamps → embed → Qdrant (ttt_videos)
  → payload chứa video_id, start_sec, url deep-link
```

### 3. Chat (RAG answer)

```
POST /api/chat/
  → get history từ session_memory (file-backed)
  → RAGChain.answer(query, history, domain):
      → Retriever parallel 3 nguồn + dedup + sort
      → CrossEncoderReranker → top 3-5
      → build system_prompt (domain preset + base suffix)
      → build context_block (+ table_data)
      → Claude Sonnet 4.generate()
      → split "---GỢI Ý---" → (answer, suggestions[3])
  → memory.add_turn(session_id, user_msg, answer)
```

---

## Hướng phát triển

- **Streaming response** — chain đã có `answer_stream()` (SSE-ready), cần wire `/api/chat/stream`.
- **Auth thật** — hiện `user_id` là string tự do client truyền; tích hợp đăng nhập để verify + chống impersonate memory của user khác.
- **User profile extract** — tách fact cá nhân (tên, team, ngân sách, preference) ra block riêng thay vì lẫn trong recall pairs — tăng độ bền trước khi recall score rơi dưới threshold.
- **Admin CRUD knowledge** — delete/re-index tài liệu từ `knowledge.html`.
- **Evaluation harness** — tập test câu hỏi + ground truth để đo recall/precision.
- **Prompt cache tối ưu** — move `<retrieved_documents>` từ system block sang user turn để cache stable persona cross-query (giảm ~90% cost phần prompt ổn định).
- **Multi-tenant** — tách namespace theo organization.
