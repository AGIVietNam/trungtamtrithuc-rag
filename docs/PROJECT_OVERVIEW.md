# Trung Tâm Tri Thức — Tổng Quan Dự Án

## Giới thiệu

**Trung Tâm Tri Thức** (Knowledge Center) là hệ thống **RAG (Retrieval-Augmented Generation)** xây dựng trên FastAPI, phục vụ hỏi đáp tri thức doanh nghiệp bằng tiếng Việt với trích dẫn nguồn.

Tính năng chính:

1. **Nạp tài liệu** đa định dạng (PDF, DOCX, PPTX, XLSX, TXT, MD) qua pipeline 3 tier.
2. **Nạp video** (YouTube URL, YouTube Playlist, file MP4/MKV/AVI/MOV/WebM/FLV/WMV) — phiên âm + embed theo timestamp.
3. **Async ingest + batch upload** — endpoint `*/async` trả `job_id` ngay, worker pool xử lý nền; endpoint `/files` upload nhiều file 1 request; `/from-urls` nhận list URL có Bearer token (cho BE đẩy file từ S3/SharePoint sau này). Polling tiến độ qua `GET /jobs/{id}` và `/batches/{id}`. Streaming upload 1MB/chunk → file 5GB không OOM.
4. **AI auto-metadata** khi upload: tự sinh `title`, `description`, `domain`, `tags` — Haiku tool use + Pydantic schema, FE prefill form để user review. Preview cắt 5 trang đầu / 3 phút đầu video → file 422 trang preview ~13s thay vì ~10 phút Docling full.
5. **Hỏi đáp chuyên gia** theo 10 domain (BIM, MEP, marketing, pháp lý, sản xuất, công nghệ thông tin, nhân sự, tài chính, kinh doanh, thiết kế) hoặc **chat chung** (không chọn domain — fanout 20 collection).
6. **Streaming SSE** — token-by-token qua `POST /api/chat/stream` (fallback JSON tại `POST /api/chat/`).
7. **Gợi ý câu hỏi tiếp theo** tự động sau mỗi câu trả lời.
8. **Hybrid memory 3 tầng** — sliding window + rolling summary (cùng session) + vector recall cross-session theo user.
9. **Prompt caching 2 breakpoint** — cache persona+rules (system) và lịch sử hội thoại (messages) để cắt ~90% cost+latency ở turn thứ 2 trở đi.

Bộ công nghệ lõi: **Claude Sonnet 4** (trả lời) + **Claude Haiku 4.5** (Vision + describe table + query rewrite + metadata gen), **Voyage AI** (`voyage-3`, 1024-dim) cho embedding, **Qdrant Cloud** làm vector store.

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
                        HTTP REST + SSE
                                 |
                    +------------v-------------+
                    |   FastAPI Server         |
                    |   (app/main.py)          |
                    |   warmup: reranker +     |
                    |           payload index  |
                    +------------+-------------+
                                 |
            +--------------------+---------------------+
            |                                          |
  +---------v---------+                    +-----------v-----------+
  |  /api/chat        |                    |  /api/ingest          |
  |   / (JSON)        |                    |   file / video / yt   |
  |   /stream (SSE)   |                    |   + preview endpoints |
  +---------+---------+                    +-----------+-----------+
            |                                          |
  +---------v---------+                    +-----------v-----------+
  |   RAG Chain       |                    |  Ingestion Pipelines  |
  |  - Query rewrite  |                    |  - doc_parser (3-tier)|
  |    (conditional)  |                    |  - doc_pipeline       |
  |  - Embed ONCE     |                    |    (table + Vision    |
  |  - Retrieve ∥     |                    |     with context)     |
  |    recall         |                    |  - video_pipeline     |
  |  - Rerank (GPU)   |                    |  - metadata_generator |
  |  - Refuse guard   |                    |    (Haiku tool use)   |
  |  - Prompt build   |                    |                       |
  |  - Claude Sonnet4 |                    |                       |
  +---------+---------+                    +-----------+-----------+
            |                                          |
  +---------v------------------------------------------v----------+
  |                    Qdrant Vector Database                    |
  |                                                              |
  |  +---------------------------------------------------------+ |
  |  | QdrantRegistry — 20 collections phân theo domain        | |
  |  | tdi_docs_{slug}   × 10 (marketing, mep, bim, phap_ly,   | |
  |  | tdi_videos_{slug} × 10   san_xuat, cntt, nhan_su,       | |
  |  |                         tai_chinh, kinh_doanh, thiet_ke)| |
  |  | Slug khớp NestJS Categories seeder (cntt, không         | |
  |  |   cong_nghe — align với categories.seeder.ts)           | |
  |  | Ingestion route theo metadata.domain (slug)             | |
  |  | Chat có domain    → 2 collection (docs + videos slug X) | |
  |  | Chat chung (null) → fanout 20 collections song song     | |
  |  +---------------------------------------------------------+ |
  |                                                              |
  |  +----------------+                                          |
  |  | ttt_memory     |  ← Conversation memory per user/session  |
  |  |  user_id idx   |                                          |
  |  |  session_id    |                                          |
  |  +----------------+                                          |
  |                                                              |
  |  +---------------------------------------------------------+ |
  |  | vmedia_* (READ ONLY — cluster riêng)                    | |
  |  | content, design, digital, documents, fonts, image,      | |
  |  | media, qa, ttnb                                         | |
  |  +---------------------------------------------------------+ |
  +--------------------------------------------------------------+

            +-----------------------------------------------------+
            |   S3-compatible Object Storage                      |
            |   (AWS / Viettel IDC / MinIO / R2)                  |
            |   bucket/docs/<domain-slug>/<sha256>.<ext>          |
            |   bucket/videos/<domain-slug>/<sha256>.<ext>        |
            |   Domain slug ASCII: 'Pháp lý' → 'phap-ly',         |
            |   trống → 'unsorted'                                |
            |   Upload lúc ingest; URL lưu ở payload.url          |
            +-----------------------------------------------------+
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
[STORE] Qdrant — route theo metadata.domain (slug):
    collection = tdi_docs_{slug}  (VD: tdi_docs_bim, tdi_docs_cntt)
    domain BẮT BUỘC có ở form upload; _resolve_domain_store raise ValueError
      nếu trống hoặc không ∈ 10 slug hợp lệ (accept cả persona VN cũ:
      "thiết kế" → "thiet_ke" qua PERSONA_TO_DOMAIN backward compat)
    Delete chunks cũ theo doc_id trong CHÍNH collection này trước upsert
      (đổi domain cho cùng file → bản cũ ở domain khác còn lại, có chủ ý)

    Payload:
      text: plain text (cho search + LLM đọc)
      table_data: bảng gốc Markdown (chỉ khi có bảng dữ liệu)
      heading_path: ["Chương", "Mục"]
      source_name, page, doc_id, uploaded_at
      domain, title, description, tags, url (từ form)
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
    └─ File local: video_transcriber.get_transcriber()
         Thứ tự: Groq (nếu có GROQ_API_KEY) > Whisper local
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
[STORE] Qdrant — route theo metadata.domain:
    collection = tdi_videos_{slug}
    Payload: text, video_id, title, start_sec, file_source (youtube|local),
    url (https://www.youtube.com/watch?v=XXX&t=YYs), domain, tags, description
```

Playlist: lấy danh sách video → ingest tuần tự → trả về tổng hợp
`total_videos / success_count / total_chunks`.

---

## Async Ingest & Batch Upload

Endpoint sync `/file` `/video/file` `/youtube` (giữ nguyên — không phá tương thích) phù hợp cho file nhỏ. Với file lớn (>100MB) HTTP request giữ kết nối 5-15 phút → browser timeout, retry, OOM. Phase này thêm nhánh **async + batch + streaming**.

### Sơ đồ

```
                ┌─────────────────────────────────────────┐
                │  Endpoint cũ (sync, giữ tương thích)    │
                │  /file, /video/file, /youtube           │
                │  → đợi parse xong mới response          │
                └─────────────────────────────────────────┘

                ┌─────────────────────────────────────────┐
                │  Endpoint mới (async)                   │
                │  /file/async                            │
                │  /video/file/async                      │
                │  /youtube/async                         │
                │  /files            ← multipart array    │
                │  /from-urls        ← list URL + headers │
                └────────────────┬────────────────────────┘
                                 │ stream-to-disk (1MB/chunk)
                                 │ create JobStatus → InMemoryJobStore
                                 │ submit asyncio.Queue
                                 ▼ (return job_id ngay <2s)
                ┌─────────────────────────────────────────┐
                │  JobRunner (asyncio queue)              │
                │  N workers song song (default 2)        │
                │   ├─ download nếu là /from-urls         │
                │   │     (httpx.stream + size guard)     │
                │   ├─ to_thread(ingest_document)         │
                │   ├─ to_thread(ingest_video_file)       │
                │   └─ to_thread(ingest_youtube)          │
                │  → cập nhật JobStatus: parsing→done     │
                └─────────────────────────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────────┐
                │  GET /jobs/{job_id}                     │
                │  GET /batches/{batch_id}                │
                │  → poll trạng thái real-time            │
                │  TTL 1h tự dọn job đã terminal          │
                └─────────────────────────────────────────┘
```

### 5 endpoint mới

| Endpoint | Mô tả | Khi nào dùng |
|----------|-------|--------------|
| `POST /api/ingest/file/async` | 1 file doc, trả `job_id` ngay | File >100MB / Docling parse lâu |
| `POST /api/ingest/video/file/async` | 1 file video | Video >50MB / Whisper chậm |
| `POST /api/ingest/youtube/async` | URL YouTube đơn / playlist | Tránh giữ HTTP suốt playlist 30 video |
| `POST /api/ingest/files` | Batch multipart (max 50 file/req) | User chọn nhiều file 1 lần ở FE |
| `POST /api/ingest/from-urls` | Batch JSON `[{download_url, filename, headers, metadata}]` | BE đã upload S3 → đẩy URL sang AI; SharePoint/Graph download URL có Bearer token |
| `GET /api/ingest/jobs/{id}` | Trạng thái 1 job | Poll mỗi 2-5s |
| `GET /api/ingest/batches/{id}` | Tổng hợp batch | UI progress bar tổng |

### Streaming upload — chống OOM

`app/api/ingest.py::_stream_to_disk` đọc UploadFile theo chunk `UPLOAD_STREAM_CHUNK` (1MB), ghi xuống tmp file, tính sha256 cùng lúc. Reject sớm nếu vượt `MAX_UPLOAD_MB` (5GB default). Sau khi có hash → rename thành tên persistent `{hash8}_{safe_name}.{ext}` trong `data/uploads/`. Trước fix này, `await file.read()` load cả file vào RAM → file 1GB chiếm 1GB RAM Python overhead × N concurrent uploads.

### JobStore — in-memory tracking

`app/core/job_store.py::InMemoryJobStore`:

```python
JobStatus(
  job_id, job_type, filename,
  status: queued | downloading | parsing | embedding | done | failed,
  progress: 0..1,
  chunks_added, pages, error,
  batch_id, created_at, started_at, finished_at, metadata,
)
```

- Dict trong RAM, có `asyncio.Lock`.
- TTL `INGEST_JOB_TTL_SEC=3600` — job terminal quá 1h tự dọn (loop 5 phút/lần).
- Restart process mất hết job đang chạy — chấp nhận được vì BE/FE sẽ retry. Khi BE chuyển sang RabbitMQ, swap class này bằng `RedisJobStore` cùng interface.

### JobRunner — asyncio queue + worker pool

`app/core/job_runner.py::JobRunner`:

- N worker tasks (`INGEST_WORKER_CONCURRENCY=2` default) `await queue.get()` song song.
- Pipeline ingest hiện tại là sync (Docling, Whisper, Voyage) → wrap qua `asyncio.to_thread()` để không block event loop.
- Dispatch theo `job_type`:
  - `document` → `ingest_document(file_path, original_name, metadata)`
  - `video_file` → `ingest_video_file(local_path, original_name, metadata)`
  - `youtube` → `ingest_youtube` hoặc `ingest_youtube_playlist`
- Resolve file: `payload.file_path` (multipart đã lưu disk) HOẶC `payload.download_url` (httpx.stream chunked download với headers Bearer token).
- Job-level fault isolation: 1 job fail không phá batch, không kill worker.

### Batch tracking

```python
BatchSummary(
  batch_id, total,
  queued, in_progress, done, failed,
  chunks_added,
  jobs: [JobStatus.to_dict(), ...]
)
```

- `batch_id` link mọi job trong cùng batch
- `GET /batches/{id}` aggregate live từ JobStore — không cần ghi DB

### Preview tối ưu — `max_pages` / `clip_duration_sec`

| Định dạng | Cách lấy đoạn đầu | File 422 trang / 2h |
|-----------|-------------------|---------------------|
| PDF | `pypdf` cắt N trang đầu trước khi đưa Docling | ~13s thay vì ~10 phút |
| DOCX | `python-docx` break sau `max_paragraphs` (5 × max_pages) | <2s |
| PPTX | `python-pptx` `slides[:max_slides]` | <2s |
| XLSX | `openpyxl read_only` cap rows (50 × max_pages) | <2s |
| TXT/MD | Đọc N bytes đầu (10KB × max_pages) | <500ms |
| Video | `ffmpeg -t {sec} -c copy` stream-copy clip → Whisper transcribe clip | ~20s thay vì 30 phút |

Endpoint `/file/preview` dùng `parse(path, max_pages=PREVIEW_MAX_PAGES)` (mặc định 5). Endpoint `/video/file/preview` dùng `transcriber.transcribe(path, clip_duration_sec=PREVIEW_VIDEO_CLIP_SEC)` (mặc định 180s). Endpoint ingest thật **không** truyền các param này — parse full để có chunk đầy đủ vào Qdrant.

### Lifecycle FastAPI

`app/main.py::_lifespan` extend thêm 2 hook:

- **Startup**: `await get_runner().start()` — tạo N worker task + cleanup loop
- **Shutdown**: `await get_runner().stop()` — cancel + drain pending tasks

### Khi BE có RabbitMQ queue (phase sau)

Chỉ thay `JobRunner` bằng `aio_pika.Consumer` gọi cùng `_handle_document` / `_handle_video_file` / `_handle_youtube`. Pipeline `ingest_document` / `ingest_video_file` không phải sửa. Endpoint async hiện tại có thể giữ lại để FE/test thủ công không cần RabbitMQ.

---

## AI Auto-Metadata (preview before ingest)

Khi user upload, FE gọi endpoint **preview** để AI sinh metadata trước khi commit ingest. User review + sửa → submit endpoint chính với metadata đã duyệt.

### 3 endpoint preview

| Endpoint | Input | Cơ chế | Sinh field gì |
|----------|-------|--------|---------------|
| `POST /api/ingest/file/preview` | multipart file | Docling parse → lấy 5 trang đầu (~3000 token) → Haiku tool use | title, description, domain, tags |
| `POST /api/ingest/video/file/preview` | multipart file video | `get_transcriber()` (Groq nếu có, fallback Whisper local) → ghép segment text → Haiku tool use | title, description, domain, tags |
| `POST /api/ingest/youtube/preview?url=…` | query url | `yt-dlp --dump-single-json` (title/description/thumbnail/channel/duration) + transcript best-effort → Haiku chỉ classify domain/tags | title+description (từ YouTube) + domain+tags (từ AI) |

**Playlist URL** → `/youtube/preview` trả `status=ok` với `is_playlist: true`, metadata cấp playlist (title/description/uploader/thumbnail/video_count). Metadata sẽ gắn vào mọi chunk con khi ingest thực sự.

### Structured output — Anthropic tool use + Pydantic

`app/ingestion/metadata_generator.py::generate_document_metadata()` dùng **tool use** ép Claude gọi virtual tool `save_document_metadata` với JSON schema:

```python
{
  "title":       {"type": "string", "minLength": 3, "maxLength": 200},
  "description": {"type": "string", "minLength": 10, "maxLength": 500},
  "domain":      {"type": "string", "enum": [
                   "marketing", "mep", "bim",
                   "phap_ly", "san_xuat", "cntt",
                   "nhan_su", "tai_chinh",
                   "kinh_doanh", "thiet_ke"]},
  "tags":        {"type": "array", "items": {"type": "string"},
                  "minItems": 0, "maxItems": 10}
}
```

Domain là `Literal` enum trong Pydantic → LLM **không thể sinh label ngoài 10 giá trị** (constrained decoding ở API level). Slug ASCII khớp với `knowledge_center_backend/src/database/seeds/categories.seeder.ts` — đảm bảo Python RAG và NestJS nói chung 1 ngôn ngữ. Upload bắt buộc chọn 1 trong 10 slug → không còn nhãn `"mac_dinh"` ở classifier (persona `"mac_dinh"` chỉ còn dùng cho chat chung). `"kết cấu"` đã được gộp vào `"thiet_ke"` để khớp với UI 10 domain. `DOMAIN_VALUES` được derive từ `get_args(DomainLiteral)`, kèm assert đầu module: (a) `DOMAIN_VALUES ⊂ DOMAIN_PERSONAS.keys()`, (b) `DOMAIN_PERSONAS.keys() − DOMAIN_VALUES == {"mac_dinh"}` — build fail ngay nếu ai đổi persona mà quên sync classifier. Tags được normalize sau (lowercase, dedup, strip punctuation, cap 8).

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
[REWRITE] app/core/conv_query_rewriter.py — 2 chiến lược, trigger độc lập
    Query ≥ CONV_REWRITE_MIN_LEN (40 ký tự) → skip toàn bộ (tự đủ ngữ cảnh).
    Nếu < 40 ký tự, kiểm tra 2 trigger:

    1. Anaphora rewrite (cần history):
       Câu lược chủ ngữ / dùng đại từ ("cái đó thế nào?", "nó là gì?")
       → Haiku resolve về danh từ cụ thể từ 2 turn gần nhất.
       Chỉ trigger khi CÓ history + match _ANAPHORA_MARKERS.

    2. Short-query expansion (KHÔNG cần history):
       Query ≤ 5 từ (_SHORT_QUERY_MAX_TOKENS) + không có _QUESTION_MARKERS
       ("bài này chill phết", "báo cáo Q1", "authen teams")
       → Haiku expand thành câu hỏi đầy đủ giữ nguyên keyword gốc.
       Fix dense embedding yếu với title/keyword-match query.
       Prompt riêng (_EXPAND_SYSTEM_PROMPT) — KHÔNG đổi keyword, KHÔNG đoán nội dung.

    Nếu không trigger cả 2 → giữ nguyên query.
    Cắt 1-3s/turn khi query đã standalone và đủ dài.
    │
    ▼
[EMBED ONCE] Voyage embed_query (LRU cache 256 keys)
    Cùng query string (normalized) trong phiên → trả vector cached, 0ms.
    Vector này REUSE cho cả retriever + conv_memory — tiết kiệm 1 Voyage call/turn
    (quan trọng với free tier 3 RPM).
    │
    ▼
[RETRIEVE ∥ RECALL] concurrent.futures.ThreadPoolExecutor(max_workers=2)
    │   Routing theo domain — KHÔNG dùng Qdrant filter vì đã phân collection:
    │     • domain = X (slug "cntt" hoặc persona VN "công nghệ thông tin")
    │         → registry.get_by_persona(X, "docs") + get_by_persona(X, "videos")
    │         → 2 collection (tdi_docs_X + tdi_videos_X)
    │     • domain = None (chat chung — "general" / "mac_dinh" / "mặc định")
    │         → all_docs_stores() + all_videos_stores()
    │         → fanout 20 collections song song
    │     • slug/persona không map được → fallback fanout 20 + log warning
    │
    ├─ retriever.retrieve()  (parallel 2 nguồn khi có domain, 20+vmedia khi chat chung)
    │    dedup theo prefix text (80 ký tự) + sort theo score
    │
    └─ conv_memory.retrieve() — nếu `should_skip_recall(query) = False`
         skip cho: len < 6, chào hỏi, ack, yes/no
         filter: user_id = X, kind ∈ {conversation_pair, null}
         exclude: session_id hiện tại (L1 window đã cover)
    │
    ▼
[RERANK] CrossEncoderReranker (BAAI/bge-reranker-v2-m3, ~568M)
    Auto-detect device: cuda > mps > cpu (override RERANKER_DEVICE)
    Warmup eager-load tại FastAPI lifespan → request đầu ~35ms thay vì 5s
    Trả về top-RERANK_TOP_K (mặc định 5)
    │
    ▼
[GUARD] chain._should_refuse(hits)
    hits rỗng → trả refusal template cứng, KHÔNG gọi Claude.
    top_score < 0.15 → chèn _LOW_CONFIDENCE_HINT vào user_turn,
    để Claude tự quyết qua <refusal_protocol> thay vì hard-refuse.
    │
    ▼
[PROMPT BUILD] — phân chia theo trục CACHE / KHÔNG CACHE
    │
    ├─ system (STABLE → cache_control: ephemeral)
    │    persona[domain] + _BASE_RULES (language/grounding/refusal/
    │    answer_format/citation/followup)
    │
    └─ messages:
         ...history (sliding window, đã được pin cache_control ở assistant
            gần nhất — xem "Prompt caching" phía dưới)...
         {
           role: "user",
           content:
             <retrieved_documents>...</retrieved_documents>    ← TOP (long-context)
             <user_context>...</user_context>                   ← recall pairs
             <session_summary>...</session_summary>             ← rolling summary
             <task>Dựa CHỈ vào <retrieved_documents>...</task>  ← reminder ngắn
             Câu hỏi của tôi: {query}                           ← BOTTOM (+30% quality)
         }
    │
    ▼
[LLM] Claude Sonnet 4 — sync (generate) hoặc stream (generate_stream)
    Log usage: input / cache_write / cache_read / output tokens
    │
    ▼
[PARSE]
    - Split tại "---GỢI Ý---" → (clean_answer, suggested_questions[≤3])
    - Lọc sources: chỉ giữ score ≥ 0.2 (_MIN_SOURCE_SCORE_TO_SHOW),
      tối đa 2 nguồn (_MAX_SOURCES_TO_SHOW)
    - Mỗi source kèm excerpt (~320 chars) từ chunk gốc để FE hiển thị
    - Confidence = high / medium / low theo top score
```

### Domain chuyên gia (10 slug + 1 chat-chung)

| Slug | Vai trò | Thuật ngữ chốt (không dịch) | Upload | Chat |
|------|---------|-----------------------------|:------:|:----:|
| `mac_dinh` | Trợ lý Tri thức TDI tổng quát | — | ❌ | ✅ (chat chung) |
| `bim` | Chuyên gia BIM 10+ năm | LOD, clash detection, federated model, CDE, ISO 19650 | ✅ | ✅ |
| `mep` | Kỹ sư trưởng MEP 12+ năm | HVAC, sprinkler, busduct, ELV, BMS, TCVN/ASHRAE/NFPA | ✅ | ✅ |
| `marketing` | Giám đốc Marketing 10+ năm | brand positioning, conversion rate, customer journey, MQL/SQL | ✅ | ✅ |
| `phap_ly` | Trưởng phòng Pháp chế 12+ năm | Điều/Khoản/Điểm, Chủ đầu tư vs Nhà đầu tư, Nghị định/Thông tư | ✅ | ✅ |
| `san_xuat` | Giám đốc Sản xuất 12+ năm | OEE, takt vs cycle time, Kaizen, yield rate | ✅ | ✅ |
| `cntt` | Giám đốc CNTT 12+ năm | VPN, SSO, backup vs DR, endpoint | ✅ | ✅ |
| `nhan_su` | CHRO 12+ năm | C&B, OKR vs KPI, thử việc ≠ học việc, BHXH/BHYT/BHTN | ✅ | ✅ |
| `tai_chinh` | CFO 12+ năm | doanh thu/lợi nhuận/dòng tiền, EBITDA, NPV, VAS/IFRS | ✅ | ✅ |
| `kinh_doanh` | Giám đốc Kinh doanh 12+ năm | pipeline, KAM, closing rate, forecast | ✅ | ✅ |
| `thiet_ke` | Giám đốc Thiết kế 12+ năm (gồm cả kết cấu) | concept vs schematic design, shop drawing, BTCT, mác bê tông | ✅ | ✅ |

Mỗi persona có khối **PHẠM VI** (trả lời / không trả lời) và **THUẬT NGỮ CHUẨN** (tránh dịch sai gây nhiễu). Slug ASCII là nguồn chân lý cross-layer — khớp với NestJS Categories seeder, S3 path, và Qdrant collection name.

- **Upload**: bắt buộc chọn 1 trong 10 slug concrete. Form gửi `domain=""` hoặc `"mac_dinh"` → `_validate_domain` reject với HTTP 200 error message. FE nên gửi slug; legacy client gửi persona VN (`"thiết kế"`) vẫn được accept qua `PERSONA_TO_DOMAIN` map.
- **Chat**: FE gửi slug (VD `"cntt"`) hoặc persona VN (legacy). `chat.py` map `"general"`/`"mặc định"`/`"mac_dinh"` → `None` → retriever vào nhánh fanout 20 collection. Slug/persona khác → route single collection pair.

Lookup case-insensitive + trim (`.lower().strip()`). Registry `get_by_persona` nhận cả slug (`"cntt"`) lẫn persona VN (`"công nghệ thông tin"`) — map về cùng `tdi_docs_cntt`/`tdi_videos_cntt`. `build_system_prompt` cũng dual-accept: slug trực tiếp → DOMAIN_PERSONAS["cntt"], persona VN → fallback qua PERSONA_TO_DOMAIN.

### Prompt caching — 2 breakpoint

Anthropic cho phép tối đa 4 `cache_control` block/request. Ở đây dùng 2:

1. **System prompt** (persona + `_BASE_RULES`) — stable cross-turn → cache_control ephemeral ngay trong block system. Cache HIT từ turn 2 trở đi cho cùng domain.
2. **Lịch sử assistant gần nhất** — `_attach_history_cache()` trong `claude_client.py` scan từ cuối `messages` lùi về, gắn `cache_control` vào block cuối của **assistant gần nhất** (bỏ qua user mới vì user mới chứa `<retrieved_documents>` đổi mỗi request). Turn N+1 sẽ cache HIT toàn bộ prefix `[system + user1 + asst1 + ... + asstN]` trong lookback window của Anthropic (≤20 blocks).

Tại sao KHÔNG cache documents/conv_block? Chúng đổi mỗi turn → cache luôn miss → chỉ tốn 1.25× cache-write premium mà không bao giờ đọc lại.

Log usage in-line (log level INFO, namespace `app`):

```
claude usage [sync]: in=12340 cache_write=8200 cache_read=0   out=612    # turn 1
claude usage [sync]: in=12840 cache_write=500  cache_read=8200 out=580   # turn 2 — HIT
```

### Refusal policy — delegate sang Claude

Trước đây `_should_refuse()` hard-refuse khi `top_score < ngưỡng` (0.18 → 0.25). Vấn đề: BGE reranker cho score thấp/âm với query ngắn ("bài này chill phết") dù chunk thật sự liên quan → refuse sai.

**Cơ chế mới (2 tầng):**

| Tình huống | Xử lý |
|-----------|-------|
| `hits` rỗng hoàn toàn | Hard-refuse: trả `_REFUSAL_TEMPLATE` cứng, không gọi Claude |
| `hits[0].score < 0.15` (`_LOW_CONFIDENCE_SCORE`) | Chèn `_LOW_CONFIDENCE_HINT` vào `user_turn` — Claude đọc hint + tự áp `<refusal_protocol>` nếu doc không liên quan |
| Score ≥ 0.15 | Bình thường — Claude trả lời dựa trên `<retrieved_documents>` |

```
_LOW_CONFIDENCE_HINT:
"<retrieval_confidence>low</retrieval_confidence>
LƯU Ý: độ tương đồng giữa câu hỏi và tài liệu truy xuất đang THẤP.
Kiểm tra kỹ <retrieved_documents> có thật sự trả lời được câu hỏi không —
nếu không, áp dụng <refusal_protocol>. Không suy đoán từ kiến thức chung."
```

Lợi ích: query ngắn title-match vẫn qua được guard; Claude phân biệt được "doc khớp nhưng ít thông tin" vs "doc không liên quan" và phản hồi phù hợp thay vì refuse cứng sai.

Refusal template cứng:
```
Tài liệu TDI hiện chưa có thông tin về câu hỏi này.
Bạn có thể:
- Bổ sung tài liệu liên quan qua trang nạp dữ liệu.
- Thử đổi sang lĩnh vực 'Tất cả lĩnh vực' để mở rộng tìm kiếm.
- Diễn đạt lại câu hỏi với từ khoá cụ thể hơn.
```

`<refusal_protocol>` vẫn có trong `_BASE_RULES` là rào cuối khi Claude tự đánh giá docs không đủ.

### Suggested Questions

Mỗi câu trả lời kèm tối đa 3 câu hỏi gợi ý ở cuối, bắt đầu bằng `---GỢI Ý---`:

```
1. Timeline cụ thể từng tháng ra sao?
2. Có bao nhiêu sự kiện nội bộ trong Q2?
3. Ngân sách dự kiến?
```

Parser regex tách ra trường `suggested_questions[]` trong response JSON. Câu xã giao (chào, cảm ơn) → Claude bỏ luôn khối `---GỢI Ý---`.

**Quy tắc câu gợi ý (chặt hơn từ refactor prompt):**
- Mỗi câu PHẢI hỏi về chi tiết/bước/thông số **đã xuất hiện nguyên văn** trong `<retrieved_documents>` — không mở rộng ra lĩnh vực khác.
- Không gợi ý câu mà câu trả lời đã trả lời xong (tránh lặp).
- Nếu docs chỉ có 0-1 ý đủ để gợi ý → cho phép xuất 0-2 câu thay vì ép đủ 3. Thà thiếu còn hơn bịa.

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
  │     ├─ conv_query_rewriter.rewrite()   # anaphora (cần history) | short-query expand
  │     ├─ voyage.embed_query()            # LRU cache, 1 lần, reuse xuống dưới
  │     ├─ parallel (max_workers=2):
  │     │   ├─ retriever.retrieve(query_vec=vec)   # doc RAG (ttt_* + vmedia_*)
  │     │   └─ conv_memory.retrieve(query_vec=vec) # tầng 3 — skip nếu chào/ack
  │     ├─ reranker.rerank()               # GPU auto (cuda > mps > cpu)
  │     ├─ _should_refuse(hits)            # pre-LLM guard, score < 0.18 → refusal
  │     ├─ build_system_prompt(domain)     # stable → cache breakpoint #1
  │     ├─ build_documents_block(hits)     # <retrieved_documents>
  │     ├─ build_conversation_block()      # <user_context> + <session_summary>
  │     ├─ build_user_turn()               # docs + conv + <task> + query ở BOTTOM
  │     └─ claude.generate()               # _attach_history_cache pin breakpoint #2
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

**Read-side skip** (không liên quan upsert):

- `ConversationMemory.should_skip_recall(query)` — nếu query < 6 ký tự hoặc match `_SKIP_PATTERNS` → chain **không submit** Qdrant recall call. Chào/ack không cần recall, tiết kiệm 1 round-trip + noise prompt.
- **Same-session filter** — `retrieve()` loại pair cùng `session_id` hiện tại vì đã có trong L1 window + L2 summary.

**Bootstrap collection + indexes** — tại lifespan `FastAPI`, `conv_memory.ensure_indexes()` chạy 2 bước (idempotent):

1. `ensure_collection()` — `GET /collections/ttt_memory`. Nếu 404 → `PUT` tạo mới với vector config `size=VOYAGE_DIM (1024), distance=Cosine, unnamed vector`. Trước fix này Qdrant KHÔNG auto-create collection → upsert đầu tiên trả 404, pair bị `try/except` nuốt im lặng.
2. `PUT /collections/ttt_memory/index` cho `user_id` + `session_id` (keyword schema). Trước fix này, mọi filter bị Qdrant từ chối 400 → tier-3 recall + semantic dedup chưa từng chạy đúng.

Registry cho domain collections (`QdrantRegistry.ensure_all`) cũng idempotent — cùng chạy ở lifespan: tạo đủ **20 tdi_* collection + 1 ttt_memory = 21 collection** sau lần khởi động đầu.

Threshold chọn dựa trên: Mem0 paper dùng 0.95 cho entity merge, EMem paper dùng 0.90 cho synonym edge. 0.92 = trung dung cho Voyage-3 1024-dim. Log mỗi lần skip để audit volume ("conv_memory skip upsert (heuristic/hash/semantic): user=... reason=...").

### Endpoints quản lý

| Endpoint | Hành vi |
|----------|---------|
| `DELETE /api/chat/memory/user/{user_id}` | Xoá toàn bộ pair của user trong Qdrant (GDPR / reset test) |
| `DELETE /api/chat/memory/session/{session_id}` | Xoá file JSON + pair cùng session_id trong Qdrant |

---

## Voyage Embed — tối ưu free tier 3 RPM

Voyage free plan giới hạn **3 request/phút**. Không xử lý khéo, chỉ 2-3 user chat song song là dính 429 và backoff nối đuôi → latency >60s. 3 đòn bẩy trong `app/core/voyage_embed.py`:

1. **LRU query cache** (`_QUERY_CACHE_SIZE=256`) — key theo `(model, text_normalized)` với text lowercased + collapsed whitespace. `embed_query()` trả vector cached nếu hit → **0 network call**. Chat thực tế có 15-25% turn user gõ lại query gần giống (typo, follow-up) nên cache hit-rate cao.
2. **429 backoff dùng `Retry-After` header** — trước đây hardcoded 25s × 3 retry = 75s idle. Giờ parse `Retry-After` thật (clamp `[1, 15]`), fallback 5s nếu header trống.
3. **Embed-once reuse** — `RAGChain.answer()` embed 1 lần, truyền `query_vec=` xuống `retriever.retrieve()` và `conv_memory.retrieve()` thay vì mỗi nơi embed lại. Cắt **2 → 1 Voyage call/turn**.

Kết quả commit `d525940`: turn 2+ latency **>60s → ~3-5s** trên Voyage free tier.

---

## Cấu trúc thư mục

```
trungtamtrithuc/
├── app/
│   ├── main.py                  # FastAPI + CORS + static + lifespan (warmup + ensure_indexes)
│   ├── config.py                # Env vars (Voyage, Qdrant, Claude, Proxy, Conv tuning)
│   ├── schemas.py               # ChatRequest/Response, Ingest, KnowledgeSearch
│   ├── api/
│   │   ├── chat.py              # POST /api/chat/ (JSON) + /api/chat/stream (SSE)
│   │   └── ingest.py            # SYNC: POST /api/ingest/{file,video/file,youtube,youtube-playlist}
│   │                            #   + /…/preview (cắt N trang/N giây cho file lớn)
│   │                            # ASYNC: /file/async, /video/file/async, /youtube/async
│   │                            #   → trả job_id ngay, JobRunner xử lý nền
│   │                            # BATCH: /files (multipart array), /from-urls (list URL+headers)
│   │                            # POLL: GET /jobs/{id}, GET /batches/{id}
│   │                            # Streaming upload 1MB/chunk (chống OOM với file >100MB)
│   ├── core/
│   │   ├── chunker.py           # Heading-aware chunking (tiktoken cl100k_base)
│   │   ├── claude_client.py     # Anthropic messages client (sync + stream)
│   │   │                        #   + _attach_history_cache (cache breakpoint #2)
│   │   │                        #   + _log_usage (in/cache_write/cache_read/out)
│   │   ├── voyage_embed.py      # Voyage embedder + LRU query cache + 429 backoff
│   │   ├── qdrant_store.py      # QdrantStore + QdrantRegistry (20 coll)
│   │   │                        #   + DOMAINS (10 slug) + PERSONA_TO_DOMAIN
│   │   │                        #   + VMediaReadOnlyStore
│   │   ├── s3_client.py         # boto3 wrapper cho S3-compatible (AWS/Viettel IDC/MinIO/R2)
│   │   │                        #   + slugify_domain(), PREFIX_DOCS/PREFIX_VIDEOS
│   │   ├── session_memory.py    # Sliding window + rolling summary (file JSON)
│   │   ├── conv_memory.py       # Vector recall Qdrant ttt_memory (cross-session)
│   │   │                        #   + ensure_indexes, should_skip_recall, 4-layer guards
│   │   ├── conv_summarizer.py   # Haiku summarize rolled turns
│   │   ├── conv_query_rewriter.py  # Haiku rewrite — anaphora-conditional
│   │   ├── job_store.py         # InMemoryJobStore + JobStatus + BatchSummary
│   │   │                        #   (TTL cleanup, asyncio.Lock thread-safe)
│   │   └── job_runner.py        # JobRunner: asyncio.Queue + N workers
│   │                            #   dispatch document/video_file/youtube
│   │                            #   httpx stream-download cho /from-urls
│   ├── ingestion/
│   │   ├── doc_parser.py        # 3-tier parser + typo fix
│   │   │                        #   + parse(max_pages) — preview cắt N trang đầu
│   │   │                        #   + _truncate_pdf (pypdf) cho PDF preview
│   │   ├── doc_pipeline.py      # Table detect + LLM describe + Vision+context
│   │   ├── metadata_generator.py   # Haiku tool use + Pydantic → title/desc/domain/tags
│   │   ├── video_pipeline.py    # YouTube + local + playlist
│   │   ├── video_transcriber.py # Groq (if key) > Whisper local
│   │   │                        #   + transcribe(clip_duration_sec) — preview cắt N giây
│   │   │                        #   + _clip_media (ffmpeg stream-copy ~1s)
│   │   └── youtube_fetcher.py   # youtube-transcript-api + yt-dlp + proxy rotation
│   └── rag/
│       ├── chain.py             # Rewrite → embed-once → retrieve∥recall → rerank → guard → generate
│       ├── retriever.py         # Multi-source parallel search (3 nguồn, query_vec reuse)
│       ├── reranker.py          # Cross-encoder (BGE v2-m3), GPU auto + warmup
│       └── prompt_builder.py    # DOMAIN_PERSONAS (12) + _BASE_RULES + build_system/docs/conv/user_turn
├── web/                         # Static frontend
│   ├── index.html
│   ├── chat.html                # Chat UI + SSE stream + markdown render
│   │                            #   source pills: icon doc/video, excerpt trích đoạn per trang/timestamp
│   ├── ingest.html              # Upload form + AI preview badge (AI tím, YT đỏ)
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
- **GPU tuỳ chọn**: NVIDIA CUDA (Linux) hoặc Apple MPS (Mac). Reranker tự
  detect; không có GPU vẫn chạy trên CPU.
- `poppler` (pdf2image): `brew install poppler` / `apt install poppler-utils`
- `ffmpeg` (nếu ingest video local với Whisper): `brew install ffmpeg`

### Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install poppler

cp .env.example .env
# Bắt buộc: ANTHROPIC_API_KEY, VOYAGE_API_KEY, QDRANT_URL, QDRANT_API_KEY
# Tuỳ chọn: RERANKER_DEVICE=cpu|cuda|mps (mặc định auto-detect)
# Tuỳ chọn Groq (video transcribe nhanh hơn Whisper local): GROQ_API_KEY
# Tuỳ chọn vmedia cross-cluster: QDRANT_VMEDIA_URL, QDRANT_VMEDIA_API_KEY

./run.sh
```

### Biến môi trường chính

| Group | Biến | Default | Ghi chú |
|-------|------|---------|---------|
| Retrieval | `TOP_K` | 7 | Số chunk lấy từ mỗi nguồn trước rerank |
| Retrieval | `RERANK_TOP_K` | 5 | Số chunk giữ lại sau rerank gửi vào prompt |
| Conv | `CONV_WINDOW_TURNS` | 3 | Số turn giữ trong sliding window |
| Conv | `CONV_REWRITE_MIN_LEN` | 40 | Query ≥ len này → skip rewrite (auto-standalone) |
| Conv | `CONV_DEDUP_THRESHOLD` | 0.92 | Cosine ngưỡng semantic dedup ttt_memory |
| Conv | `CONV_HASH_CACHE_SIZE` | 2000 | LRU hash dup per-user |
| Conv | `CONV_MIN_USER_CHARS` | 20 | User msg < → skip upsert |
| Conv | `CONV_MIN_BOT_CHARS` | 40 | Bot msg < (và không "Nguồn:") → skip |
| Rerank | `RERANKER_DEVICE` | auto | `cpu` / `cuda` / `mps` |
| Ingest | `MAX_UPLOAD_MB` | 5120 | Cap kích thước 1 file (MB). Vượt → reject 413. nginx phải set `client_max_body_size` khớp |
| Ingest | `INGEST_WORKER_CONCURRENCY` | 2 | Số job xử lý song song. Tăng nếu RAM khoẻ (mỗi PPTX/Vision tốn 1-2GB) |
| Ingest | `INGEST_MAX_BATCH_SIZE` | 50 | Cap số file/batch (multipart `/files` hoặc `/from-urls`) |
| Ingest | `INGEST_JOB_TTL_SEC` | 3600 | Job terminal giữ trong RAM bao lâu trước khi dọn |
| Ingest | `UPLOAD_STREAM_CHUNK` | 1048576 | Chunk size khi stream-to-disk (bytes, default 1MB) |
| Preview | `PREVIEW_MAX_PAGES` | 5 | Số trang/slide đầu cho `/file/preview` |
| Preview | `PREVIEW_VIDEO_CLIP_SEC` | 180 | Số giây đầu video cho `/video/file/preview` |

### Truy cập

| URL | Mô tả |
|-----|-------|
| `http://localhost:8000/` | Trang chủ |
| `http://localhost:8000/chat.html` | Chat (SSE stream) |
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
| Claude Sonnet 4 answer (turn 1, no cache) | Mỗi câu | ~$0.005-0.02 |
| Claude Sonnet 4 answer (turn 2+, cache HIT) | Mỗi câu | ~$0.001-0.005 ~5-10× rẻ hơn |
| Haiku query rewrite | Turn có anaphora + short | ~$0.0002/turn |
| **File text thuần** (Docling OK) | | **~$0.001** |
| **File phức tạp** (Vision + LLM) | | **~$0.005-0.01** |
| **Chat 1 turn (turn 1)** | | **~$0.005-0.02** |
| **Chat 1 turn (turn 2+ cache HIT)** | | **~$0.001-0.005** |

---

## Bảng công nghệ

| Thành phần | Công nghệ |
|------------|-----------|
| Backend | Python 3.12, FastAPI, Uvicorn |
| LLM trả lời | Claude Sonnet 4 (`claude-sonnet-4-20250514`) |
| LLM Vision / Describe table / Rewrite / Metadata | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) |
| Prompt caching | Anthropic ephemeral cache — 2 breakpoint (system + history) |
| Embedding | Voyage AI `voyage-3` — 1024-dim, LRU cache 256 keys |
| Vector DB | Qdrant Cloud (2 cluster: main + vmedia read-only) |
| PDF parser | Docling (IBM) → Claude Vision → pdfplumber |
| PDF → image | pdf2image + poppler |
| PDF preview truncation | pypdf (cắt N trang đầu trước Docling) |
| Async ingest queue | `asyncio.Queue` + N workers (default 2), `httpx.stream` cho `/from-urls` |
| Job tracking | In-memory dict + `asyncio.Lock`, TTL cleanup 1h |
| DOCX | Docling / python-docx |
| XLSX | openpyxl |
| Video transcribe | yt-dlp, youtube-transcript-api, Groq (nếu có key), Whisper local |
| Reranker | sentence-transformers cross-encoder (BGE v2-m3), torch (CUDA/MPS/CPU auto) |
| Tokenizer | tiktoken (`cl100k_base`) |
| Object storage | boto3 (S3-compatible: AWS/Viettel IDC/MinIO/R2) |
| Streaming | Server-Sent Events (SSE) |
| Frontend | HTML/CSS/JS thuần (static) |

---

## Các dòng chảy dữ liệu chính

### 1. Ingest tài liệu

```
Upload → _validate_domain(domain) — reject sớm nếu trống/sai (HTTP 200 error)
  → tempfile
  → payload.url = URL từ metadata/form → chat trả link click tải file gốc + jump #page=N
  → parse (3-tier) → doc_pipeline
  → detect tables → process (strip / describe / vision+context)
  → typo fix → chunk → embed
  → _resolve_domain_store(metadata, "docs") → tdi_docs_{slug}
  → Qdrant.upsert (tdi_docs_{slug})
  → xoá chunks cũ cùng doc_id trong CHÍNH collection này trước khi upsert mới
```

### 2. Ingest video

```
URL / File → _validate_domain(domain) — reject sớm nếu trống/sai
  → transcribe → segments
  → [S3] (chỉ với file local) upload videos/<domain-slug>/<sha>.<ext>,
         public-read. Cùng slug như tài liệu.
         → meta["url"] = S3 URL
  → chunk_transcript_with_timestamps → embed
  → _resolve_domain_store(metadata) → tdi_videos_{slug}
  → Qdrant.upsert (tdi_videos_{slug})
  → payload chứa video_id, start_sec, source_url (S3 URL với local,
    YouTube URL với remote), timestamp, youtube_url kèm &t=Xs nếu YouTube
```

### 3. Chat (RAG answer — streaming SSE)

```
POST /api/chat/stream
  → get history + summary từ session_memory (file-backed)
  → RAGChain.answer_stream(query, history, domain):
      → conv_query_rewriter.rewrite()  # anaphora (cần history) HOẶC short-query expand
                                       # skip nếu query ≥ 40 ký tự hoặc không trigger
      → voyage.embed_query()           # LRU cache
      → parallel:
           retriever.retrieve(query_vec=vec)
           conv_memory.retrieve(query_vec=vec)  # skip nếu chào/ack
      → CrossEncoderReranker (GPU auto) → top RERANK_TOP_K=5
      → _should_refuse(hits): hits rỗng → emit refusal SSE, không gọi Claude
      → build system_prompt (domain + _BASE_RULES, cache_control)
      → build user_turn: <retrieved_documents> + <user_context> +
                         <session_summary> + [_LOW_CONFIDENCE_HINT nếu score<0.15]
                         + <task> + "Câu hỏi của tôi: …"
      → _attach_history_cache (pin assistant gần nhất)
      → claude.generate_stream()
          emit events: meta → delta* → done
      → log usage in/cache_write/cache_read/out
      → split "---GỢI Ý---" → (answer, suggestions[3])
  → background: session_memory.add_turn + rolling_summary + conv_memory.upsert_pair
```

Fallback JSON tại `POST /api/chat/` giữ nguyên contract cho client chưa hỗ trợ SSE.

---

## Hướng phát triển

- ~~**Streaming response**~~ ✅ `POST /api/chat/stream` SSE, FE render token dần.
- ~~**Prompt cache tối ưu**~~ ✅ 2 breakpoint (system stable + history assistant). Cache HIT từ turn 2 trở đi, cắt ~5-10× cost prompt ổn định.
- ~~**GPU reranker**~~ ✅ Auto-detect cuda > mps > cpu + warmup lifespan.
- ~~**Pre-LLM refusal guard**~~ ✅ Hard-refuse chỉ khi `hits` rỗng. Score thấp (<0.15) → chèn `_LOW_CONFIDENCE_HINT` vào user_turn, Claude tự quyết qua `<refusal_protocol>`. Bỏ ngưỡng score cứng vì BGE reranker cho score âm với query ngắn dù chunk liên quan.
- ~~**Short-query expansion**~~ ✅ `conv_query_rewriter` thêm chiến lược 2: query ≤5 từ + không có question marker → Haiku expand thành câu hỏi đầy đủ. Fix dense embedding yếu với title-match query ("bài này chill phết" → câu hỏi rõ intent).
- ~~**Source excerpt**~~ ✅ `build_sources_mapping()` thêm field `excerpt` (~320 chars) cho mỗi source và position. FE (`chat.html`) hiển thị trích đoạn dưới mỗi trang/timestamp pill. Source lọc theo `score ≥ 0.2`, tối đa 2 nguồn.
- ~~**S3 domain-slug layout**~~ ✅ `docs/<domain-slug>/<sha256>.<ext>`.
- ~~**Qdrant collection per-domain**~~ ✅ 20 collection (`tdi_{docs,videos}_{slug}`) + registry route ingestion theo `metadata.domain`. Trước đây ingest vào `ttt_documents`/`ttt_videos` nhưng retriever tra `tdi_*_{domain}` → 0 hits cho mọi query. Upload form bắt buộc domain (1 trong 10), chat chung fanout 20 collection song song.
- ~~**Slug alignment với NestJS**~~ ✅ Domain key đổi từ persona VN (`"thiết kế"`, `"công nghệ thông tin"`) sang slug ASCII (`"thiet_ke"`, `"cntt"`) khớp với `knowledge_center_backend/src/database/seeds/categories.seeder.ts`. `cong_nghe` → `cntt`. `DOMAIN_PERSONAS` keys, `DomainLiteral` enum, web/chat.html + web/ingest.html option values, `conv_memory` default domain đều chuyển slug. Legacy client gửi persona VN vẫn hoạt động qua `PERSONA_TO_DOMAIN` backward compat map.
- ~~**Conv memory bootstrap**~~ ✅ `ensure_indexes()` giờ gọi `ensure_collection()` trước — tạo `ttt_memory` với unnamed vector 1024-dim Cosine nếu 404. Trước đây upsert đầu tiên trả 404 bị nuốt im lặng.
- ~~**Async ingest + worker pool**~~ ✅ `app/core/job_runner.py` + `job_store.py`. Endpoint `*/async` + `/files` + `/from-urls` + `/jobs/{id}` + `/batches/{id}`. asyncio.Queue + N worker (default 2), wrap pipeline sync qua `asyncio.to_thread()`. Job-level fault isolation. Endpoint cũ giữ nguyên cho tương thích.
- ~~**Streaming upload**~~ ✅ `_stream_to_disk` đọc 1MB/chunk, tính sha256 cùng lúc, reject sớm nếu vượt `MAX_UPLOAD_MB=5GB`. File 1GB không OOM (trước: `await file.read()` load cả file vào RAM × N concurrent uploads).
- ~~**Batch upload**~~ ✅ `POST /files` (multipart array max 50 file) + `POST /from-urls` (JSON list URL+headers, runner tự `httpx.stream` download). Tự route doc/video theo suffix. Errors capture per-file, không phá batch.
- ~~**Preview "đọc đoạn đầu"**~~ ✅ `parse(max_pages=5)` cắt PDF với pypdf trước Docling, DOCX/PPTX break sớm, XLSX limit row, TXT/MD đọc N bytes. `transcribe(clip_duration_sec=180)` ffmpeg stream-copy clip trước Whisper. File 422 trang preview ~13s thay vì ~10 phút.
- **Persistent job store** — In-memory hiện tại mất job khi restart. Phase sau swap sang Redis (`RedisJobStore` cùng interface) hoặc RabbitMQ-backed khi BE ready. Pipeline + handler không phải sửa.
- **Bigfile / SharePoint / PPTX 3-tier** — Phase tiếp theo: BE build Microsoft Graph integration (passport-azure-ad đã có) → publish RabbitMQ message → AI worker consume từ queue thay vì HTTP. PPTX nâng tier-2 LibreOffice→PDF→Vision per-slide.
- **Auth thật** — hiện `user_id` là string tự do client truyền; tích hợp đăng nhập để verify + chống impersonate memory của user khác.
- **User profile extract** — tách fact cá nhân (tên, team, ngân sách, preference) ra block riêng thay vì lẫn trong recall pairs — tăng độ bền trước khi recall score rơi dưới threshold.
- **Admin CRUD knowledge** — delete/re-index tài liệu từ `knowledge.html`.
- **Evaluation harness** — tập test câu hỏi + ground truth để đo recall/precision.
- **Multi-tenant** — tách namespace theo organization.
- **429 graceful UI** — chat hiển thị thông báo thân thiện khi Voyage rate-limit thay vì show raw URL + HTTPError (xem screenshot issue ngày 2026-04-22).
