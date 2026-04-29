# Handoff — Trả lời kèm ảnh trong RAG chatbot

Tài liệu bàn giao **AI side → BE/FE team** cho task "user hỏi câu hỏi → ứng dụng trả lời kèm hình ảnh được trích từ tài liệu PDF".

> **Trạng thái**: AI ingestion + retrieval đã hoàn thành và verify E2E. BE/FE còn 3 phần: aggregate hits → API endpoint serve ảnh → render UI.

---

## 1. TL;DR

Khi user hỏi 1 câu, hệ thống đã có sẵn pipeline AI:
- Trích ảnh từ PDF khi ingest, sinh caption tiếng Việt qua Haiku, lưu vào `data/images/{doc_id}/{image_id}.png`.
- Mỗi ảnh được "đại diện" bằng 1 chunk synthetic trong Qdrant với `text=caption`, `chunk_type="image_caption"`, `payload.images=[ảnh đó]`.
- Retrieval (Voyage cosine) + Rerank (BGE cross-encoder) trả về hits có `payload.images[]` đính kèm. Score top-1 cho query liên quan ảnh đạt **~0.95-0.98** sau rerank.

**Việc còn lại của BE/FE:**
1. **BE** sửa `app/rag/prompt_builder.py:build_sources_mapping` để gom `payload.images[]` từ hits → đính vào mỗi entry trong `sources[]` trả về FE. Dedupe theo `image_id`.
2. **BE** thêm endpoint `GET /api/images/{doc_id}/{image_id}` serve file PNG từ disk với cache header.
3. **FE** render ảnh trong card "Nguồn tham khảo" bên dưới câu trả lời, click để xem to.

---

## 2. Schema payload trong Qdrant (BE đọc)

### 2.1. Hai loại chunk

Mỗi document ingest ra **2 loại chunk** trong cùng collection `tdi_docs_{domain}`:

| chunk_type | Mục đích | Có ảnh trong payload? |
|---|---|---|
| `"text"` (mặc định, không set field) | Text nội dung doc — recall khi query match nội dung | `images[]` đầy đủ (mọi ảnh của trang đó) |
| `"image_caption"` | Synthetic chunk: 1 chunk / 1 ảnh duy nhất, text = caption tiếng Việt | `images[]` chỉ chứa 1 ảnh (precision) |

→ **BE không phải phân biệt khi aggregate** — cứ đọc `payload.images[]` và dedupe theo `image_id`. Field `chunk_type` chỉ dùng cho debug và tune sau (nếu cần).

### 2.2. Format `payload.images[]`

```jsonc
{
  // ... các field text chunk hiện có (text, doc_id, page, source_name, score, ...) ...
  "chunk_type": "image_caption" | "text" | undefined,
  "images": [
    {
      "image_id": "7b06bb80e4944c01",       // 16-char hex, dedupe key
      "filename": "7b06bb80e4944c01.png",   // tên file trong data/images/{doc_id}/
      "caption": "Sơ đồ kiến trúc YOLOv8 với Backbone, Neck, Head...",
      "page": 4,                            // page thật của ảnh trong PDF
      "ord": 0,                             // thứ tự ảnh trong trang
      "width": 800,
      "height": 600
    }
  ]
}
```

**Quy tắc đọc:**
- Field `images` có thể **không tồn tại** trên chunks cũ (ingest trước khi merge code mới) → BE dùng `.get("images", [])`.
- `image_id` là dedupe key. Cùng ảnh xuất hiện nhiều page hoặc nhiều chunks → BE chỉ giữ 1 entry trong response.
- File PNG vật lý nằm tại path: `<DATA_DIR>/images/{chunk.doc_id}/{image.filename}`.

### 2.3. Aggregate logic đề xuất cho `build_sources_mapping`

Hiện tại file [`app/rag/prompt_builder.py`](../app/rag/prompt_builder.py) function `build_sources_mapping(hits)` gom hits theo `(title, base_url)` thành sources. Cần mở rộng:

```python
# Trong vòng for hit in hits:
hit_images = hit.payload.get("images", []) or []
entry = seen[key]
entry.setdefault("images", {})  # dict[image_id → image_meta] để dedupe
for img in hit_images:
    image_id = img.get("image_id")
    if not image_id or image_id in entry["images"]:
        continue
    entry["images"][image_id] = {
        "image_id": image_id,
        "url": f"/api/images/{hit.payload['doc_id']}/{image_id}",
        "caption": img.get("caption", ""),
        "page": img.get("page"),
        "width": img.get("width", 0),
        "height": img.get("height", 0),
    }

# Cuối hàm khi build mapping list:
mapping_entry["images"] = list(entry["images"].values())[:_MAX_IMAGES_PER_SOURCE]
```

**Constants đề xuất** (đặt trên đầu module):
```python
_MAX_IMAGES_PER_SOURCE = 4  # giới hạn ảnh hiển thị / source
```

---

## 3. API endpoint serve ảnh (BE implement)

### 3.1. Spec

```
GET /api/images/{doc_id}/{image_id}

Path params:
  - doc_id    : 16-char hex (regex ^[a-f0-9]{16}$)
  - image_id  : 16-char hex (regex ^[a-f0-9]{16}$)

Response:
  200  → image/png file bytes
  404  → JSON {"error": "image not found"}
  400  → JSON {"error": "invalid id format"}

Headers (production):
  Cache-Control: public, max-age=86400, immutable
  Content-Type:  image/png
```

### 3.2. Implementation note

- Resolve path: `DATA_DIR / "images" / doc_id / f"{image_id}.png"`.
- **Phòng path traversal**: validate cả 2 ID khớp regex `^[a-f0-9]{16}$` trước khi dùng. Tuyệt đối **không** ghép path từ user input chưa validate.
- Sau resolve, check `path.is_file()` và `path.resolve()` nằm trong `DATA_DIR.resolve()` (defense-in-depth).
- Cache `immutable` an toàn vì path đã content-addressed bằng sha256 — file mới = id mới.

### 3.3. Mount router

Thêm vào [`app/api/server.py`](../app/api/server.py):
```python
from app.api.images import router as images_router
app.include_router(images_router)
```

Tạo file mới [`app/api/images.py`](../app/api/images.py) (~30 dòng): handler dùng `fastapi.responses.FileResponse`.

---

## 4. UI requirement (FE implement)

### 4.1. Schema response từ `/api/chat/`

Sau khi BE sửa `build_sources_mapping`, mỗi entry `sources[i]` sẽ có thêm field `images`:

```jsonc
{
  "answer": "...",
  "sources": [
    {
      "index": 1,
      "title": "test2.pdf",
      "url": "...",
      "page": 4,
      "score": 0.98,
      "excerpt": "Sơ đồ kiến trúc YOLOv8...",
      "positions": [...],
      "images": [
        {
          "image_id": "73424ff78759fd51",
          "url": "/api/images/99ac006f57bf2003/73424ff78759fd51",
          "caption": "Sơ đồ kiến trúc YOLOv8 với Backbone, Neck, Head...",
          "page": 4,
          "width": 800,
          "height": 600
        }
      ]
    }
  ]
}
```

### 4.2. Render

Trong card "Nguồn tham khảo" (đang hiển thị `title`, `excerpt`):
- Nếu `images.length > 0` → render grid 2-3 cột thumbnail bên dưới `excerpt`.
- Mỗi thumbnail: `<img src="{image.url}" alt="{image.caption}" loading="lazy" />`, max width ~200px, aspect ratio giữ theo `width/height`.
- Click thumbnail → mở lightbox/modal với ảnh full + caption làm title.
- Caption dùng làm `alt` (accessibility) và `title` (tooltip hover).

### 4.3. Edge cases UI

- `images.length === 0` (source thuần text, không có ảnh) → render card như cũ, không có grid.
- Image fail load (404) → hide thumbnail im lặng, không show broken icon.
- Caption rỗng → vẫn hiển thị thumbnail, không có alt text fallback `"Hình ảnh từ tài liệu"`.

---

## 5. Migration path — doc cũ cần re-ingest

**Quan trọng**: doc đã ingest **trước** khi merge code AI mới sẽ KHÔNG có `payload.images[]` → BE đọc ra `[]` → FE không hiển thị ảnh nào.

→ Để có ảnh, **mỗi doc PDF phải re-ingest qua pipeline mới**.

### 5.1. Cách re-ingest

**Option A — User upload lại qua FE** (đơn giản nhất):
- User mở FE → trang upload → chọn file cũ → upload.
- `ingest_document()` tự `delete_by_filter(doc_id)` chunks cũ → upsert chunks mới có ảnh.
- Phù hợp khi số lượng doc nhỏ (<50).

**Option B — Script batch migration** (cho team có nhiều doc):
Cần file PDF gốc còn lưu ở `data/uploads/` hoặc S3. Pseudo-code:
```python
for pdf_path in DATA_DIR / "uploads":
    metadata = lookup_existing_metadata(pdf_path)  # domain, title, ... từ Qdrant
    ingest_document(str(pdf_path), pdf_path.name, metadata=metadata)
```

→ Việc viết script migration thuộc team BE / DevOps, không phải AI side.

### 5.2. Cost re-ingest

| Doc size | Cost | Time |
|---|---|---|
| 1 PDF, ~5 ảnh | ~$0.005 (Haiku caption + Voyage embed) | ~10s sau khi Docling đã warm |
| 100 PDF, ~500 ảnh | ~$0.50 | Tùy concurrency |

---

## 6. Code đã thay đổi (cho code review)

| File | Thay đổi |
|---|---|
| [`app/ingestion/doc_parser.py`](../app/ingestion/doc_parser.py) | Thêm hằng số config ảnh + 3 helper: `_extract_images_from_pdf` (PyMuPDF), `_caption_images` (Haiku), `_doc_image_dir`. `parse_pdf(path, doc_id)` signature mới, trả `images` per page. |
| [`app/ingestion/doc_pipeline.py`](../app/ingestion/doc_pipeline.py) | Flatten pages giờ kéo theo `page_images`. Sau text-chunk loop sinh thêm **synthetic image-caption chunks**: 1 chunk / 1 ảnh duy nhất, embed batch caption trong 1 Voyage call. |
| [`requirements.txt`](../requirements.txt) | Thêm `pymupdf` |
| [`scratch/test_image_extract.py`](../scratch/test_image_extract.py) | Spike test extract + caption trên 1 PDF (không cần Qdrant). |
| [`scratch/test_retrieval_with_images.py`](../scratch/test_retrieval_with_images.py) | E2E test: ingest → retrieve → rerank, in score 2 stage để debug. |

**Không thay đổi** (BE/FE sẽ chỉnh sau):
- `app/rag/chain.py` — không đụng prompt builder, refusal protocol.
- `app/rag/prompt_builder.py` — đợi BE thêm aggregate logic vào `build_sources_mapping`.
- `app/api/*.py` — đợi BE thêm endpoint images.

---

## 7. Limits + edge cases (BE cần biết)

### 7.1. Filter ảnh khi ingest (đã làm trong `doc_parser.py`)

| Tham số | Giá trị | Lý do |
|---|---|---|
| `MIN_IMAGE_DIMENSION` | 100 px | Loại logo/icon noise trước khi tốn Haiku call |
| `MAX_IMAGE_BYTES` | 5 MB | Bỏ ảnh khổng lồ (full-page scan) |
| `MAX_IMAGES_PER_PAGE` | 10 | Cap để tránh slide hoa văn trang trí |
| Dedupe | sha256(bytes)[:16] | Logo lặp ở header chỉ lưu 1 file PNG |

→ Một số ảnh PDF sẽ KHÔNG được trích → ảnh "biến mất" có chủ đích, không phải bug.

### 7.2. Edge case "trang chỉ có ảnh, không text"
- Ví dụ: presentation slide chỉ có 1 ảnh full-page, không có text.
- Hiện tại pipeline **skip page này** (chunks rỗng → continue) → ảnh không vào synthetic chunks.
- **v1 chấp nhận** giới hạn này. Nếu team cần fix sau → emit synthetic chunk dùng caption làm text trực tiếp (đã có code chuẩn bị).

### 7.3. Caption rỗng / Anthropic API fail
- Nếu Haiku call fail (rate limit, network) → caption = `""`.
- Synthetic chunk **sẽ không được tạo** cho ảnh có caption rỗng (filter trong code).
- Ảnh vẫn xuất hiện trong `images[]` của text chunk (recall), nhưng không search được qua caption.

### 7.4. Format file
- Hiện tại **chỉ PDF** có pipeline ảnh. DOCX/XLSX chưa có (Surface 1+2 chỉ làm cho PDF).
- Nếu team cần DOCX → AI side làm thêm ~1 ngày (Docling đã extract ảnh DOCX, chỉ cần pipe vào extractor).

---

## 8. Test / verify

### 8.1. Smoke test pipeline AI (không cần Qdrant)
```bash
venv/bin/python scratch/test_image_extract.py docs/your_pdf.pdf
```
Kiểm tra: số ảnh extract, caption tiếng Việt OK, file PNG saved đúng.

### 8.2. E2E test retrieval (cần Voyage + Qdrant keys)
```bash
venv/bin/python scratch/test_retrieval_with_images.py docs/your_pdf.pdf [domain]
```
Kiểm tra:
- Stage 1 (Voyage): caption-specific query → top hits là `image_caption` chunks, score retr ~0.5+.
- Stage 2 (BGE rerank): top-1 image_caption đạt rerank score **~0.95-0.98**, ảnh không liên quan rớt xuống ~0.

### 8.3. BE test sau khi thêm aggregate logic
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Có sơ đồ kiến trúc nào trong tài liệu không?", "domain": "bim"}'
```
Verify response `sources[0].images[0].url` có path `/api/images/{doc_id}/{image_id}` và GET URL đó trả về PNG.

### 8.4. FE test
- Hỏi câu liên quan ảnh trong tài liệu đã re-ingest.
- Kiểm tra card source hiển thị thumbnail, click mở lightbox, ảnh load OK.

---

## 9. Liên hệ AI side

Nếu BE/FE gặp vấn đề về:
- **Ảnh không xuất hiện trong response**: kiểm tra doc đã re-ingest chưa (mục 5).
- **Caption sai/không hợp ngữ cảnh**: tune prompt `IMAGE_CAPTION_PROMPT` trong `app/ingestion/doc_parser.py`.
- **Score retrieval thấp / ảnh sai trả về**: AI side check reranker, có thể thêm boost theo `chunk_type` ở `app/rag/reranker.py`.
- **Ảnh cụ thể bị filter mất khi extract**: kiểm tra dimension/bytes limits ở mục 7.1.

Pipeline AI để mở field `chunk_type` trong payload — team BE/FE có thể tận dụng cho phân tích/tune (vd: log số % hits là image_caption per query, tune `_MAX_IMAGES_PER_SOURCE` theo dữ liệu thực tế).
