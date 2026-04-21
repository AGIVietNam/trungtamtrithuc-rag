# Hướng dẫn tích hợp — Trung Tâm Tri Thức

Tài liệu này dành cho đội FE/BE công ty muốn tích hợp Trung Tâm Tri Thức qua REST API.

## Tổng quan

- Base URL: `http://<host>:8000` (hoặc domain production)
- Tất cả request/response: `Content-Type: application/json` (trừ upload file: `multipart/form-data`)
- CORS: cho phép tất cả origins (`*`) — có thể giới hạn bằng biến môi trường trong production
- Không có authentication phía client (auth nên đặt tại API gateway / reverse proxy)

## Session flow

```
Client                          Server
  │                               │
  │  Tạo session_id (UUID v4)     │
  │  localStorage.setItem(...)    │
  │                               │
  │  POST /api/chat/ {session_id} │
  │ ─────────────────────────────►│
  │                               │  Tra cứu lịch sử hội thoại
  │                               │  từ session store
  │  {answer, sources}            │
  │ ◄─────────────────────────────│
  │                               │
  │  POST /api/chat/ {session_id} │  (Câu hỏi tiếp theo)
  │ ─────────────────────────────►│
  │                               │  Server tự nhớ context
  │  {answer, sources}            │
  │ ◄─────────────────────────────│
```

`session_id` phải là UUID v4, tạo phía client và giữ suốt phiên làm việc:

```js
const sessionId = localStorage.getItem('ttt_session_id') ?? crypto.randomUUID();
localStorage.setItem('ttt_session_id', sessionId);
```

## Endpoints chi tiết

### POST /api/chat/

**Request:**
```json
{
  "message": "string (bắt buộc)",
  "session_id": "string UUID (bắt buộc)",
  "domain": "general | education | technology | science | business | health",
  "history": [
    {"role": "user", "content": "câu hỏi trước"},
    {"role": "assistant", "content": "câu trả lời trước"}
  ]
}
```

> Trường `history` là tùy chọn. Server đã lưu lịch sử theo `session_id`; truyền `history` để override hoặc để trống `[]`.

**Response 200:**
```json
{
  "answer": "Nội dung câu trả lời...",
  "sources": [
    {
      "title": "Tên tài liệu hoặc video",
      "source": "ten_file.pdf",
      "score": 0.91,
      "content": "Đoạn trích liên quan...",
      "url": null,
      "youtube_url": "https://www.youtube.com/watch?v=ID&t=123s",
      "timestamp": 123
    }
  ],
  "session_id": "uuid-v4"
}
```

**Trường sources:**

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `title` | string | Tên tài liệu hoặc tiêu đề video |
| `source` | string | Tên file gốc |
| `score` | float 0–1 | Độ tương đồng (0.75+ = cao) |
| `content` | string | Đoạn văn bản được trích dẫn |
| `url` | string\|null | URL tài liệu nếu có |
| `youtube_url` | string\|null | Link YouTube kèm timestamp `?t=Xs` |
| `timestamp` | int\|null | Giây bắt đầu đoạn trong video |

**Confidence tag:**
```
score >= 0.75  → "Cao"
score >= 0.40  → "Trung bình"
score < 0.40   → "Thấp"
```

---

### POST /api/chat/stream

SSE streaming — cùng schema request với `/api/chat/`, response là `text/event-stream`. Dùng cho UX render token dần trong bubble chat.

**Request:** giống `/api/chat/`.

**Response:** `Content-Type: text/event-stream`. Mỗi event format `data: <json>\n\n`, `type` thuộc:

| Event | Khi nào | Payload chính |
|-------|---------|---------------|
| `meta` | Sau retrieval + rerank, trước khi LLM sinh token | `confidence`, `rewritten_query`, `recall_count` |
| `delta` | Mỗi chunk Claude stream về | `text` — client nối tiếp vào bubble đang hiện |
| `done` | Generation xong | `answer` (đã strip `---GỢI Ý---`), `sources` (nếu trích "Nguồn:"), `suggested_questions` |
| `error` | Exception trong chain | `message` |

> FE nên ẩn phần text sau marker `---GỢI Ý---` trong các event `delta` (split `/\n*---GỢI Ý---/`), sau đó render bubble cuối dùng `answer` từ event `done` để không lộ marker thô.

**Ví dụ JS (fetch + ReadableStream):**

```js
const res = await fetch('/api/chat/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message, session_id, user_id, domain, history: [],
  }),
});
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = '';
let text = '';
while (true) {
  const {value, done} = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, {stream: true});
  let idx;
  while ((idx = buffer.indexOf('\n\n')) >= 0) {
    const raw = buffer.slice(0, idx);
    buffer = buffer.slice(idx + 2);
    for (const line of raw.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const evt = JSON.parse(line.slice(5).trim());
      if (evt.type === 'delta') {
        text += evt.text;
        renderStream(text.split(/\n*---GỢI Ý---/)[0]);
      } else if (evt.type === 'done') {
        renderFinal(evt.answer, evt.sources, evt.suggested_questions);
      } else if (evt.type === 'error') {
        throw new Error(evt.message);
      }
    }
  }
}
```

> Proxy lưu ý: server đã set `X-Accel-Buffering: no`, nhưng nếu đặt sau CDN/proxy khác cần tắt compression và buffering cho route này (nginx: `proxy_buffering off`, `proxy_cache off`).

---

### POST /api/ingest/file

Upload tài liệu (PDF/DOCX/TXT/MD) hoặc video (MP4/MKV/MOV/AVI).

**Request:** `multipart/form-data`

| Field | Kiểu | Mô tả |
|-------|------|-------|
| `file` | file | File cần nạp |
| `collection` | string | `ttt_documents` (tài liệu) hoặc `ttt_videos` (video) |

**Response 200:**
```json
{
  "status": "ok",
  "chunks_added": 42,
  "message": "Đã nạp thành công 42 đoạn."
}
```

**Ví dụ JS (fetch):**
```js
async function uploadDoc(file) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('collection', 'ttt_documents');
  const resp = await fetch('/api/ingest/file', { method: 'POST', body: fd });
  return resp.json();
}
```

---

### POST /api/ingest/youtube

**Request:** Query parameter

```
POST /api/ingest/youtube?url=https://www.youtube.com/watch?v=ID
```

| Param | Kiểu | Mô tả |
|-------|------|-------|
| `url` | string | URL YouTube đầy đủ |
| `collection` | string | Mặc định: `ttt_videos` |

**Response 200:**
```json
{
  "status": "ok",
  "chunks_added": 87,
  "message": "Đã phiên âm và nạp 87 đoạn từ YouTube."
}
```

> Lưu ý: request này có thể mất vài phút tùy độ dài video. Nên gọi bất đồng bộ hoặc hiển thị loading indicator.

---

### GET /health

```json
{"status": "ok"}
```

Dùng để kiểm tra server còn sống trong load balancer / container orchestration.

---

## CORS

Mặc định server cho phép `*`. Để giới hạn trong production, sửa `app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.company.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
```

## Xử lý lỗi

| HTTP Status | Nguyên nhân | Xử lý |
|-------------|-------------|-------|
| `400` | Request sai format | Kiểm tra body/params |
| `422` | Validation error (Pydantic) | Xem trường `detail` trong response |
| `500` | Lỗi server (API key sai, Qdrant không kết nối được) | Xem logs server |
| `503` | Model đang khởi động | Retry sau 5–10s |

**Response lỗi mẫu:**
```json
{
  "detail": [
    {
      "loc": ["body", "message"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

## Lưu ý vmedia collection

Collection `vmedia` là kho video **chỉ đọc** do bên ngoài cung cấp:
- Được truy vấn tự động trong mỗi câu hỏi chat (nếu cấu hình `QDRANT_VMEDIA_API_KEY`)
- **Không thể** nạp dữ liệu vào collection này qua API của service này
- Kết quả từ vmedia xuất hiện trong `sources` với `source: "vmedia"`

## Chạy production với Docker

```bash
# Build image
docker build -t ttt-chatbot:latest .

# Chạy với env file
docker run -d \
  --name ttt-chatbot \
  --env-file .env \
  -p 8000:8000 \
  --restart unless-stopped \
  ttt-chatbot:latest

# Xem logs
docker logs -f ttt-chatbot
```

## Checklist tích hợp

- [ ] Tạo `session_id` bằng `crypto.randomUUID()` và lưu vào `localStorage`
- [ ] Luôn truyền `session_id` trong mỗi request chat
- [ ] Hiển thị `sources` kèm link YouTube `?t=Xs` nếu có `youtube_url`
- [ ] Dùng `score` để hiển thị tag Cao/TB/Thấp
- [ ] Gọi `/health` để kiểm tra server trước khi render UI
- [ ] Xử lý `422` và hiển thị thông báo lỗi thân thiện với người dùng
- [ ] Upload file: dùng `FormData`, không set `Content-Type` header thủ công (browser tự set boundary)
