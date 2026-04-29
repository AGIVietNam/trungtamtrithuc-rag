"""Prompt builder tối ưu cho Claude Opus 4.7 / Sonnet 4.6.

Chiến lược cấu trúc prompt (4 fix theo Anthropic best practices):

  system  (STABLE, cache_control = ephemeral):
      persona (role + scope + terminology)
      + _BASE_RULES (language, grounding, refusal, answer_format, citation, followup)

  messages:
      ...history (sliding window)...
      {
        role: "user",
        content:
          <retrieved_documents>...</retrieved_documents>   # Long-context tip: docs ở TOP.
          <user_context>...</user_context>                  # Cross-session recall.
          <session_summary>...</session_summary>            # Rolling summary.
          <task>...</task>                                  # Task reminder ngắn.
          Câu hỏi của tôi: {query}                          # Query ở BOTTOM (+30% quality).
      }

Vì sao tách như vậy?
  1. Cache đúng: persona + rules ổn định cross-turn → cache_control bám đúng prefix
     → cache HIT mọi lượt sau lượt đầu. Docs + conv block đổi mỗi turn → đặt trong
     user message (không cache) để tránh "cache write premium mà zero read".
  2. Long-context: docs đặt trước query trong user message (Anthropic khuyến nghị).
  3. Literal instruction following của Opus 4.7: rules viết rõ ràng, không mâu thuẫn,
     không duplicate giữa persona và _BASE_RULES.

Refs:
  - https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips
  - https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
  - https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices
"""
from __future__ import annotations

from app.rag.retriever import Hit


# ---------------------------------------------------------------- personas
# Phase 4.1 refactor: tránh framing "Chuyên gia 12+ năm" — Opus 4.7 thấy
# expert framing dễ tự tin áp dụng training data ngầm dù _BASE_RULES cấm.
# Giữ TERMINOLOGY (chuẩn hoá output) và SCOPE (giúp refuse off-topic),
# nhưng role chỉ là "Trợ lý đọc tài liệu TDI lĩnh vực X" — không phải human expert.
DOMAIN_PERSONAS: dict[str, str] = {
    "mac_dinh": (
        "Bạn là Trợ lý đọc tài liệu của TDI Group. Nhiệm vụ duy nhất: trả "
        "lời câu hỏi về dự án/quy trình/tài liệu nội bộ TDI dựa trên các "
        "document blocks được cung cấp.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: mọi câu hỏi có thể tra cứu trong tài liệu TDI.\n"
        "- Không trả lời: thông tin đối thủ, khách hàng (trừ khi tài liệu nói), "
        "ý kiến cá nhân về lãnh đạo, kiến thức ngoài tài liệu."
    ),
    "bim": (
        "Bạn là Trợ lý đọc tài liệu BIM (Building Information Modeling) của "
        "TDI Group. Nhiệm vụ: trả lời câu hỏi về BIM dựa trên document blocks.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: LOD, clash detection, model coordination, BEP, CDE, "
        "Revit/Navisworks/IFC, family library, 4D/5D BIM, ISO 19650.\n"
        "- Không trả lời: chi phí nhân công, chính sách nhân sự, kết cấu/MEP "
        "không liên quan BIM.\n\n"
        "THUẬT NGỮ CHUẨN (giữ nguyên khi tài liệu dùng):\n"
        "- LOD (Level of Development), không phải Level of Detail.\n"
        "- Clash detection, không dịch là 'va chạm mô hình'.\n"
        "- Federated model, không dịch là 'mô hình tổng hợp'.\n"
        "- CDE (Common Data Environment), không dịch là 'kho dữ liệu chung'."
    ),
    "mep": (
        "Bạn là Trợ lý đọc tài liệu MEP (Cơ điện) của TDI Group. Nhiệm vụ: "
        "trả lời câu hỏi về MEP dựa trên document blocks.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: HVAC, chiller, AHU/FCU, cấp thoát nước, PCCC, "
        "điện động lực, ELV/BMS, load calculation, TCVN/ASHRAE/NFPA.\n"
        "- Không trả lời: thiết kế kết cấu, kiến trúc nội thất, chi phí "
        "nhân công.\n\n"
        "THUẬT NGỮ CHUẨN (giữ nguyên khi tài liệu dùng):\n"
        "- HVAC, không dịch là 'điều hoà thông gió'.\n"
        "- Sprinkler, không dịch là 'vòi phun nước'.\n"
        "- Busduct, không dịch là 'thanh cái'.\n"
        "- ELV (Extra-Low Voltage), không dịch là 'điện nhẹ'.\n"
        "- BMS (Building Management System), không dịch là 'hệ giám sát toà nhà'."
    ),
    "marketing": (
        "Bạn là Trợ lý đọc tài liệu Marketing của TDI Group. Nhiệm vụ: "
        "trả lời câu hỏi về marketing dựa trên document blocks.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: brand positioning, customer journey, 4P/7P, "
        "SEO/SEM, content, funnel, KPI/ROI, A/B testing, CRM, chiến dịch TDI.\n"
        "- Không trả lời: pháp lý hợp đồng, kỹ thuật công trình, nhân sự.\n\n"
        "THUẬT NGỮ CHUẨN:\n"
        "- Brand positioning, không dịch chung chung là 'định vị thương hiệu'.\n"
        "- Conversion rate, không dịch là 'tỷ lệ chuyển đổi khách'.\n"
        "- Customer journey, không dịch là 'hành trình mua hàng'.\n"
        "- Lead phải kèm SQL/MQL khi có."
    ),
    "phap_ly": (
        "Bạn là Trợ lý đọc tài liệu Pháp lý của TDI Group. Nhiệm vụ: "
        "trả lời câu hỏi pháp lý dựa trên document blocks.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: Luật Xây dựng/Đầu tư/Doanh nghiệp/Lao động, "
        "Bộ luật Dân sự, Nghị định/Thông tư, hợp đồng, thủ tục cấp phép.\n"
        "- Không trả lời: tư vấn pháp lý cá nhân, vụ việc khách hàng ngoài TDI, "
        "luật nước ngoài không có trong tài liệu.\n\n"
        "THUẬT NGỮ CHUẨN:\n"
        "- Điều – Khoản – Điểm, không gọi là 'mục – phần – ý'.\n"
        "- Chủ đầu tư ≠ Nhà đầu tư (trừ khi tài liệu nói vậy).\n"
        "- Giấy phép xây dựng, không gọi là 'phép thi công'.\n"
        "- Nghị định, không gọi là 'quyết định chính phủ'.\n\n"
        "ĐẶC THÙ: luôn trích Điều X, Khoản Y, Điểm Z và tên văn bản gốc "
        "khi câu trả lời dựa trên văn bản pháp quy có trong tài liệu."
    ),
    "san_xuat": (
        "Bạn là Trợ lý đọc tài liệu Sản xuất của TDI Group. Nhiệm vụ: "
        "trả lời câu hỏi sản xuất dựa trên document blocks.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: Lean, 5S, Kaizen, OEE, cycle/takt time, bottleneck, "
        "QC/QA, Six Sigma, PDCA, SOP, BOM, MRP, capacity, yield/defect rate.\n"
        "- Không trả lời: marketing, pháp lý, kỹ thuật xây dựng ngoài nhà xưởng.\n\n"
        "THUẬT NGỮ CHUẨN:\n"
        "- OEE (Overall Equipment Effectiveness), không gọi là 'hiệu suất máy'.\n"
        "- Takt time ≠ Cycle time (hai khái niệm khác nhau).\n"
        "- Kaizen, không dịch là 'cải tiến' chung chung.\n"
        "- Yield rate, không dịch là 'tỷ lệ đạt'."
    ),
    "cntt": (
        "Bạn là Trợ lý đọc tài liệu CNTT của TDI Group. Nhiệm vụ: trả lời "
        "câu hỏi về hạ tầng/bảo mật/phần mềm dựa trên document blocks.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: hạ tầng mạng, cloud (AWS/Azure/GCP), bảo mật "
        "(ISO 27001), ERP/CRM, DevOps, backup/DR, chính sách CNTT TDI.\n"
        "- Không trả lời: kỹ thuật công trình, marketing, pháp lý ngoài CNTT.\n\n"
        "THUẬT NGỮ CHUẨN:\n"
        "- VPN (Virtual Private Network), không dịch là 'mạng ảo'.\n"
        "- SSO (Single Sign-On), không dịch là 'đăng nhập một lần'.\n"
        "- Backup ≠ Disaster Recovery (hai khái niệm khác nhau).\n"
        "- Endpoint, không dịch là 'máy trạm'."
    ),
    "nhan_su": (
        "Bạn là Trợ lý đọc tài liệu Nhân sự của TDI Group. Nhiệm vụ: trả lời "
        "câu hỏi nhân sự dựa trên document blocks.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: tuyển dụng, C&B, KPI/OKR, đào tạo, Bộ luật Lao động, "
        "nội quy TDI, onboarding, đánh giá nhân viên.\n"
        "- Không trả lời: thông tin cá nhân nhân viên, lương cá nhân, "
        "tranh chấp đang diễn ra.\n\n"
        "THUẬT NGỮ CHUẨN:\n"
        "- C&B (Compensation & Benefits), không gọi là 'lương và phụ cấp'.\n"
        "- OKR ≠ KPI (hai khung đo khác nhau).\n"
        "- Thử việc, không gọi là 'học việc'.\n"
        "- BHXH – BHYT – BHTN ghi đầy đủ, không gộp chung."
    ),
    "tai_chinh": (
        "Bạn là Trợ lý đọc tài liệu Tài chính của TDI Group. Nhiệm vụ: trả lời "
        "câu hỏi tài chính dựa trên document blocks.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: BCTC, dòng tiền, NPV/IRR, ROI, ngân sách, kế toán "
        "quản trị, thuế TNDN/GTGT, VAS/IFRS, kiểm soát nội bộ.\n"
        "- Không trả lời: tư vấn đầu tư cá nhân, giá cổ phiếu tương lai, "
        "thông tin tài chính mật chưa công bố.\n\n"
        "THUẬT NGỮ CHUẨN:\n"
        "- Doanh thu ≠ Lợi nhuận ≠ Dòng tiền (ba khái niệm tách biệt).\n"
        "- EBITDA, không gọi là 'lợi nhuận ròng'.\n"
        "- NPV (Net Present Value), không dịch chung chung là 'giá trị hiện tại'.\n"
        "- VAS (Vietnamese Accounting Standards), không gọi là 'chuẩn mực kế toán' chung chung."
    ),
    "kinh_doanh": (
        "Bạn là Trợ lý đọc tài liệu Kinh doanh của TDI Group. Nhiệm vụ: trả "
        "lời câu hỏi kinh doanh dựa trên document blocks.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: quy trình bán hàng, pipeline, KAM, chính sách giá, "
        "hoa hồng, hợp đồng khung, KPI kinh doanh, sales forecast.\n"
        "- Không trả lời: kỹ thuật công trình, pháp lý hợp đồng chi tiết, "
        "tư vấn cá nhân cho khách hàng ngoài TDI.\n\n"
        "THUẬT NGỮ CHUẨN:\n"
        "- Pipeline, không dịch là 'danh sách khách hàng'.\n"
        "- KAM (Key Account Management), không dịch là 'chăm sóc khách VIP'.\n"
        "- Closing rate, không dịch chung chung là 'tỷ lệ chốt'.\n"
        "- Forecast, không dịch là 'dự báo miệng'."
    ),
    "thiet_ke": (
        "Bạn là Trợ lý đọc tài liệu Thiết kế của TDI Group. Nhiệm vụ: trả "
        "lời câu hỏi thiết kế dựa trên document blocks.\n\n"
        "PHẠM VI:\n"
        "- Được trả lời: kiến trúc, quy hoạch, nội thất, concept design, "
        "công năng, QCVN/TCVN thiết kế, AutoCAD/Revit/SketchUp, material board.\n"
        "- Không trả lời: tính toán kết cấu, MEP chi tiết, tài chính dự án.\n\n"
        "THUẬT NGỮ CHUẨN:\n"
        "- Concept design, không dịch là 'ý tưởng sơ bộ' chung chung.\n"
        "- Schematic design, không dịch là 'bản vẽ mẫu'.\n"
        "- Shop drawing ≠ bản vẽ thi công (hai giai đoạn khác nhau).\n"
        "- Mặt bằng công năng, không gọi là 'mặt bằng bố trí'."
    ),
}


# Nguồn chân lý cho tập domain cả hệ thống (chat persona + metadata classifier).
# metadata_generator import DOMAIN_KEYS này và assert cùng tập khi module load,
# để khi thêm/bớt persona ở đây, classifier không silently drift.
DOMAIN_KEYS: tuple[str, ...] = tuple(DOMAIN_PERSONAS.keys())


# ---------------------------------------------------------------- base rules
# Rules ổn định cross-turn, đặt trong system prompt + cache_control ephemeral.
# Opus 4.7 literal-instruction-following → rules rõ ràng, tránh mâu thuẫn; mọi
# tham chiếu nguồn (<retrieved_documents>, <user_context>, <session_summary>)
# đều đồng nhất để Claude map đúng tag trong user message.
_BASE_RULES = """
<language_style>
- Trả lời bằng tiếng Việt, lịch sự nhưng thân thiện. Xưng "tôi", gọi người dùng là "bạn".
- Giữ nguyên thuật ngữ chuyên ngành tiếng Anh khi không có bản dịch thông dụng (BIM, LOD, HVAC, KPI…).
- Định dạng số theo kiểu Việt Nam: dấu chấm phân cách hàng nghìn, dấu phẩy cho thập phân (vd: 1.000.000 đồng, 13,3 triệu).
</language_style>

<grounding_rules>
QUY TẮC TUYỆT ĐỐI — không có ngoại lệ:

1. Chỉ được trích fact từ <retrieved_documents> (hoặc document blocks gắn
   citations). Đây là NGUỒN CHÍNH duy nhất cho mọi fact công ty/sản phẩm/
   quy trình/URL/tên hệ thống.

   <session_summary> và <conversation_recall> là BỐI CẢNH PHỤ:
   - Chỉ dùng để hiểu ngữ cảnh hội thoại (tên user, mục tiêu họ TỰ khai báo).
   - CẤM trích fact công ty/sản phẩm/URL/tên hệ thống từ đây — kể cả khi
     summary có vẻ chứa thông tin liên quan, đó có thể là nội dung bot cũ
     hallucinate hoặc keyword khớp tình cờ.
   - Nếu câu hỏi hiện tại có CHỦ ĐỀ KHÁC nội dung summary/recall → BỎ QUA
     toàn bộ summary/recall, chỉ dựa vào <retrieved_documents>.

2. Tuyệt đối KHÔNG dùng kiến thức đào tạo (training data) để:
   - Giải thích định nghĩa, khái niệm, thuật ngữ mà tài liệu không định nghĩa.
   - Liệt kê bước quy trình, công thức, mô hình (4P, PDCA, SWOT, LOD, BEP…)
     nếu chính tài liệu không liệt kê.
   - Đưa ví dụ, con số, mốc thời gian, tên người/tổ chức/sản phẩm.
   - Trả lời "theo best practice ngành", "thông thường...", "thông lệ...".

3. Được phép tính toán số học đơn thuần trên số liệu user/tài liệu đã cung cấp
   (vd: user nói "ngân sách 80 triệu / 6 video" → "≈ 13,3 triệu/video").
   Không được "ước lượng" hay "giả định" số liệu mà nguồn không có.

4. Nếu câu hỏi yêu cầu nội dung NGOÀI các nguồn trên (vd user hỏi khái niệm chung,
   lý thuyết ngành, best practice) và tài liệu không nói đến → kích hoạt
   <refusal_protocol>. KHÔNG được "bù" bằng kiến thức chung.

5. Khi không chắc fact có trong tài liệu hay là do tôi nhớ: chọn refusal.
   Thà refuse đúng hơn là trả lời sai từ training data.
</grounding_rules>

<reasoning_process>
Trước khi viết câu trả lời, thầm xác định (không in ra):
1. Có đoạn nào trong <retrieved_documents> trả lời trực tiếp câu hỏi không?
   Nếu KHÔNG → áp dụng <refusal_protocol>.
2. Fact tôi sắp viết có nguyên văn hoặc phái sinh trực tiếp từ đoạn đã đọc không?
3. Phép tính/tổng hợp tôi sắp làm có dùng CHỈ số liệu trong tài liệu hoặc user đã khai không?
</reasoning_process>

<refusal_protocol>
Kích hoạt khi <retrieved_documents> rỗng, không khớp chủ đề câu hỏi, hoặc
không chứa thông tin đủ để trả lời.

Trả lời đúng template sau (không thêm/bớt):

"Tài liệu TDI hiện chưa có thông tin về [cụm từ ngắn mô tả chủ đề user hỏi].
Bạn có thể:
- Bổ sung tài liệu liên quan qua trang nạp dữ liệu.
- Thử đổi sang lĩnh vực 'Tất cả lĩnh vực' để mở rộng tìm kiếm.
- Diễn đạt lại câu hỏi với từ khoá cụ thể hơn."

Khi refuse: KHÔNG thêm mục "Nguồn:" và KHÔNG thêm khối "---GỢI Ý---".
Tuyệt đối không viết trả lời chung chung từ training data kiểu "Thông thường, về X thì...".

NGOẠI LỆ — câu meta (chào hỏi, hỏi danh tính bạn/tôi, hỏi capability, cảm ơn, tạm biệt):
- KHÔNG áp refusal template ở trên. Trả lời ngắn gọn 1-2 câu theo persona,
  không kèm "Nguồn:" và không kèm "---GỢI Ý---".
- Lưu ý: pipeline có intent gate ở pre-retrieval đã short-circuit phần lớn
  case này. Nếu lọt tới đây nghĩa là gate miss biến thể → vẫn xử lý đúng tone.
</refusal_protocol>

<answer_format>
Chọn format theo dạng câu hỏi:
- Câu hỏi quy trình → trình bày theo Bước 1 / Bước 2 / Bước 3.
- Câu hỏi thông số kỹ thuật → bảng hoặc danh sách có đơn vị rõ ràng.
- Câu hỏi sự cố → chia 3 mục: Nguyên nhân / Xử lý / Phòng tránh.
- Câu hỏi khái niệm / xã giao → trả lời tự nhiên theo văn phong thường, bỏ mục "Nguồn:" và khối "---GỢI Ý---".
</answer_format>

<citation_rules>
- Trong phần nội dung: trích nguồn bằng TÊN tài liệu (vd: "(CHÂN DUNG ĐỐI TƯỢNG...)", không dùng "[NGUỒN 1]").
- Nếu câu trả lời có dùng tài liệu: kết thúc bằng đúng 1 dòng "Nguồn:" liệt kê tên tài liệu — bắt buộc kèm URL ngay sau tên nếu trong <source> có URL, dạng: "Tên tài liệu — https://...". Cấm bỏ URL.
- Nếu câu trả lời chỉ dùng dữ kiện user (không dùng tài liệu): ghi "Nguồn: Thông tin bạn đã cung cấp trong cuộc trò chuyện."
- Với câu xã giao (chào hỏi, cảm ơn, giới thiệu bản thân, trò chuyện thường): trả lời tự nhiên, BỎ mục "Nguồn:" và BỎ khối "---GỢI Ý---".
</citation_rules>

<followup_suggestions>
Khi câu trả lời có nội dung kiến thức, sau mục "Nguồn:" thêm đúng khối sau:
---GỢI Ý---
1. <câu hỏi 1>
2. <câu hỏi 2>
3. <câu hỏi 3>

QUY TẮC TUYỆT ĐỐI — gợi ý PHẢI trả lời được bằng chính <retrieved_documents>
vừa đọc, không được "gợi ý xuông":
- Mỗi câu hỏi gợi ý phải hỏi về MỘT chi tiết, bước, thông số, khái niệm HOẶC
  phần phụ đã xuất hiện nguyên văn trong <retrieved_documents>. Nếu không
  chắc trong docs có câu trả lời → KHÔNG đưa câu đó vào gợi ý.
- CẤM gợi ý câu hỏi mở rộng ra lĩnh vực khác, so sánh với sản phẩm/quy trình
  không xuất hiện trong docs, hoặc hỏi chung chung (vd "còn cách nào khác
  không?", "có lưu ý gì không?") nếu docs không nêu cụ thể.
- CẤM gợi ý câu hỏi mà câu trả lời vừa viết đã trả lời xong — tránh lặp.
- Nếu docs chỉ chứa 0-1 ý đủ để gợi ý: cho phép xuất 0-2 câu thay vì ép đủ 3.
  Thà thiếu còn hơn bịa câu không trả lời được.
- Ngắn gọn, xưng "tôi" như user thật hỏi (không phải bot nói về bot).
- Tự kiểm tra lần cuối trước khi xuất: với từng câu gợi ý, thầm tìm đoạn
  trong <retrieved_documents> có thể trả lời nó — nếu không thấy, bỏ câu đó.
</followup_suggestions>
"""


def build_system_prompt(expert_domain: str | None = None) -> str:
    """System prompt STABLE cross-turn — cache được qua cache_control ephemeral.

    Bao gồm persona (role + scope + terminology) và _BASE_RULES. KHÔNG chèn
    documents/conv_block ở đây — chúng động, phải đi vào user message qua
    ``build_user_turn`` để tránh cache write premium mỗi turn.
    """
    domain = (expert_domain or "mac_dinh").lower().strip()
    # Fallback: chấp nhận persona VN cũ (backward compat) qua PERSONA_TO_DOMAIN.
    from app.core.qdrant_store import PERSONA_TO_DOMAIN
    if domain not in DOMAIN_PERSONAS and domain in PERSONA_TO_DOMAIN:
        domain = PERSONA_TO_DOMAIN[domain]
    persona = DOMAIN_PERSONAS.get(
        domain,
        f"Bạn là chuyên gia về '{expert_domain}'. Trả lời tiếng Việt chính xác, có nguồn rõ ràng.",
    )
    return persona.strip() + "\n" + _BASE_RULES.strip()


# ---------------------------------------------------------------- conversation
def _fmt_relative_time(created_at: int) -> str:
    import time as _t
    if not created_at:
        return ""
    delta = int(_t.time()) - int(created_at)
    if delta < 3600:
        return f"{max(1, delta // 60)} phút trước"
    if delta < 86400:
        return f"{delta // 3600} giờ trước"
    return f"{delta // 86400} ngày trước"


def build_conversation_block(summary: str, recall_pairs: list[dict]) -> str:
    """XML block <conversation_recall> + <session_summary> cho USER message.

    Phase 3.1 đổi tag từ <user_context> → <conversation_recall> kèm framing
    "low priority — DO NOT cite làm fact". Lý do: bug BKVN có nguyên nhân
    Claude treat recall pair (chứa từ "BKVN") ngang hàng với
    <retrieved_documents>, kéo từ khoá lạc đề thành fact trong câu trả lời.

    Tách khỏi system (cache) vì nội dung đổi mỗi turn. Trả "" nếu rỗng cả hai.
    """
    parts: list[str] = []

    if recall_pairs:
        lines = [
            "<conversation_recall>",
            "BỐI CẢNH PHỤ — các trao đổi trước giữa user này và bạn (truy xuất "
            "theo độ tương đồng ngữ nghĩa với câu hỏi hiện tại).",
            "QUY TẮC dùng block này:",
            "- Đây KHÔNG phải <retrieved_documents>. KHÔNG cite làm nguồn fact.",
            "- Chỉ dùng để hiểu ngữ cảnh user (tên, vai trò, mục tiêu user TỰ "
            "khai báo). Tuyệt đối không lấy nội dung BOT trả lời cũ làm fact.",
            "- Nếu câu hỏi hiện tại chủ đề khác hẳn nội dung block này → BỎ QUA.",
            "",
        ]
        for i, p in enumerate(recall_pairs, 1):
            ts = _fmt_relative_time(p.get("created_at", 0))
            tag = f"[#{i}" + (f" — {ts}" if ts else "") + "]"
            lines.append(tag)
            lines.append(p.get("text", "").strip())
            lines.append("")
        lines.append("</conversation_recall>")
        parts.append("\n".join(lines))

    if summary and summary.strip():
        parts.append(
            "<session_summary>\n"
            "BỐI CẢNH PHỤ — tóm tắt các lượt hội thoại TRƯỚC trong phiên này "
            "(đã rớt khỏi sliding window).\n"
            "QUY TẮC dùng block này:\n"
            "- KHÔNG cite làm nguồn fact công ty/sản phẩm/URL.\n"
            "- Chỉ dùng để hiểu ngữ cảnh user (mục tiêu user tự nêu, "
            "tên/vai trò họ tự khai).\n"
            "- Nếu câu hỏi hiện tại chủ đề khác hẳn nội dung này → BỎ QUA.\n"
            "- Tuyệt đối KHÔNG suy luận hệ thống/sản phẩm/URL từ keyword "
            "trong summary — keyword có thể đã được nhắc trong context khác.\n"
            "\n"
            f"{summary.strip()}\n"
            "</session_summary>"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------- documents
def _parse_timestamp(ts) -> tuple[int | None, str | None]:
    if ts is None:
        return None, None
    ts_str = str(ts)
    try:
        if ":" in ts_str:
            parts = ts_str.split(":")
            secs = int(parts[0]) * 60 + int(parts[1])
            return secs, ts_str
        secs = int(float(ts_str))
        return secs, f"{secs // 60}:{secs % 60:02d}"
    except (ValueError, TypeError):
        return None, str(ts)


def _build_youtube_url_with_timestamp(base_url: str, seconds: int | None) -> str:
    if not base_url or seconds is None or seconds <= 0:
        return base_url or ""
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}t={seconds}s"


def _resolve_source_fields(payload: dict) -> dict:
    title = (
        payload.get("title")
        or payload.get("source_name")
        or payload.get("filename")
        or payload.get("source", "Không rõ nguồn")
    )
    page = payload.get("page")
    raw_ts = payload.get("start") or payload.get("timestamp")
    base_url = (
        payload.get("url")
        or payload.get("source_url")
        or payload.get("source")
        or payload.get("youtube_url")
        or ""
    )
    ts_secs, ts_display = _parse_timestamp(raw_ts)
    return {
        "title": title,
        "page": page,
        "base_url": base_url,
        "ts_secs": ts_secs,
        "ts_display": ts_display,
    }


def build_documents_block(hits: list[Hit]) -> str:
    """<retrieved_documents> XML — đặt vào user message ở TOP (long-context tip).

    Trả "" nếu không có hit nào.

    LEGACY: dùng cho path không bật Citations API. Khi Citations bật, dùng
    ``build_documents_blocks()`` (số nhiều) trả list document blocks native.
    """
    if not hits:
        return ""

    parts: list[str] = ["<retrieved_documents>"]
    for n, hit in enumerate(hits, 1):
        fields = _resolve_source_fields(hit.payload)

        source_parts = [fields["title"]]
        if fields["base_url"]:
            source_parts.append(
                _build_youtube_url_with_timestamp(fields["base_url"], fields["ts_secs"])
            )
        if fields["page"] is not None:
            source_parts.append(f"trang {fields['page']}")
        if fields["ts_display"] is not None:
            source_parts.append(fields["ts_display"])
        source_line = " — ".join(source_parts)

        parts.append(f'  <document index="{n}">')
        parts.append(f"    <source>{source_line}</source>")
        parts.append("    <content>")
        parts.append(hit.text.strip())

        table_data = hit.payload.get("table_data", "")
        if table_data:
            parts.append("")
            parts.append("Dữ liệu bảng chi tiết:")
            parts.append(table_data.strip())

        parts.append("    </content>")
        parts.append("  </document>")

    parts.append("</retrieved_documents>")
    return "\n".join(parts)


import re as _re

# Phase 4.2 — defensive sanitization patterns. Tài liệu nội bộ TDI thường
# sạch, nhưng nếu sau này index PDF/HTML từ nguồn ngoài có thể chứa
# instructions cố tình hoặc vô tình. Strip trước khi đưa vào Citations API
# để Claude không bị adversarial-prompted qua document blocks.
_PROMPT_INJECTION_PATTERNS = [
    # System prompt markers
    _re.compile(r"\[\s*(SYSTEM|ASSISTANT|USER|HUMAN)\s*\]", _re.IGNORECASE),
    _re.compile(r"</?\s*(system|assistant|user|human)\s*>", _re.IGNORECASE),
    # Anthropic/OpenAI special tokens
    _re.compile(r"<\|[^>]+\|>"),
    _re.compile(r"\bclaude:\s|\bassistant:\s|\bsystem:\s", _re.IGNORECASE),
    # Common jailbreak phrases (English + Vietnamese)
    _re.compile(
        r"\b(ignore|disregard|forget)\s+(all|the|previous|prior|above)\s+"
        r"(instructions?|rules?|prompts?|directives?)",
        _re.IGNORECASE,
    ),
    _re.compile(
        r"\b(bỏ qua|phớt lờ|quên đi)\s+(tất cả|toàn bộ|mọi|các)?\s*"
        r"(hướng dẫn|chỉ dẫn|quy tắc|mệnh lệnh)",
        _re.IGNORECASE,
    ),
    # Role override attempts
    _re.compile(r"you are now\s+", _re.IGNORECASE),
    _re.compile(r"new role:?\s*", _re.IGNORECASE),
]


def _sanitize_document_text(text: str) -> str:
    """Strip prompt-injection patterns khỏi document text trước khi đưa vào Claude.

    Defensive layer — bình thường docs nội bộ sạch, nhưng safety-in-depth
    chặn case adversarial document indexing (vd PDF download có hidden
    instruction trong metadata).

    Replace pattern bằng "[redacted]" thay vì xoá hẳn → vẫn giữ vị trí
    cho Citations API char-range tracking, không làm vỡ alignment.
    """
    if not text:
        return text
    out = text
    for pat in _PROMPT_INJECTION_PATTERNS:
        out = pat.sub("[redacted]", out)
    return out


def build_documents_blocks(hits: list[Hit]) -> list[dict]:
    """Anthropic Citations API — mỗi hit thành 1 ``document`` block.

    Claude tự động cite vào char-range của từng block khi trả lời.
    Không có hit → trả [] (caller tự xử lý: thường refuse).

    Lưu ý:
      - ``citations.enabled = True`` BẮT BUỘC để API trả citations[].
      - ``title`` hiện ra trong response.content[*].citations[*].document_title.
      - ``context`` là metadata phụ Claude đọc nhưng KHÔNG cite được —
        nhét URL/score/page vào đây để debug, không ảnh hưởng generation.
      - Không gắn cache_control: docs đổi mỗi turn, cache write = phí thuần.
      - table_data (nếu có) append vào source.data sau text — Claude vẫn
        cite được vì nằm trong cùng document.
    """
    blocks: list[dict] = []
    for hit in hits:
        fields = _resolve_source_fields(hit.payload)
        text = hit.text.strip()
        table_data = hit.payload.get("table_data", "")
        if table_data:
            text = f"{text}\n\nDữ liệu bảng chi tiết:\n{table_data.strip()}"
        # Phase 4.2: defensive sanitize trước khi gửi vào Citations API
        text = _sanitize_document_text(text)

        ctx_parts: list[str] = []
        if fields["base_url"]:
            ctx_parts.append(
                f"url={_build_youtube_url_with_timestamp(fields['base_url'], fields['ts_secs'])}"
            )
        if fields["page"] is not None:
            ctx_parts.append(f"page={fields['page']}")
        if fields["ts_display"] is not None:
            ctx_parts.append(f"timestamp={fields['ts_display']}")
        if hit.score is not None:
            ctx_parts.append(f"rerank_score={hit.score:.4f}")

        blocks.append(
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": text,
                },
                "title": str(fields["title"])[:500],  # Anthropic giới hạn ~500 chars
                "context": " | ".join(ctx_parts) if ctx_parts else "",
                "citations": {"enabled": True},
            }
        )
    return blocks


_EXCERPT_MAX_CHARS = 320


def _build_excerpt(text: str) -> str:
    """Cắt trích đoạn hiển thị ở FE: collapse whitespace + giới hạn độ dài."""
    if not text:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= _EXCERPT_MAX_CHARS:
        return normalized
    return normalized[:_EXCERPT_MAX_CHARS].rstrip() + "…"


def build_sources_from_citations(
    hits: list[Hit], citations: list[dict]
) -> list[dict]:
    """Source mapping bám đúng từng citation Claude trả về.

    Khác với cách cũ (cắt 320 ký tự đầu của cả chunk), hàm này iterate theo
    từng citation và lấy ``cited_text`` — đoạn nguyên văn Anthropic Citations
    API khẳng định Claude đã đọc — làm excerpt cho mỗi position. Nhờ vậy FE
    hiển thị đúng câu được dùng trả lời, neo theo trang/timestamp của chunk
    chứa câu đó.

    Dedup: nhiều citation cùng trỏ về (source, page/timestamp, cited_text)
    sẽ gộp thành 1 position duy nhất.
    """
    seen: dict[str, dict] = {}
    for c in citations:
        idx = c.get("doc_index")
        if not isinstance(idx, int) or not (0 <= idx < len(hits)):
            continue
        hit = hits[idx]
        fields = _resolve_source_fields(hit.payload)
        cited_raw = (c.get("cited_text") or "").strip()
        # Fallback hiếm: API trả cited_text rỗng → dùng excerpt cả chunk để
        # vẫn có gì đó hiển thị dưới pill thay vì khoảng trống.
        cited_excerpt = (
            _build_excerpt(cited_raw) if cited_raw else _build_excerpt(hit.text)
        )

        key = f"{fields['title']}||{fields['base_url']}"
        if key not in seen:
            seen[key] = {
                "source_type": hit.source_type,
                "title": fields["title"],
                "base_url": fields["base_url"],
                "page": fields["page"],
                "timestamp": fields["ts_display"],
                "timestamp_secs": fields["ts_secs"],
                "score": hit.score,
                "excerpt": cited_excerpt,
                "positions": [],
                "_seen_pos": set(),
            }
        entry = seen[key]
        if hit.score > entry["score"]:
            entry["score"] = hit.score

        if fields["ts_display"] is not None:
            position: dict = {
                "timestamp": fields["ts_display"],
                "url": _build_youtube_url_with_timestamp(
                    fields["base_url"], fields["ts_secs"]
                ),
                "excerpt": cited_excerpt,
            }
        elif fields["page"] is not None:
            position = {"page": fields["page"], "excerpt": cited_excerpt}
        else:
            position = {"excerpt": cited_excerpt}

        dedup_key = (
            position.get("timestamp"),
            position.get("page"),
            position["excerpt"],
        )
        if dedup_key not in entry["_seen_pos"]:
            entry["_seen_pos"].add(dedup_key)
            entry["positions"].append(position)

    mapping: list[dict] = []
    for idx, entry in enumerate(seen.values(), 1):
        entry.pop("_seen_pos", None)
        first_pos_url = (
            entry["positions"][0]["url"]
            if entry["positions"] and "url" in entry["positions"][0]
            else entry["base_url"]
        )
        mapping.append(
            {
                "index": idx,
                "source_type": entry["source_type"],
                "title": entry["title"],
                "url": first_pos_url or entry["base_url"],
                "page": entry.get("page"),
                "timestamp": entry.get("timestamp"),
                "score": entry["score"],
                "excerpt": entry.get("excerpt", ""),
                "positions": entry["positions"],
            }
        )
    return mapping


# ---------------------------------------------------------------- user turn
# Vì system prompt (rules) được Claude xử lý TRƯỚC user message, task reminder
# ngắn này giúp neo lại ý định "answer from <retrieved_documents> above" — rẻ
# (~40 tokens) và củng cố grounding khi user ở turn sâu đã có docs trong user turn
# mà không còn thấy rules ở đầu prompt.
_TASK_REMINDER = (
    "<task>\n"
    "Dựa CHỈ vào <retrieved_documents> và <user_context>/<session_summary> ở trên "
    "(cùng các quy tắc trong system prompt), trả lời câu hỏi sau. "
    "Nếu tài liệu không đủ thông tin → dùng <refusal_protocol>.\n"
    "</task>"
)

# Hint khi retriever trả hit nhưng top-score thấp — thay cho hard-refuse cũ.
# Claude đọc hint + áp <refusal_protocol> nếu tài liệu thực sự không liên quan,
# hoặc trả lời cẩn trọng nếu có phần nào khớp (vd: title match nhưng chunk ít nội dung).
_LOW_CONFIDENCE_HINT = (
    "<retrieval_confidence>low</retrieval_confidence>\n"
    "LƯU Ý: độ tương đồng giữa câu hỏi và tài liệu truy xuất đang THẤP. "
    "Kiểm tra kỹ <retrieved_documents> có thật sự trả lời được câu hỏi không — "
    "nếu không, áp dụng <refusal_protocol>. Không suy đoán từ kiến thức chung."
)


def build_user_turn(
    query: str,
    docs_block: str,
    conv_block: str,
    low_confidence: bool = False,
) -> str:
    """Ráp user message cuối theo pattern long-context của Anthropic.

    Thứ tự: documents (top) → conversation context → [low-conf hint] → task
    reminder → query (bottom). Query ở cuối tăng chất lượng tới ~30% (per
    Anthropic long-context tips).

    LEGACY (non-Citations path). Khi bật Citations API → ``build_user_content()``.
    """
    parts: list[str] = []
    if docs_block:
        parts.append(docs_block)
    if conv_block:
        parts.append(conv_block)
    # Chỉ gắn reminder khi có context phía trên — với câu xã giao không retrieve
    # thì không cần, tránh buộc refusal sai.
    if docs_block or conv_block:
        if low_confidence:
            parts.append(_LOW_CONFIDENCE_HINT)
        parts.append(_TASK_REMINDER)
    parts.append(f"Câu hỏi của tôi: {query.strip()}")
    return "\n\n".join(parts)


# Task reminder cho Citations API path — khác bản XML cũ vì docs giờ là
# native ``document`` blocks chứ không phải <retrieved_documents> XML.
# Reminder này nhấn: cite từng claim vào document blocks ở trên, từ chối
# nếu không tìm được supporting passage.
_TASK_REMINDER_CITATIONS = (
    "<task>\n"
    "Trả lời câu hỏi sau dựa CHỈ vào các tài liệu (document blocks) ở đầu "
    "tin nhắn này và <user_context>/<session_summary> (nếu có). Mọi fact "
    "trong câu trả lời PHẢI được hỗ trợ bởi đoạn cụ thể trong document — "
    "Claude sẽ tự động cite. Nếu không có đoạn nào trả lời được câu hỏi → "
    "áp dụng <refusal_protocol> (KHÔNG được bịa URL, tên hệ thống, hay "
    "thông tin không xuất hiện trong tài liệu, kể cả khi câu hỏi gợi ý).\n"
    "</task>"
)


def build_user_identity_block(user_profile: dict | None) -> str:
    """XML block <user_identity> — auth context (tên/vai trò user đã đăng nhập).

    Backend (NestJS) decode JWT → IUser.fullName/role → forward qua ChatRequest
    → chain.py truyền profile xuống đây. AI module KHÔNG persist, mỗi turn nhận
    fresh.

    Trả "" nếu profile rỗng (user chưa đăng nhập / dev mode).

    Lý do tách block riêng (không trộn với <user_context>): đây là FACT từ auth
    server, được phép cite/dùng để xưng hô. Khác với conv_recall là "có thể bot
    cũ hallucinate".
    """
    if not user_profile:
        return ""
    name = (user_profile.get("name") or "").strip()
    role = (user_profile.get("role") or "").strip()
    if not name and not role:
        return ""

    lines = [
        "<user_identity>",
        "Người dùng đang trò chuyện đã đăng nhập với hệ thống. Thông tin này "
        "ĐÚNG (từ auth server, không phải bot cũ):",
    ]
    if name:
        lines.append(f"- Tên: {name}")
    if role:
        lines.append(f"- Vai trò: {role}")
    lines.append(
        "Khi user hỏi về danh tính của họ ('tôi là ai', 'bạn có biết tôi'...) "
        "→ trả lời bằng tên/vai trò trên. Khi xưng hô, dùng tên đầy đủ hoặc "
        "tên gọi ngắn nếu phù hợp ngữ cảnh."
    )
    lines.append("</user_identity>")
    return "\n".join(lines)


def build_user_content(
    query: str,
    doc_blocks: list[dict],
    conv_block: str,
    low_confidence: bool = False,
    user_profile: dict | None = None,
) -> list[dict]:
    """Ráp user message dạng list content cho Anthropic Citations API.

    Output:
        [
          {type:document, citations:{enabled:true}, ...},   # hit 1
          {type:document, ...},                              # hit 2
          ...
          {type:text, text: "<user_identity>...<conv>...<task>... Câu hỏi: ..."}
        ]

    Document blocks ở TOP (long-context tip) — Anthropic cite được tới
    char-range cụ thể trong từng block. user_identity (auth context) + conv
    block + task reminder + query đi vào 1 text block ở cuối.
    """
    text_parts: list[str] = []
    identity_block = build_user_identity_block(user_profile)
    if identity_block:
        text_parts.append(identity_block)
    if conv_block:
        text_parts.append(conv_block)
    if doc_blocks or conv_block:
        if low_confidence:
            text_parts.append(_LOW_CONFIDENCE_HINT)
        # Dùng reminder phiên bản Citations (yêu cầu cite vào document blocks)
        text_parts.append(_TASK_REMINDER_CITATIONS if doc_blocks else _TASK_REMINDER)
    text_parts.append(f"Câu hỏi của tôi: {query.strip()}")
    text_block = {"type": "text", "text": "\n\n".join(text_parts)}

    return [*doc_blocks, text_block]
