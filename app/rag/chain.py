from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from collections.abc import Iterator
from typing import Any

from app.config import CLAUDE_HAIKU_MODEL
from app.core.claude_client import ClaudeClient
from app.core.conv_memory import ConversationMemory
from app.core.conv_query_rewriter import rewrite as rewrite_query
from app.rag.faithfulness import check_faithfulness
from app.rag.intent_gate import (
    canned_response,
    classify_intent,
    is_meta_intent,
)
from app.rag.retriever import Retriever
from app.rag.reranker import CrossEncoderReranker
from app.rag.prompt_builder import (
    build_system_prompt,
    build_documents_blocks,
    build_sources_mapping,
    build_conversation_block,
    build_user_content,
)

logger = logging.getLogger(__name__)

_SUGGESTION_PATTERN = re.compile(
    # Chấp nhận các biến thể marker mà Opus/Sonnet có thể đẻ ra khi không
    # tuân thủ literal 100%: "---GỢI Ý---", "--- Gợi ý ---", "**Gợi ý:**",
    # "## Gợi ý", "Gợi ý câu hỏi:"... Luôn yêu cầu marker đứng riêng 1 dòng
    # (có newline trước + sau) để tránh false-positive với "gợi ý" trong body.
    r"\n[ \t]*"                                    # đầu dòng (newline + indent)
    r"[\*#\-=_|: \t]*"                             # decorator mở (tuỳ chọn)
    r"G[ỢợOo][Ii][ \t]*[ÝýYy]"                    # "Gợi ý" / "GỢI Ý" (diacritic flex)
    r"(?:[ \t]+(?:c[âấ]u[ \t]*h[ỏỎ]i|tiếp[ \t]*theo|theo[ \t]*dõi))?"  # optional suffix
    r"[\*#\-=_|:\. \t]*"                           # decorator đóng (tuỳ chọn)
    r"\n+"
    r"(\s*\d+[\.\)].*)",                           # content phải bắt đầu "1. " hoặc "1) "
    re.IGNORECASE | re.DOTALL,
)

# Fallback: LLM quên hẳn marker nhưng vẫn liệt kê 2-5 câu hỏi đánh số ở cuối.
# Yêu cầu: đứng sau dòng trống + là khối cuối cùng của answer.
_TRAILING_NUMBERED_PATTERN = re.compile(
    r"\n\s*\n((?:[ \t]*\d+[\.\)][ \t]+[^\n]+(?:\n|$)){2,5})\s*$",
    re.DOTALL,
)

# Refusal/confidence policy (Phase 2 — sigmoid scale).
# Reranker giờ trả sigmoid score ∈ [0,1]. Threshold mới:
#
#   sigmoid >= 0.7  → high confidence (Claude trả lời tự nhiên)
#   sigmoid 0.4-0.7 → medium (default — Claude tự cẩn trọng qua _BASE_RULES)
#   sigmoid < 0.6   → chèn _LOW_CONFIDENCE_HINT vào user_turn (gentle warning)
#   sigmoid < 0.4   → HARD REFUSE pre-Claude (tiết kiệm 1 Claude call,
#                     tránh hallucinate "stretch context")
#
# Rationale: với BGE v2-m3, sigmoid 0.4 ≈ raw logit -0.4 → "rõ ràng không
# liên quan". Trước đây giữ Claude tự refuse mềm dẫn đến bug BKVN —
# Claude vẫn cố trả lời từ chunk top-1 yếu rồi bịa fact.
_LOW_CONFIDENCE_SCORE: float = 0.6
_HARD_REFUSE_SCORE: float = 0.4

_REFUSAL_TEMPLATE = (
    "Tài liệu TDI hiện chưa có thông tin về câu hỏi này.\n\n"
    "Bạn có thể:\n"
    "- Bổ sung tài liệu liên quan qua trang nạp dữ liệu.\n"
    "- Thử đổi sang lĩnh vực 'Tất cả lĩnh vực' để mở rộng tìm kiếm.\n"
    "- Diễn đạt lại câu hỏi với từ khoá cụ thể hơn."
)


# Ngưỡng score tối thiểu để một nguồn được hiển thị ở phía dưới câu trả lời (FE).
# Giúp loại bỏ các nguồn "cố đấm ăn xôi" có điểm thấp nhưng vẫn lọt vào context.
_MIN_SOURCE_SCORE_TO_SHOW: float = 0.2
_MAX_SOURCES_TO_SHOW: int = 2

# Citations API gating — Phase 1 anti-hallucination layer.
# Khi Claude trả lời câu KIẾN THỨC dài (>= ngưỡng) nhưng KHÔNG cite chunk
# nào → nhiều khả năng hallucinate. Replace bằng refusal template.
# Ngưỡng 100 chars nhỏ đủ để câu xã giao ("Chào bạn!") không bị refuse oan.
_MIN_ANSWER_LEN_REQUIRE_CITATION: int = 100
# Ngưỡng tỉ lệ câu không-cite tối đa cho phép. Tính trên text blocks: nếu
# > 60% câu không có citation kèm → coi như hallucinate. 0.6 lỏng vì câu
# tiếng Việt ngắn + intro/transitions chính đáng không cần cite.
_MAX_UNCITED_SENTENCE_RATIO: float = 0.6

# Marker để nhận biết answer đang là refusal template (đã viết bởi Claude
# hoặc inject thủ công) — KHÔNG enforce citation requirement nữa.
_REFUSAL_PREFIX = "Tài liệu TDI hiện chưa có thông tin"

# Regex tìm URL trong answer (dùng cho URL fabrication check — kỹ thuật
# deterministic detect hallucinate URL không cần gọi LLM judge).
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"')\]]+",
    re.IGNORECASE,
)


# Patterns nhận diện fact cụ thể (số liệu, ngày, URL, chuẩn, tham chiếu pháp luật).
# Khi answer match >=1 pattern → có claim → faithfulness gate phải verify.
# Khi answer KHÔNG match → response thuần xã giao/lễ phép → trust LLM, skip gate.
#
# Lý do thiết kế: gate cũ trip khi `citations==0 AND len(answer)>100` — bị false
# positive trên MỌI câu xã giao dài (chào hỏi có persona, intro tên user, capability
# description...). Bug này xuất hiện cho mọi user, không enumerate hết được bằng
# intent regex. Fix gốc rễ: chỉ trip khi answer chứa CLAIM cần grounding.
#
# Trade-off: pattern bỏ sót sẽ lọt soft-hallucination dạng "TDI là tập đoàn lớn
# nhất VN" (không số/ngày). Chấp nhận vì alternative (always-trip) gây UX bug
# nghiêm trọng hơn. Layer 1 URL fabrication vẫn chạy độc lập với mọi answer.
_FACTUAL_CLAIM_PATTERNS: list[re.Pattern] = [
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\b(19|20)\d{2}\b"),                                # Years 1900-2099
    re.compile(r"\d+([.,]\d+)?\s*(%|‰)"),                            # Percentages
    re.compile(
        r"\d+([.,]\d+)?\s*"
        r"(triệu|tỷ|nghìn|đồng|usd|eur|vnd|"
        r"km|cm|mm|kg|tấn|m²|m2|m³|m3|ha|"
        r"giờ|phút|giây|ngày|tuần|tháng|năm|"
        r"chi\s+nhánh|nhân\s+viên|cán\s+bộ|dự\s+án|hợp\s+đồng|sản\s+phẩm|khách\s+hàng)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(ISO|TCVN|QCVN|ASTM|NFPA|IEC|EN)\s*\d+", re.IGNORECASE),
    re.compile(r"\b(điều|khoản|điểm|chương|mục|phụ\s+lục)\s+\d+", re.IGNORECASE),
    re.compile(
        r"\b(nghị\s+định|thông\s+tư|quyết\s+định|công\s+văn|luật)\s+(số\s+)?\d",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(thành\s+lập|sáng\s+lập|ra\s+đời|công\s+bố|ban\s+hành|ký\s+kết|"
        r"phê\s+duyệt|khai\s+trương|khánh\s+thành)\b.{0,30}\d",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4,}\b"),                                       # Big numbers (IDs/codes)
]


def _has_factual_claims(text: str) -> bool:
    """True nếu text chứa fact cần grounding (số/ngày/URL/chuẩn/tham chiếu).

    False với câu xã giao thuần ("Chào bạn Tuấn!", "Cảm ơn bạn", "Tôi là Trợ
    lý đọc tài liệu TDI...") — không có gì để hallucinate, cho phép skip gate.
    """
    if not text:
        return False
    return any(p.search(text) for p in _FACTUAL_CLAIM_PATTERNS)


def _extract_urls(text: str) -> list[str]:
    """Lấy tất cả URL trong text. Strip ký tự cuối thường gặp (.,!?;:)."""
    raw = _URL_PATTERN.findall(text or "")
    out: list[str] = []
    for u in raw:
        u = u.rstrip(".,!?;:")
        if u and u not in out:
            out.append(u)
    return out


def _has_fabricated_url(answer: str, hits: list) -> tuple[bool, str]:
    """Detect URL trong answer KHÔNG xuất hiện trong bất kỳ hit text/payload nào.

    Đây là fast deterministic check (không gọi LLM) — catch case Claude bịa
    URL kiểu `bkvn.tdigroup.vn` rất hiệu quả vì URL phải match exact substring.
    """
    answer_urls = _extract_urls(answer)
    if not answer_urls:
        return False, ""
    # Build search corpus: tất cả text + payload URL fields của hits
    corpus_parts: list[str] = []
    for h in hits:
        corpus_parts.append(h.text or "")
        for k in ("url", "source_url", "youtube_url", "source"):
            v = h.payload.get(k) if isinstance(h.payload, dict) else None
            if isinstance(v, str):
                corpus_parts.append(v)
    corpus = "\n".join(corpus_parts)
    for u in answer_urls:
        # Strip protocol để match cả http/https variant
        bare = re.sub(r"^https?://", "", u, flags=re.IGNORECASE).rstrip("/")
        if bare and bare not in corpus:
            return True, u
    return False, ""


def _should_refuse(hits: list) -> bool:
    """True khi retriever rỗng HOẶC top sigmoid < _HARD_REFUSE_SCORE.

    Phase 2 đổi từ "luôn để Claude tự quyết" sang "hard gate khi rerank
    quá yếu". Lý do: với BGE v2-m3 score sigmoid < 0.4 nghĩa là chunk top-1
    thực sự không liên quan — để Claude xử lý chỉ tốn API call và risk
    hallucinate stretch (BKVN bug case).

    Caller bypass-able: ``_HARD_REFUSE_SCORE = 0`` → quay về behaviour cũ.
    """
    if not hits:
        return True
    if hits[0].score < _HARD_REFUSE_SCORE:
        logger.info(
            "hard refuse: top sigmoid=%.4f < %.2f (hits=%d) → skip Claude call",
            hits[0].score, _HARD_REFUSE_SCORE, len(hits),
        )
        return True
    return False


def _looks_like_refusal(text: str) -> bool:
    """Detect Claude tự áp <refusal_protocol> theo template trong _BASE_RULES."""
    return bool(text) and text.lstrip().startswith(_REFUSAL_PREFIX)


def _looks_like_smalltalk(text: str) -> bool:
    """Câu xã giao ngắn (chào hỏi, cảm ơn) — không cần cite, không gate."""
    return bool(text) and len(text.strip()) < _MIN_ANSWER_LEN_REQUIRE_CITATION


def _is_hallucinated_uncited(text: str, citations: list) -> bool:
    """True khi answer dài nhưng cite quá ít → likely hallucinate.

    Logic:
      - Skip nếu là refusal template (Claude tự refuse).
      - Skip nếu là câu xã giao ngắn (< _MIN_ANSWER_LEN_REQUIRE_CITATION).
      - Trip nếu có 0 citation trên answer dài.

    Tỉ lệ câu không-cite chi tiết là KHO sệ tới Phase 3 (HHEM); ở Phase 1
    chỉ cần coarse "0 citations on long answer".
    """
    if _looks_like_refusal(text):
        return False
    if _looks_like_smalltalk(text):
        return False
    return len(citations) == 0


def _build_premise_from_citations(
    hits: list, citations: list[dict], max_chars: int = 6000
) -> str:
    """Ghép text các chunk đã được Claude cite thành 1 premise duy nhất.

    Dùng làm input cho faithfulness gate (Phase 3.3). Premise chỉ chứa text
    thật sự được cite — KHÔNG nhét all hits vào, để gate đánh giá đúng phần
    Claude tuyên bố đã đọc.
    """
    cited_indices: set[int] = set()
    for c in citations:
        idx = c.get("doc_index")
        if isinstance(idx, int) and 0 <= idx < len(hits):
            cited_indices.add(idx)
    if not cited_indices:
        return ""
    parts = [hits[i].text.strip() for i in sorted(cited_indices)]
    joined = "\n\n---\n\n".join(parts)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "…"
    return joined


def _build_sources_from_citations(
    hits: list, citations: list[dict]
) -> list[dict]:
    """Chỉ trả các source thực sự được Claude cite.

    Anthropic ``document_index`` đếm trong content array của user message.
    Vì ``build_user_content()`` đặt N doc blocks ở vị trí 0..N-1 và 1 text
    block cuối, doc_index 0..N-1 map trực tiếp sang hits[0..N-1].

    Nếu không có citation → trả [] (caller sẽ refuse hoặc không show source).
    """
    cited_indices: set[int] = set()
    for c in citations:
        idx = c.get("doc_index")
        if isinstance(idx, int) and 0 <= idx < len(hits):
            cited_indices.add(idx)
    if not cited_indices:
        return []
    cited_hits = [hits[i] for i in sorted(cited_indices)]
    return build_sources_mapping(cited_hits)


def _top_score(hits: list) -> float:
    if not hits:
        return 0.0
    s = hits[0].score
    return s if s is not None else 0.0


def _parse_numbered_questions(raw: str) -> list[str]:
    """Trích câu hỏi từ block kiểu "1. ...\\n2. ...\\n3. ..."."""
    questions: list[str] = []
    for line in raw.splitlines():
        stripped = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        # Strip markdown decorators (bold, italic)
        stripped = re.sub(r"^[\*_`]+|[\*_`]+$", "", stripped).strip()
        if stripped:
            questions.append(stripped)
    return questions


def _extract_suggestions(answer: str) -> tuple[str, list[str]]:
    """Split answer into (clean_answer, suggested_questions).

    Thử 2 chiến lược theo thứ tự:
      1. Marker "GỢI Ý" (nhiều biến thể) + numbered list phía sau.
      2. Fallback: trailing numbered block 2-5 dòng ở cuối answer.

    Nếu cả 2 fail → trả về (answer, []) — LLM không emit gợi ý.
    """
    m = _SUGGESTION_PATTERN.search(answer)
    if m:
        clean = answer[: m.start()].rstrip()
        questions = _parse_numbered_questions(m.group(1))
        if questions:
            logger.debug("suggestions extracted via marker: %d items", len(questions))
            return clean, questions

    # Fallback: không có marker nhưng có block đánh số ở cuối.
    m2 = _TRAILING_NUMBERED_PATTERN.search(answer)
    if m2:
        questions = _parse_numbered_questions(m2.group(1))
        if len(questions) >= 2:
            clean = answer[: m2.start()].rstrip()
            logger.info(
                "suggestions extracted via trailing-numbered fallback: %d items "
                "(LLM bỏ marker — xem lại prompt nếu lặp lại)",
                len(questions),
            )
            return clean, questions

    return answer, []


def _confidence(top_score: float) -> str:
    if top_score >= 0.7:
        return "high"
    if top_score >= 0.4:
        return "medium"
    return "low"


class RAGChain:
    def __init__(
        self,
        retriever: Retriever,
        reranker: CrossEncoderReranker,
        claude: ClaudeClient,
        top_k: int = 10,
        rerank_top_k: int = 3,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.claude = claude
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def answer(
        self,
        query: str,
        history: list[dict] | None = None,
        expert_domain: str | None = None,
        sources_filter: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        summary: str = "",
        conv_memory: ConversationMemory | None = None,
    ) -> dict[str, Any]:
        history = history or []
        t0 = time.perf_counter()

        # --- 1. Query rewrite (Haiku) — 2 chiến lược:
        #   * Anaphora (cần history): resolve đại từ "cái đó/nó/...".
        #   * Short-query expansion (không cần history): mở rộng câu ngắn
        #     thiếu intent ("bài này chill phết" → "Thông tin về bài hát ...").
        # Rewriter tự quyết skip nếu query đủ dài / đủ intent (~1-3s save).
        search_query = query
        try:
            search_query = rewrite_query(self.claude, query, history or [])
        except Exception:
            search_query = query
        t_rewrite = time.perf_counter()

        # --- 1b. Intent gate (pre-retrieval).
        # Greeting/identity/capability không retrieval-able → trả canned response,
        # bypass cả retrieval lẫn Claude lẫn faithfulness gate. Tránh bug stream-rồi-
        # ghi-đè-bằng-refusal khi câu chào dài >100 chars + citations==0.
        intent = classify_intent(query)
        if is_meta_intent(intent):
            logger.info("intent gate hit: %s → canned response (skip retrieval+Claude)", intent)
            return {
                "answer": canned_response(intent, query),
                "sources": [],
                "confidence": "high",
                "suggested_questions": [],
                "rewritten_query": search_query if search_query != query else None,
                "recall_count": 0,
                "intent": intent,
                "refused": False,
            }

        # --- 2. Embed query 1 LẦN, reuse cho cả retriever + conv_memory.
        # Trước fix này, mỗi turn gọi Voyage 2 lần cho cùng 1 query → góp phần
        # đẩy free-tier 3 RPM vào 429 → backoff 25s × N retry.
        query_vec = self.retriever.voyage.embed_query(search_query)
        t_embed = time.perf_counter()

        # --- 3. Parallel: document retrieval + conversation recall
        # Recall gating: skip Qdrant call cho greeting/yes-no — sẽ chỉ recall
        # ra pair noise lạc đề và tốn 1 round-trip Qdrant.
        recall_pairs: list[dict] = []
        do_recall = (
            conv_memory is not None
            and bool(user_id)
            and not ConversationMemory.should_skip_recall(search_query)
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_docs = ex.submit(
                self.retriever.retrieve,
                query=search_query, top_k=self.top_k, domain=expert_domain, sources=sources_filter,
                query_vec=query_vec,
            )
            f_conv = None
            if do_recall:
                f_conv = ex.submit(
                    conv_memory.retrieve,
                    user_id, search_query, session_id,
                    query_vec=query_vec,
                )
            hits = f_docs.result()
            if f_conv is not None:
                try:
                    recall_pairs = f_conv.result()
                except Exception:
                    recall_pairs = []
        t_retrieve = time.perf_counter()

        hits = self.reranker.rerank(search_query, hits, top_k=self.rerank_top_k)
        # Phase 3.2: rerank conv_memory recall với threshold cao (0.75) —
        # chặn pair similar-vector nhưng OFF-TOPIC khỏi đầu độc context
        # (case BKVN: pair "Form thiết kế" recall vào query đăng nhập).
        if recall_pairs:
            recall_pairs = self.reranker.rerank_memory(search_query, recall_pairs)
        t_rerank = time.perf_counter()

        # --- 3b. Guard: không đủ context → refuse cứng, KHÔNG gọi Claude
        # (tránh hallucinate từ training data).
        if _should_refuse(hits):
            logger.info(
                "RAG refused: rewrite=%.2fs embed=%.2fs retrieve=%.2fs "
                "rerank=%.2fs total=%.2fs (recall=%s)",
                t_rewrite - t0, t_embed - t_rewrite, t_retrieve - t_embed,
                t_rerank - t_retrieve, t_rerank - t0,
                "yes" if do_recall else "skipped",
            )
            return {
                "answer": _REFUSAL_TEMPLATE,
                "sources": [],
                "confidence": "low",
                "suggested_questions": [],
                "rewritten_query": search_query if search_query != query else None,
                "recall_count": len(recall_pairs),
                "refused": True,
            }

        # --- 4. Build prompt (Citations API path):
        # - system: persona + _BASE_RULES (STABLE → cache hit mọi turn sau lượt đầu)
        # - user message content: list[document_block × N] + 1 text block (conv +
        #   task + query). Mỗi document block bật ``citations.enabled`` để Claude
        #   tự cite tới char-range cụ thể trong từng chunk.
        system_prompt = build_system_prompt(expert_domain)
        doc_blocks = build_documents_blocks(hits)
        conv_block = build_conversation_block(summary, recall_pairs)

        low_conf = _top_score(hits) < _LOW_CONFIDENCE_SCORE
        user_content = build_user_content(query, doc_blocks, conv_block, low_confidence=low_conf)
        messages = list(history) + [{"role": "user", "content": user_content}]

        result = self.claude.generate_with_citations(
            system_prompt=system_prompt,
            messages=messages,
        )
        answer_text = result["text"]
        citations = result["citations"]
        t_claude = time.perf_counter()

        # --- 4b. Anti-hallucination gates (selective, low false-positive):
        #
        # Layer 1 — Fabricated URL check (deterministic, ~0ms):
        #   Mọi URL trong answer phải xuất hiện trong text/payload của hits.
        #   Catch case BKVN-style: Claude bịa `bkvn.tdigroup.vn`, dù có
        #   citations[] hợp lệ trỏ tới chunk khác.
        #
        # Layer 2 — Faithfulness judge (Haiku, ~500ms):
        #   CHỈ chạy khi citations==0 AND answer dài. Lý do: Citations API
        #   guarantee char-range valid → khi citations present, trust nó.
        #   Khi citations==0, Claude paraphrase nặng → cần judge xác nhận.
        #
        # Tránh always-judge vì gây false-positive refuse trên paraphrase
        # legitimate (user complaint "in ra câu trả lời nhưng cuối cùng
        # ra refusal").
        forced_refusal = False
        forced_reason = ""

        if (
            hits
            and not _looks_like_refusal(answer_text)
        ):
            # Layer 1: URL fabrication — chạy với mọi answer (URL nguy hiểm bất kể độ dài).
            fab, fab_url = _has_fabricated_url(answer_text, hits)
            if fab:
                logger.warning(
                    "URL fabrication detected: %s — not in any hit corpus",
                    fab_url,
                )
                answer_text = _REFUSAL_TEMPLATE
                citations = []
                forced_refusal = True
                forced_reason = f"fabricated_url:{fab_url[:80]}"

            # Layer 2: faithfulness judge — chỉ khi answer có factual claim
            # AND không cite chunk nào. Skip cho câu xã giao thuần (no claim →
            # không có gì để hallucinate, dù LLM tự dệt prose lễ phép).
            elif not citations and _has_factual_claims(answer_text):
                premise = "\n\n---\n\n".join(h.text.strip() for h in hits)
                if len(premise) > 6000:
                    premise = premise[:6000] + "…"
                grounded, reason = check_faithfulness(
                    self.claude, premise, answer_text, judge_model=CLAUDE_HAIKU_MODEL,
                )
                if not grounded:
                    logger.warning(
                        "Faithfulness gate TRIPPED (no-citations path): %s | len=%d",
                        reason, len(answer_text),
                    )
                    answer_text = _REFUSAL_TEMPLATE
                    citations = []
                    forced_refusal = True
                    forced_reason = f"faithfulness:{reason[:80]}"

        clean_answer, suggested_questions = _extract_suggestions(answer_text)

        top_score = hits[0].score if hits else 0.0
        logger.info(
            "RAG done: rewrite=%.2fs embed=%.2fs retrieve=%.2fs rerank=%.2fs "
            "claude=%.2fs total=%.2fs (recall=%s, hits=%d, citations=%d, forced_refuse=%s)",
            t_rewrite - t0, t_embed - t_rewrite, t_retrieve - t_embed,
            t_rerank - t_retrieve, t_claude - t_rerank, t_claude - t0,
            "yes" if do_recall else "skipped", len(hits), len(citations), forced_refusal,
        )

        final_answer = answer_text
        # Only append if we HAVE questions but they WEREN'T found (stripped) from the text
        if suggested_questions and len(clean_answer) >= len(answer_text):
            final_answer += "\n\n--- GỢI Ý ---\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(suggested_questions)])

        # Sources giờ build từ chunk thực sự được Claude cite, không phải all hits.
        # Khi forced_refusal hoặc Claude tự refuse → citations rỗng → sources rỗng.
        # Bỏ filter _MIN_SOURCE_SCORE_TO_SHOW: Citations API ĐÃ verify chunk hữu
        # ích bằng cách Claude thực sự cite vào — rerank score thấp không có
        # nghĩa chunk vô dụng (vd query ngắn cho score thấp dù chunk relevant).
        if citations:
            cited_sources = _build_sources_from_citations(hits, citations)
        else:
            cited_sources = []
        filtered_sources = cited_sources[:_MAX_SOURCES_TO_SHOW]

        return {
            "answer": final_answer,
            "sources": filtered_sources,
            "confidence": _confidence(top_score),
            "suggested_questions": suggested_questions,
            "rewritten_query": search_query if search_query != query else None,
            "recall_count": len(recall_pairs),
            "citations_count": len(citations),
            "forced_refusal": forced_refusal,
        }

    def answer_stream(
        self,
        query: str,
        history: list[dict] | None = None,
        expert_domain: str | None = None,
        sources_filter: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        summary: str = "",
        conv_memory: ConversationMemory | None = None,
    ) -> Iterator[dict[str, Any]]:
        history = history or []
        t0 = time.perf_counter()

        # Rewrite: anaphora (cần history) + short-query expansion (không cần
        # history) — xem chi tiết trong conv_query_rewriter.rewrite.
        search_query = query
        try:
            search_query = rewrite_query(self.claude, query, history or [])
        except Exception:
            search_query = query
        t_rewrite = time.perf_counter()

        # Intent gate (pre-retrieval) — xem comment ở RAGChain.answer().
        # Stream path: emit meta + delta + done với canned text, không gọi Claude.
        intent = classify_intent(query)
        if is_meta_intent(intent):
            logger.info("intent gate hit (stream): %s → canned response", intent)
            text = canned_response(intent, query)
            yield {
                "type": "meta",
                "confidence": "high",
                "rewritten_query": search_query if search_query != query else None,
                "recall_count": 0,
                "intent": intent,
            }
            yield {"type": "delta", "text": text}
            yield {
                "type": "done",
                "answer": text,
                "sources": [],
                "suggested_questions": [],
                "intent": intent,
            }
            return

        # Embed query 1 lần, reuse cho retriever + conv_memory.
        query_vec = self.retriever.voyage.embed_query(search_query)
        t_embed = time.perf_counter()

        recall_pairs: list[dict] = []
        do_recall = (
            conv_memory is not None
            and bool(user_id)
            and not ConversationMemory.should_skip_recall(search_query)
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_docs = ex.submit(
                self.retriever.retrieve,
                query=search_query, top_k=self.top_k, domain=expert_domain, sources=sources_filter,
                query_vec=query_vec,
            )
            f_conv = None
            if do_recall:
                f_conv = ex.submit(
                    conv_memory.retrieve,
                    user_id, search_query, session_id,
                    query_vec=query_vec,
                )
            hits = f_docs.result()
            if f_conv is not None:
                try:
                    recall_pairs = f_conv.result()
                except Exception:
                    recall_pairs = []
        t_retrieve = time.perf_counter()

        hits = self.reranker.rerank(search_query, hits, top_k=self.rerank_top_k)
        # Phase 3.2: rerank conv_memory recall với threshold cao (0.75) —
        # chặn pair similar-vector nhưng OFF-TOPIC khỏi đầu độc context
        # (case BKVN: pair "Form thiết kế" recall vào query đăng nhập).
        if recall_pairs:
            recall_pairs = self.reranker.rerank_memory(search_query, recall_pairs)
        t_rerank = time.perf_counter()
        logger.info(
            "RAG stream pre-llm: rewrite=%.2fs embed=%.2fs retrieve=%.2fs "
            "rerank=%.2fs total=%.2fs (recall=%s, hits=%d)",
            t_rewrite - t0, t_embed - t_rewrite, t_retrieve - t_embed,
            t_rerank - t_retrieve, t_rerank - t0,
            "yes" if do_recall else "skipped", len(hits),
        )

        # Guard: không đủ context → emit refusal events và dừng, không gọi Claude.
        if _should_refuse(hits):
            yield {
                "type": "meta", "confidence": "low",
                "rewritten_query": search_query if search_query != query else None,
                "recall_count": len(recall_pairs),
                "refused": True,
            }
            yield {"type": "delta", "text": _REFUSAL_TEMPLATE}
            yield {
                "type": "done",
                "answer": _REFUSAL_TEMPLATE,
                "sources": [],
                "suggested_questions": [],
            }
            return

        system_prompt = build_system_prompt(expert_domain)
        doc_blocks = build_documents_blocks(hits)
        conv_block = build_conversation_block(summary, recall_pairs)

        low_conf = _top_score(hits) < _LOW_CONFIDENCE_SCORE
        user_content = build_user_content(query, doc_blocks, conv_block, low_confidence=low_conf)
        messages = list(history) + [{"role": "user", "content": user_content}]

        top_score = hits[0].score if hits else 0.0
        yield {
            "type": "meta",
            "confidence": _confidence(top_score),
            "rewritten_query": search_query if search_query != query else None,
            "recall_count": len(recall_pairs),
        }

        # Stream với Citations API: text deltas yield realtime, citations chỉ
        # hoàn chỉnh sau khi generate_stream_with_citations emit "final" event.
        full_text = ""
        citations: list[dict] = []
        for evt in self.claude.generate_stream_with_citations(
            system_prompt=system_prompt,
            messages=messages,
        ):
            if evt["type"] == "delta":
                yield {"type": "delta", "text": evt["text"]}
            elif evt["type"] == "final":
                full_text = evt["text"]
                citations = evt["citations"]

        # --- Anti-hallucination gates cho streaming path (cùng logic answer()):
        # Layer 1: URL fabrication (deterministic).
        # Layer 2: faithfulness judge CHỈ khi citations==0 + answer dài.
        # Note: text đã stream ra client; FE thay thế bằng evt.answer trong
        # "done" event nếu refuse + show notice cho user (chat.html xử lý).
        forced_refusal = False
        if hits and not _looks_like_refusal(full_text):
            # Layer 1 chạy với mọi answer — URL fabrication không phụ thuộc độ dài.
            fab, fab_url = _has_fabricated_url(full_text, hits)
            if fab:
                logger.warning("URL fabrication (stream): %s", fab_url)
                full_text = _REFUSAL_TEMPLATE
                citations = []
                forced_refusal = True
            # Layer 2 chỉ chạy khi answer có factual claim (số/URL/ngày/chuẩn).
            # Smalltalk thuần (chào, lễ phép, không số) → skip để tránh override
            # response hợp lệ thành refusal template.
            elif not citations and _has_factual_claims(full_text):
                premise = "\n\n---\n\n".join(h.text.strip() for h in hits)
                if len(premise) > 6000:
                    premise = premise[:6000] + "…"
                grounded, reason = check_faithfulness(
                    self.claude, premise, full_text, judge_model=CLAUDE_HAIKU_MODEL,
                )
                if not grounded:
                    logger.warning(
                        "Faithfulness gate TRIPPED (stream, no-citations): %s | len=%d",
                        reason, len(full_text),
                    )
                    full_text = _REFUSAL_TEMPLATE
                    citations = []
                    forced_refusal = True

        _, suggested_questions = _extract_suggestions(full_text)

        # Trust Citations API: bỏ filter score, Claude đã verify chunk hữu ích.
        if citations:
            cited_sources = _build_sources_from_citations(hits, citations)
        else:
            cited_sources = []
        filtered_sources = cited_sources[:_MAX_SOURCES_TO_SHOW]

        yield {
            "type": "done",
            "answer": full_text,
            "sources": filtered_sources,
            "suggested_questions": suggested_questions,
            "citations_count": len(citations),
            "forced_refusal": forced_refusal,
        }
