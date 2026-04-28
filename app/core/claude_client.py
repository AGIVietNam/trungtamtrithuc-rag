from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


def _extract_text_and_citations(message: Any) -> tuple[str, list[dict]]:
    """Đọc final message của Anthropic Citations API → (text, citations list).

    Mỗi text block có thể kèm ``citations`` (list of CitationCharLocation /
    CitationContentBlockLocation). Ta gom flat lại để chain.py post-process.

    Citation fields chuẩn hóa (tên field SDK có thể đổi giữa version):
      - doc_index : int — vị trí trong messages[*].content (đếm cả non-doc)
      - doc_title : str
      - cited_text: str — đoạn nguyên văn Claude trích
      - start, end: vị trí char (text source) hoặc block (custom content)
      - text_in_answer: phần text Claude viết được hỗ trợ bởi citation này
    """
    text_parts: list[str] = []
    citations: list[dict] = []
    content = getattr(message, "content", None) or []
    for block in content:
        btype = getattr(block, "type", None)
        if btype != "text":
            continue
        block_text = getattr(block, "text", "") or ""
        text_parts.append(block_text)
        block_citations = getattr(block, "citations", None) or []
        for c in block_citations:
            citations.append(
                {
                    "text_in_answer": block_text,
                    "doc_index": getattr(c, "document_index", None),
                    "doc_title": getattr(c, "document_title", None),
                    "cited_text": getattr(c, "cited_text", "") or "",
                    "start": (
                        getattr(c, "start_char_index", None)
                        if getattr(c, "start_char_index", None) is not None
                        else getattr(c, "start_block_index", None)
                    ),
                    "end": (
                        getattr(c, "end_char_index", None)
                        if getattr(c, "end_char_index", None) is not None
                        else getattr(c, "end_block_index", None)
                    ),
                }
            )
    return "".join(text_parts), citations


def _attach_history_cache(messages: list[dict]) -> list[dict]:
    """Đặt cache_control ephemeral trên block cuối của assistant gần nhất.

    Lý do chọn assistant gần nhất (không phải user mới):
      - User mới mỗi turn chứa <retrieved_documents> (đổi mỗi request)
        → đặt cache ở đó = anti-pattern (luôn miss, vẫn trả 1.25× write).
      - Assistant gần nhất là khối stable nhất trong prefix tăng dần. Turn
        N+1 sẽ thấy nó trong lookback window (≤20 blocks) của Anthropic
        và cache HIT toàn bộ [system + user1 + asst1 + ... + asstN].

    Immutable: clone tất cả message + content blocks, không mutate caller's
    messages array (caller dùng cùng array để store vào session_memory).

    Turn 1 (history rỗng) → không có assistant nào → trả messages y nguyên,
    chỉ system prompt cache (cache_control gắn ở _build_system).
    """
    if not messages:
        return messages
    out = [dict(m) for m in messages]
    # messages cuối luôn là user mới (chứa docs+query động) — bỏ qua.
    # Scan từ phần tử trước đó lui về đầu, gắn vào assistant đầu tiên gặp.
    for i in range(len(out) - 2, -1, -1):
        if out[i].get("role") != "assistant":
            continue
        content = out[i].get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content = [dict(b) if isinstance(b, dict) else b for b in content]
        else:
            return out  # content lạ → bỏ qua, an toàn
        if not content:
            return out
        last = content[-1]
        if isinstance(last, dict):
            content[-1] = {**last, "cache_control": {"type": "ephemeral"}}
        out[i] = {**out[i], "content": content}
        break
    return out


def _log_usage(usage: Any, mode: str) -> None:
    """Log token usage breakdown để theo dõi cache hit rate sau deploy.

    `cache_creation_input_tokens` > 0 → write cache (1.25× cost).
    `cache_read_input_tokens` > 0    → cache HIT (0.1× cost).
    Cả hai = 0 → không liên quan cache (turn 1 hoặc cache miss hoàn toàn).
    """
    if usage is None:
        return
    try:
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        logger.info(
            "claude usage [%s]: in=%d cache_write=%d cache_read=%d out=%d",
            mode, in_tok, cw, cr, out_tok,
        )
    except Exception:
        logger.debug("usage log failed", exc_info=True)


class ClaudeClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 2048):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def _build_system(self, system_prompt: str) -> list[dict]:
        """Cache chỉ phần stable cross-turn (persona + rules).

        Documents + conversation block đi vào messages (user turn), KHÔNG cache —
        vì chúng đổi mỗi request, nếu cache thì luôn miss và phải trả 25%
        cache-write premium mỗi turn mà không đạt read nào.
        """
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.3,
        model: str | None = None,
    ) -> str:
        cached_messages = _attach_history_cache(messages)
        response = self.client.messages.create(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=self._build_system(system_prompt),
            messages=cached_messages,
            temperature=temperature,
        )
        _log_usage(getattr(response, "usage", None), "sync")
        return response.content[0].text

    def generate_stream(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.3,
        model: str | None = None,
    ) -> Iterator[str]:
        cached_messages = _attach_history_cache(messages)
        with self.client.messages.stream(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=self._build_system(system_prompt),
            messages=cached_messages,
            temperature=temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text
            try:
                final = stream.get_final_message()
                _log_usage(getattr(final, "usage", None), "stream")
            except Exception:
                logger.debug("stream usage log failed", exc_info=True)

    def generate_with_citations(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.3,
        model: str | None = None,
    ) -> dict:
        """Như ``generate()`` nhưng trả {text, citations} cho path Citations API.

        Caller phải đảm bảo trong ``messages`` có ``document`` blocks với
        ``citations.enabled = True`` thì response mới có citations[].
        """
        cached_messages = _attach_history_cache(messages)
        response = self.client.messages.create(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=self._build_system(system_prompt),
            messages=cached_messages,
            temperature=temperature,
        )
        _log_usage(getattr(response, "usage", None), "sync+citations")
        text, citations = _extract_text_and_citations(response)
        return {"text": text, "citations": citations}

    def generate_stream_with_citations(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.3,
        model: str | None = None,
    ) -> Iterator[dict]:
        """Stream version của generate_with_citations.

        Yields events:
            {type:"delta", text:str}            — text chunk realtime
            {type:"final", text:str, citations:list[dict]}
                                                 — sau khi stream done

        Lý do tách "final" event: citations chỉ có complete sau khi message
        finalize. Streaming text trước cho UX, citations về sau cho post-processing.
        """
        cached_messages = _attach_history_cache(messages)
        with self.client.messages.stream(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=self._build_system(system_prompt),
            messages=cached_messages,
            temperature=temperature,
        ) as stream:
            for text in stream.text_stream:
                yield {"type": "delta", "text": text}
            try:
                final = stream.get_final_message()
                _log_usage(getattr(final, "usage", None), "stream+citations")
                full_text, citations = _extract_text_and_citations(final)
                yield {"type": "final", "text": full_text, "citations": citations}
            except Exception:
                logger.warning("stream final/citations parse failed", exc_info=True)
                yield {"type": "final", "text": "", "citations": []}

    def quick_text(
        self,
        system_prompt: str,
        user_content: str | list[dict],
        max_tokens: int = 512,
        model: str | None = None,
    ) -> str:
        """Fast one-shot call without caching — used for reranking/rewriting."""
        messages = [
            {
                "role": "user",
                "content": (
                    user_content
                    if isinstance(user_content, list)
                    else [{"type": "text", "text": user_content}]
                ),
            }
        ]
        response = self.client.messages.create(
            model=model or self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text
