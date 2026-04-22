from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


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
