from __future__ import annotations

from collections.abc import Iterator

import anthropic


class ClaudeClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 2048):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def _build_system(self, system_prompt: str, context_block: str) -> list[dict]:
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if context_block:
            system.append(
                {
                    "type": "text",
                    "text": context_block,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        return system

    def generate(
        self,
        system_prompt: str,
        context_block: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.3,
        model: str | None = None,
    ) -> str:
        response = self.client.messages.create(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=self._build_system(system_prompt, context_block),
            messages=messages,
            temperature=temperature,
        )
        return response.content[0].text

    def generate_stream(
        self,
        system_prompt: str,
        context_block: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.3,
        model: str | None = None,
    ) -> Iterator[str]:
        with self.client.messages.stream(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=self._build_system(system_prompt, context_block),
            messages=messages,
            temperature=temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text

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
