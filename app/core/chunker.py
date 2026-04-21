from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except Exception:
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    heading_path: list[str] = field(default_factory=list)
    token_count: int = 0


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _merge_with_overlap(units: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _count_tokens(unit)
        if current_tokens + unit_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            overlap_count = 0
            for u in reversed(current):
                t = _count_tokens(u)
                if overlap_count + t > overlap_tokens:
                    break
                overlap.insert(0, u)
                overlap_count += t
            current = overlap
            current_tokens = overlap_count
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def _parse_markdown_sections(text: str) -> list[tuple[list[str], str]]:
    lines = text.splitlines(keepends=True)
    sections: list[tuple[list[str], str]] = []
    current_path: list[str] = []
    current_lines: list[str] = []
    heading_re = re.compile(r"^(#{1,6})\s+(.*)")

    def flush():
        if current_lines:
            sections.append((list(current_path), "".join(current_lines)))
            current_lines.clear()

    for line in lines:
        m = heading_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            current_path = current_path[: level - 1] + [title]
        else:
            current_lines.append(line)

    flush()
    return sections


def chunk_text(text: str, max_tokens: int = 700, overlap_tokens: int = 80) -> list[Chunk]:
    sections = _parse_markdown_sections(text)
    if not sections:
        sections = [([], text)]

    chunks: list[Chunk] = []
    char_offset = 0

    for heading_path, section_text in sections:
        paragraphs = re.split(r"\n{2,}", section_text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        if not paragraphs:
            continue

        if _count_tokens(section_text) <= max_tokens:
            units = [section_text]
        else:
            units: list[str] = []
            for para in paragraphs:
                if _count_tokens(para) <= max_tokens:
                    units.append(para)
                else:
                    units.extend(_split_sentences(para))

        for chunk_str in _merge_with_overlap(units, max_tokens, overlap_tokens):
            start = text.find(chunk_str, char_offset)
            if start == -1:
                start = char_offset
            end = start + len(chunk_str)
            chunks.append(Chunk(
                text=chunk_str,
                start_char=start,
                end_char=end,
                heading_path=list(heading_path),
                token_count=_count_tokens(chunk_str),
            ))
            char_offset = max(char_offset, start)

    return chunks


def chunk_transcript_with_timestamps(segments: list[dict], max_tokens: int = 500) -> list[dict]:
    result: list[dict] = []
    current_texts: list[str] = []
    current_ids: list[int] = []
    current_tokens = 0
    chunk_start = 0.0
    chunk_end = 0.0

    for idx, seg in enumerate(segments):
        seg_text = seg.get("text", "").strip()
        seg_tokens = _count_tokens(seg_text)

        if current_tokens + seg_tokens > max_tokens and current_texts:
            result.append({"text": " ".join(current_texts), "start": chunk_start, "end": chunk_end, "segment_ids": list(current_ids)})
            current_texts = []
            current_ids = []
            current_tokens = 0

        if not current_texts:
            chunk_start = float(seg.get("start", 0))

        current_texts.append(seg_text)
        current_ids.append(idx)
        current_tokens += seg_tokens
        chunk_end = float(seg.get("end", 0))

    if current_texts:
        result.append({"text": " ".join(current_texts), "start": chunk_start, "end": chunk_end, "segment_ids": list(current_ids)})

    return result
