"""Sentence-aware text chunking shared by the API and Gradio Studio."""

from __future__ import annotations

import re

_ELLIPSIS_TOKEN = "\x00VIBEVOICE_ELLIPSIS\x00"
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;])\s+")


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split an oversized sentence without dropping text.

    Prefer whitespace boundaries. A single token longer than ``max_chars`` is split
    deterministically as a last resort.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in text.split():
        if len(word) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
            continue

        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra

    if current:
        chunks.append(" ".join(current))

    return chunks


def split_text_chunks(
    text: str,
    max_chars: int = 2000,
    min_chars: int | None = None,
) -> list[str]:
    """Pack sentences into model-safe chunks without imposing a total text limit.

    Period boundaries are preferred and punctuation is retained. Chunks target the
    inclusive ``min_chars``/``max_chars`` window (1,000-2,000 characters by
    default). A final short chunk is unavoidable when the remaining text is short;
    oversized or unpunctuated sentences fall back to lossless word-aware splitting.
    """
    if max_chars < 32:
        raise ValueError("max_chars must be at least 32")
    if min_chars is None:
        min_chars = min(1000, max_chars)
    if min_chars < 1 or min_chars > max_chars:
        raise ValueError("min_chars must be between 1 and max_chars")

    stripped = text.strip()
    if not stripped:
        return []

    protected = stripped.replace("...", _ELLIPSIS_TOKEN)
    parts = _SENTENCE_BOUNDARY_RE.split(protected)

    units: list[str] = []
    for part in parts:
        restored = part.replace(_ELLIPSIS_TOKEN, "...").strip()
        if not restored:
            continue
        units.extend(_split_oversized(restored, max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = unit

    if current:
        chunks.append(current)

    # If the tail is below the target minimum and fits when joined to the previous
    # chunk, merge it. This keeps common inputs inside the requested range while
    # retaining the hard maximum.
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        merged = f"{chunks[-2]} {chunks[-1]}"
        if len(merged) <= max_chars:
            chunks[-2:] = [merged]

    return chunks
