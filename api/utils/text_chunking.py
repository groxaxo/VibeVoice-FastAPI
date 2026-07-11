"""Text chunking helpers shared by the API routers.

The model has bounded output generation, but very long unpunctuated inputs can still
create oversized prompts. These helpers split on sentence boundaries first and then
apply a word-aware hard limit so every model call remains bounded.
"""

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


def split_text_chunks(text: str, max_chars: int = 1000) -> list[str]:
    """Split text into bounded, non-empty chunks.

    Sentence punctuation is preserved. Ellipses are protected so ``...`` does not
    produce empty fragments. Inputs without punctuation are still bounded by
    ``max_chars``.
    """
    if max_chars < 32:
        raise ValueError("max_chars must be at least 32")

    stripped = text.strip()
    if not stripped:
        return []

    protected = stripped.replace("...", _ELLIPSIS_TOKEN)
    parts = _SENTENCE_BOUNDARY_RE.split(protected)

    chunks: list[str] = []
    for part in parts:
        restored = part.replace(_ELLIPSIS_TOKEN, "...").strip()
        if not restored:
            continue
        chunks.extend(_split_oversized(restored, max_chars))

    return chunks
