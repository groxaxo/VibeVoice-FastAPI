"""Input-text sanitization for the VibeVoice TTS server.

Adapted from the Kokoro-FastAPI text-sanitization guide
(github.com/groxaxo/kokoro-fastapi -> docs/TEXT_SANITIZATION_GUIDE.md).

Scope decision
--------------
This server serves **Spanish (es-AR) voices only**. The guide's number / money /
unit / time normalizers emit *English* words ("1035" -> "one thousand and
thirty-five"), so porting them would make an Argentine voice say "one thousand".
The guide itself notes this in its Espanol section. We therefore port only the
**language-agnostic character-sanitization layer** (emoji, control chars, unicode
quotes/dashes, markup symbols, whitespace) plus two TTS-specific guards.

Why this fixes the "padding"
----------------------------
VibeVoice is a diffusion TTS that occasionally fails to emit its stop token and
keeps generating silent / babbling acoustic frames until ``max_new_tokens`` --
the "padding" of trailing silence/garbage (see ``audio_utils.trim_trailing_silence``,
which trims it *after the fact*). Messy, unterminated, or symbol-laden input makes
that stop-token failure much more likely. Two guards here attack it at the source:

  * ``_ensure_terminal`` -- every chunk ends in . ! or ? so the model has a clear
    "this utterance is over" cue (the single biggest lever against trailing padding).
  * ``is_speakable`` -- chunks with no letters/digits (pure emoji/symbols) are
    dropped by the caller instead of being fed to the model (an empty/symbol-only
    prompt is a classic trigger for run-on generation).

Everything is gated by env vars so it can be tuned/disabled without code changes:
  VIBEVOICE_SANITIZE_TEXT          (default on)  -- master switch
  VIBEVOICE_SANITIZE_SYMBOL_WORDS  (default off) -- map % & @ ... to Spanish words
  VIBEVOICE_SANITIZE_ENSURE_TERMINAL (default on) -- append . to unterminated chunks

Dependency-free (stdlib ``re`` + ``unicodedata`` only).
"""

import os
import re
import unicodedata

__all__ = ["sanitize_text", "is_speakable"]


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# Zero-width, BOM, word-joiner, soft-hyphen, bidi marks -> removed outright.
_ZERO_WIDTH = dict.fromkeys(
    map(ord, "​‌‍⁠﻿­‎‏⁡⁢⁣")
)

# C0/C1 control chars (keep \t and \n; they are normalized to space later).
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Emoji & pictographs -- the most common cause of run-on/garbled generation.
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols & pictographs (+ supplemental, emoji, extended-A)
    "\U0001f000-\U0001f0ff"  # mahjong / dominoes / playing cards
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U00002300-\U000023ff"  # misc technical (watches, hourglasses, etc.)
    "\U00002b00-\U00002bff"  # misc symbols & arrows
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "]+",
    flags=re.UNICODE,
)

# Unicode punctuation -> ASCII / speakable. NFC (used below) does NOT fold these,
# so the map is required. Dashes become a comma so the model takes a short pause.
_UNICODE_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",   # ' ' , single quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',   # " " curly doubles
    "«": '"', "»": '"', "‹": '"', "›": '"',   # << >> guillemets
    "–": ", ", "—": ", ", "―": ", ", "‐": "-", "‑": "-",  # dashes
    "−": "-",                                                  # minus sign
    "…": "...",                                               # horizontal ellipsis
    "•": ", ", "·": " ", "‧": " ",                  # bullets / mid dots
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",  # nb/thin spaces
    " ": " ", " ": " ",                                  # line/para separators
}

# CJK / fullwidth punctuation -> Western (from the guide).
_CJK_PUNCT = {
    "、": ", ", "。": ". ", "！": "! ", "，": ", ",
    "：": ": ", "；": "; ", "？": "? ", "｡": ". ",
    "「": '"', "」": '"', "『": '"', "』": '"',
}

# Markup / code symbols that are never spoken aloud -> space.
# Deliberately KEEPS Spanish punctuation: . , ; : ! ? are handled separately and
# the inverted marks (inverted-exclaim / inverted-question), ' " ( ) - / $ stay.
_STRIP_SYMBOLS = dict.fromkeys(map(ord, "*_`#|<>{}[]^~\\=+´ˆ˜"), " ")

# Optional symbol -> Spanish word (off by default; risky around URLs / dates / "y/o").
_SYMBOL_WORDS_ES = {
    "%": " por ciento ",
    "&": " y ",
    "@": " arroba ",
    "+": " mas ",
    "=": " igual ",
    "/": " barra ",
    "°": " grados ",
    "€": " euros ",
    "$": " pesos ",
    "#": " numero ",
}

_MULTISPACE_RE = re.compile(r"[^\S\n]+")          # runs of non-newline whitespace
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_DOTS_RE = re.compile(r"\.{4,}")                   # keep "..." but cap 4+ dots
_REPEAT_PUNCT_RE = re.compile(r"([!?,;:])\1+")     # !!! -> !,  ??? -> ?
_TERMINALS = ".!?"


def _ensure_terminal(text: str) -> str:
    """Make ``text`` end in sentence-final punctuation so the model has a clear
    stop cue (the main lever against trailing-silence padding)."""
    if not text:
        return text
    last = text[-1]
    if last in _TERMINALS:
        return text
    if last in ",;:":            # dangling comma/semicolon/colon -> full stop
        return text[:-1] + "."
    return text + "."


def is_speakable(text: str) -> bool:
    """True if there is anything for the model to actually voice (a letter or
    digit). Pure-symbol/emoji chunks should be dropped, not generated."""
    return any(ch.isalnum() for ch in text)


def sanitize_text(text, *, ensure_terminal: bool = None, symbol_words: bool = None) -> str:
    """Clean one line/chunk of input text before it is handed to VibeVoice.

    Returns a cleaned string (possibly empty -- caller should check
    ``is_speakable``). Conservative and language-agnostic: it removes junk that
    triggers padding/garbled audio but never translates numbers to words.
    """
    if text is None:
        return ""
    if not _flag("VIBEVOICE_SANITIZE_TEXT", True):
        return text.strip()
    if ensure_terminal is None:
        ensure_terminal = _flag("VIBEVOICE_SANITIZE_ENSURE_TERMINAL", True)
    if symbol_words is None:
        symbol_words = _flag("VIBEVOICE_SANITIZE_SYMBOL_WORDS", False)

    # 1. Conservative unicode normalization (NFC keeps a/o ordinals, accents, n~
    #    intact -- NFKC would mangle "Nº" -> "No" and superscripts).
    text = unicodedata.normalize("NFC", text)

    # 2. Drop zero-width / bidi marks and control characters.
    text = text.translate(_ZERO_WIDTH)
    text = _CTRL_RE.sub(" ", text)

    # 3. Strip emoji & pictographs.
    text = _EMOJI_RE.sub(" ", text)

    # 4. Normalize unicode + CJK/fullwidth punctuation to ASCII.
    for src, dst in _UNICODE_PUNCT.items():
        if src in text:
            text = text.replace(src, dst)
    for src, dst in _CJK_PUNCT.items():
        if src in text:
            text = text.replace(src, dst)

    # 5. Either voice a few symbols in Spanish (opt-in) or strip markup symbols.
    if symbol_words:
        for sym, word in _SYMBOL_WORDS_ES.items():
            if sym in text:
                text = text.replace(sym, word)
    text = text.translate(_STRIP_SYMBOLS)

    # 6. Remove any residual symbol/format/private-use/unassigned codepoints
    #    (category So/Sk/Cf/Co/Cs/Cn) that survived the explicit passes.
    if any(unicodedata.category(c) in ("So", "Sk", "Cf", "Co", "Cs", "Cn") for c in text):
        text = "".join(
            " " if unicodedata.category(c) in ("So", "Sk", "Cf", "Co", "Cs", "Cn") else c
            for c in text
        )

    # 7. Collapse runaway punctuation (a known babble trigger).
    text = _DOTS_RE.sub("...", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)

    # 8. Whitespace: newlines -> space, collapse runs, trim.
    text = text.replace("\n", " ").replace("\r", " ")
    text = _MULTISPACE_RE.sub(" ", text).strip()

    # 9. Guarantee a stop cue.
    if ensure_terminal:
        text = _ensure_terminal(text)

    return text


if __name__ == "__main__":
    # Quick self-test: clean Spanish should pass through ~unchanged; junk is removed.
    _cases = [
        "Ah, hola. Mira, estoy aburrida de esperar. Si queres venir, veni.",  # clean (no-op)
        "(llorando) No, no, no, ya no puedo mas...",                          # parenthetical + ...
        "Amor \U0001f60d te extrano un monton \U0001f495\U0001f495",          # emoji
        "**Hola** _mundo_ ~tachado~ `code` #hashtag",                          # markdown noise
        "Veni YA!!!! porfa porfa porfa???",                                    # runaway punctuation
        "Mira “esto” y «aquello» — dale",             # smart quotes / guillemets / em-dash
        "Sin punto final al cerrar",                                           # unterminated -> gets "."
        "\U0001f44d\U0001f44d\U0001f44d",                                      # pure emoji -> empty / not speakable
    ]
    for c in _cases:
        out = sanitize_text(c)
        print(f"IN : {c!r}\nOUT: {out!r}  speakable={is_speakable(out)}\n")
