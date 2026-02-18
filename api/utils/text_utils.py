import re
import unicodedata


def sanitize_text(text: str) -> str:
    """
    Sanitize input text for TTS generation.

    - Normalizes unicode characters (NFKC)
    - Removes non-printable control characters
    - Trims excessive whitespace
    - Preserves emojis and common punctuation

    Args:
        text: Input text string

    Returns:
        Sanitized text string
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove non-printable characters (control chars) except newlines and tabs
    # C category is control characters, Z is separators, M is marks
    # We want to keep some format, but remove weird binary junk
    # simple regex to keep alphanumeric, punctuation, whitespace, and common symbols/emojis

    # Identify control characters (Cc) and remove them, except \n and \t
    text = "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
    )

    # Replace multiple spaces/newlines with single ones (optional, but good for TTS consistency)
    # text = re.sub(r'\s+', ' ', text).strip() # careful, newlines might be semantic for pauses

    return text.strip()
