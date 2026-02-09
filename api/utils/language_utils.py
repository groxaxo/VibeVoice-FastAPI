"""Language detection utilities for TTS service."""

from langdetect import detect, LangDetectException

LANGUAGE_CODE_TO_NAME = {
    "es": "spanish",
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "ar": "arabic",
    "hi": "hindi",
    "nl": "dutch",
    "pl": "polish",
    "tr": "turkish",
    "vi": "vietnamese",
    "th": "thai",
    "id": "indonesian",
    "sv": "swedish",
    "no": "norwegian",
    "da": "danish",
    "fi": "finnish",
    "el": "greek",
    "cs": "czech",
    "hu": "hungarian",
    "ro": "romanian",
    "uk": "ukrainian",
    "he": "hebrew",
    "bg": "bulgarian",
    "sk": "slovak",
    "hr": "croatian",
    "ca": "catalan",
    "lt": "lithuanian",
    "sl": "slovenian",
    "et": "estonian",
    "lv": "latvian",
    "mt": "maltese",
    "is": "icelandic",
    "ga": "irish",
    "cy": "welsh",
    "sq": "albanian",
    "mk": "macedonian",
    "bs": "bosnian",
    "sr": "serbian",
    "me": "montenegrin",
    "xh": "xhosa",
    "zu": "zulu",
    "af": "afrikaans",
    "ms": "malay",
    "sw": "swahili",
    "ta": "tamil",
    "te": "telugu",
    "bn": "bengali",
    "ur": "urdu",
    "fa": "persian",
    "ne": "nepali",
    "si": "sinhala",
    "my": "burmese",
    "km": "khmer",
    "lo": "lao",
    "ka": "georgian",
    "hy": "armenian",
    "az": "azerbaijani",
    "kk": "kazakh",
    "ky": "kyrgyz",
    "uz": "uzbek",
    "tg": "tajik",
    "mn": "mongolian",
    "yi": "yiddish",
}


def detect_language(text: str) -> str:
    """
    Detect language from text and return language name.

    Args:
        text: Input text to detect language from

    Returns:
        Language name (e.g., "english", "spanish", "french")

    Notes:
        - Very short text may have lower accuracy
        - For best results with short text, use explicit language parameter
    """
    try:
        # Detect language code from text
        lang_code = detect(text)

        # Map language code to full name
        return LANGUAGE_CODE_TO_NAME.get(lang_code, "english")

    except LangDetectException:
        # Fallback to English if detection fails
        return "english"
