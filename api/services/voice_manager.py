"""Voice preset management and OpenAI voice mapping."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class VoiceManager:
    """Manage voice presets and cache decoded/resampled reference audio."""

    def __init__(self, voices_dir: str = "demo/voices", openai_voice_mapping: Optional[str] = None):
        self.voices_dir = Path(voices_dir)
        self.voice_presets: Dict[str, str] = {}
        self._audio_cache: dict[tuple[str, int, int], np.ndarray] = {}
        self._cache_lock = threading.RLock()

        self.OPENAI_VOICE_MAPPING = self._parse_mapping(openai_voice_mapping)
        self.load_voice_presets()

    @classmethod
    def _parse_mapping(cls, mapping_json: Optional[str]) -> Dict[str, str]:
        if not mapping_json:
            return cls._get_default_mapping()
        try:
            parsed = json.loads(mapping_json)
            if not isinstance(parsed, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
            ):
                raise ValueError("mapping must be a JSON object of string keys and values")
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Invalid OPENAI_VOICE_MAPPING (%s); using defaults", exc)
            return cls._get_default_mapping()

    @staticmethod
    def _get_default_mapping() -> Dict[str, str]:
        return {
            "alloy": "en-Alice_woman",
            "echo": "en-Carter_man",
            "fable": "en-Maya_woman",
            "onyx": "en-Frank_man",
            "nova": "en-Mary_woman_bgm",
            "shimmer": "en-Alice_woman",
        }

    def load_voice_presets(self) -> None:
        """Rescan the configured directory and invalidate removed/stale entries."""
        if not self.voices_dir.exists():
            logger.warning("Voices directory not found at %s", self.voices_dir)
            self.voice_presets.clear()
            with self._cache_lock:
                self._audio_cache.clear()
            return

        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}
        discovered: Dict[str, str] = {}
        for file_path in sorted(self.voices_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                discovered[file_path.stem] = str(file_path)

        self.voice_presets = discovered
        valid_paths = set(discovered.values())
        with self._cache_lock:
            self._audio_cache = {
                key: value for key, value in self._audio_cache.items() if key[0] in valid_paths
            }

        logger.info("Loaded %d voice presets from %s", len(discovered), self.voices_dir)
        if discovered:
            logger.debug("Available voices: %s", ", ".join(sorted(discovered)))

    def get_voice_path(self, voice_name: str, is_openai_voice: bool = False) -> Optional[str]:
        if is_openai_voice:
            voice_name = self.OPENAI_VOICE_MAPPING.get(voice_name, voice_name)
        return self.voice_presets.get(voice_name)

    @staticmethod
    def _decode_voice(path: str) -> tuple[np.ndarray, int]:
        file_ext = Path(path).suffix.lower()
        if file_ext in {".m4a", ".aac", ".mp3", ".opus"}:
            segment = AudioSegment.from_file(path)
            if segment.channels > 1:
                segment = segment.set_channels(1)
            samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
            max_int = float(1 << (8 * segment.sample_width - 1))
            return samples / max_int, segment.frame_rate

        wav, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return np.asarray(wav, dtype=np.float32), int(sample_rate)

    def load_voice_audio(
        self,
        voice_name: str,
        is_openai_voice: bool = False,
        target_sr: int = 24000,
    ) -> Optional[np.ndarray]:
        """Load and cache a voice preset at ``target_sr``."""
        voice_path = self.get_voice_path(voice_name, is_openai_voice)
        if not voice_path:
            return None

        try:
            stat = Path(voice_path).stat()
            cache_key = (voice_path, stat.st_mtime_ns, target_sr)
            # Keep cold decoding under the cache lock so simultaneous first
            # requests do not decode and resample the same file repeatedly.
            with self._cache_lock:
                cached = self._audio_cache.get(cache_key)
                if cached is not None:
                    return cached.copy()

                wav, sample_rate = self._decode_voice(voice_path)
                if wav.size == 0:
                    raise ValueError("voice sample is empty")
                if sample_rate <= 0:
                    raise ValueError("voice sample has an invalid sample rate")
                if not np.isfinite(wav).all():
                    wav = np.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
                if sample_rate != target_sr:
                    import librosa

                    wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=target_sr)

                wav = np.ascontiguousarray(wav.reshape(-1), dtype=np.float32)
                stale = [
                    key
                    for key in self._audio_cache
                    if key[0] == voice_path and key[2] == target_sr
                ]
                for key in stale:
                    self._audio_cache.pop(key, None)
                self._audio_cache[cache_key] = wav
                return wav.copy()
        except Exception as exc:
            logger.exception("Error loading voice %s from %s: %s", voice_name, voice_path, exc)
            return None

    def list_available_voices(self) -> List[Dict[str, str]]:
        return [
            {"name": name, "path": path, "language": self._guess_language(name)}
            for name, path in sorted(self.voice_presets.items())
        ]

    def list_openai_voices(self) -> List[Dict[str, str | bool]]:
        return [
            {
                "name": openai_name,
                "vibevoice_preset": preset,
                "available": preset in self.voice_presets,
            }
            for openai_name, preset in self.OPENAI_VOICE_MAPPING.items()
        ]

    @staticmethod
    def _guess_language(voice_name: str) -> str:
        prefix = voice_name.lower().split("-", 1)[0]
        return {
            "en": "English",
            "es": "Spanish",
            "zh": "Chinese",
            "in": "Indian English",
        }.get(prefix, "Unknown")

    def get_default_voice(self) -> Optional[str]:
        return next((name for name in self.voice_presets if name.startswith("en-")), None) or next(
            iter(self.voice_presets), None
        )
