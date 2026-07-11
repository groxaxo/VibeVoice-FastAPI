"""Audio format conversion and processing utilities."""

from __future__ import annotations

import io
from typing import Literal, Union

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "m4a"]
AudioArray = Union[np.ndarray, torch.Tensor]


def _to_mono_float32(audio: AudioArray) -> np.ndarray:
    """Convert tensors/arrays to a finite, contiguous mono float32 vector."""
    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu().numpy()

    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)

    array = np.squeeze(array)
    if array.ndim > 1:
        # Model outputs are commonly [1, samples], while decoded audio is usually
        # [samples, channels]. Select the likely channel axis conservatively.
        channel_axis = 0 if array.shape[0] <= 8 and array.shape[0] < array.shape[-1] else -1
        array = array.mean(axis=channel_axis)

    array = np.ascontiguousarray(array.reshape(-1), dtype=np.float32)
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    return array


def convert_to_16_bit_wav(audio: AudioArray) -> np.ndarray:
    """Convert audio to clipped 16-bit PCM without failing on empty arrays."""
    array = _to_mono_float32(audio)
    if array.size == 0:
        return np.empty(0, dtype=np.int16)

    peak = float(np.max(np.abs(array)))
    if peak > 1.0:
        array = array / peak
    array = np.clip(array, -1.0, 1.0)
    return np.rint(array * 32767.0).astype(np.int16)


def adjust_audio_speed(audio: AudioArray, speed: float) -> np.ndarray:
    """Change playback tempo while preserving pitch.

    ``librosa.effects.time_stretch`` is only imported when a non-default speed is
    requested, so the common path has no extra startup cost.
    """
    array = _to_mono_float32(audio)
    if array.size == 0 or abs(speed - 1.0) < 1e-6:
        return array
    if not 0.25 <= speed <= 4.0:
        raise ValueError("speed must be between 0.25 and 4.0")

    import librosa

    stretched = librosa.effects.time_stretch(array, rate=float(speed))
    return np.ascontiguousarray(stretched, dtype=np.float32)


def audio_to_bytes(
    audio: AudioArray,
    sample_rate: int = 24000,
    format: AudioFormat = "mp3",
    bitrate: str = "128k",
) -> bytes:
    """Encode an audio array to the requested response format."""
    audio_16bit = convert_to_16_bit_wav(audio)

    if format == "pcm":
        return audio_16bit.tobytes()

    if format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, audio_16bit, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_16bit, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    audio_segment = AudioSegment.from_wav(wav_buffer)

    output_buffer = io.BytesIO()
    if format == "m4a":
        export_format = "mp4"
    elif format == "aac":
        export_format = "adts"
    else:
        export_format = format
    export_params: dict[str, str] = {"format": export_format}

    if format in {"mp3", "opus", "aac", "m4a"}:
        export_params["bitrate"] = bitrate
    if format == "opus":
        export_params["codec"] = "libopus"
    elif format in {"aac", "m4a"}:
        export_params["codec"] = "aac"

    audio_segment.export(output_buffer, **export_params)
    return output_buffer.getvalue()


def get_audio_duration(audio: AudioArray, sample_rate: int = 24000) -> float:
    """Return audio duration in seconds."""
    return _to_mono_float32(audio).size / sample_rate


def get_content_type(format: AudioFormat) -> str:
    """Return the MIME content type for an audio format."""
    return {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "application/octet-stream",
        "m4a": "audio/mp4",
    }.get(format, "application/octet-stream")


def concatenate_audio_chunks(chunks: list[AudioArray]) -> np.ndarray:
    """Concatenate audio chunks as a contiguous mono float32 array."""
    arrays = [_to_mono_float32(chunk) for chunk in chunks]
    arrays = [array for array in arrays if array.size]
    if not arrays:
        return np.empty(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)


def trim_trailing_silence(
    audio: AudioArray,
    sample_rate: int = 24000,
    thresh_ratio: float = 0.02,
    frame_s: float = 0.02,
    pad_s: float = 0.2,
    min_peak: float = 0.01,
) -> np.ndarray:
    """Trim trailing near-silence while retaining a short natural tail."""
    array = _to_mono_float32(audio)
    if array.size == 0 or float(np.abs(array).max()) < min_peak:
        return array

    window = max(1, int(frame_s * sample_rate))
    frame_count = array.size // window
    if frame_count == 0:
        return array

    frames = array[: frame_count * window].reshape(frame_count, window)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    threshold = thresh_ratio * float(rms.max())
    voiced = np.flatnonzero(rms > threshold)
    if voiced.size == 0:
        return array

    end = min(array.size, int((voiced[-1] + 1) * window + pad_s * sample_rate))
    return array[:end]
