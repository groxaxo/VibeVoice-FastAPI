"""VibeVoice-specific TTS endpoints with multi-speaker support."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import re
import threading
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from api.config import settings
from api.models import HealthResponse, VibeVoiceGenerateRequest, VoiceListResponse
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager
from api.utils.audio_utils import (
    audio_to_bytes,
    concatenate_audio_chunks,
    get_audio_duration,
    get_content_type,
)
from api.utils.streaming import create_streaming_response
from api.utils.text_chunking import split_text_chunks
from api.utils.text_sanitizer import is_speakable, sanitize_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/vibevoice", tags=["VibeVoice Extended"])

tts_service: TTSService | None = None
voice_manager: VoiceManager | None = None

_SPEAKER_RE = re.compile(r"^\s*Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)


def get_tts_service() -> TTSService:
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    if not tts_service.is_loaded and not settings.vibevoice_lazy_load:
        raise HTTPException(status_code=503, detail="TTS model is unloaded; call /v1/vibevoice/preload")
    return tts_service


def get_tts_service_status() -> TTSService:
    """Return the service without triggering or requiring a model load."""
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    return tts_service


def get_voice_manager() -> VoiceManager:
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="Voice manager not initialized")
    return voice_manager


def _parse_script_to_chunks(
    script: str,
    num_speakers: int,
    max_chars: int,
    min_chars: int = 1000,
) -> list[tuple[int, str]]:
    """Parse labels, sanitize utterances, and enforce a hard prompt-size bound."""
    chunks: list[tuple[int, str]] = []
    for raw_line in script.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        match = _SPEAKER_RE.match(line)
        if match:
            speaker_index = int(match.group(1))
            utterance = match.group(2).strip()
        else:
            speaker_index = 0
            utterance = line

        if speaker_index >= num_speakers:
            raise ValueError(
                f"Script references Speaker {speaker_index}, but only {num_speakers} speaker(s) were configured"
            )

        for part in split_text_chunks(
            utterance,
            max_chars=max_chars,
            min_chars=min(min_chars, max_chars),
        ):
            cleaned = sanitize_text(part)
            if cleaned and is_speakable(cleaned):
                chunks.append((speaker_index, cleaned))
    return chunks


def _decode_base64_voice(encoded: str, speaker_id: int) -> np.ndarray:
    max_bytes = settings.max_voice_sample_bytes
    # Base64 expands binary data by roughly 4/3. Reject oversized payloads before
    # allocating the decoded buffer.
    max_encoded = ((max_bytes + 2) // 3) * 4 + 16
    if len(encoded) > max_encoded:
        raise HTTPException(
            status_code=413,
            detail=f"Voice sample for speaker {speaker_id} exceeds {max_bytes} bytes",
        )

    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 voice sample for speaker {speaker_id}",
        ) from exc

    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Voice sample for speaker {speaker_id} exceeds {max_bytes} bytes",
        )

    try:
        import soundfile as sf

        try:
            audio, sample_rate = sf.read(
                io.BytesIO(payload), dtype="float32", always_2d=False
            )
        except Exception:
            from pydub import AudioSegment

            segment = AudioSegment.from_file(io.BytesIO(payload))
            if segment.channels > 1:
                segment = segment.set_channels(1)
            samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
            divisor = float(1 << (8 * segment.sample_width - 1))
            audio = samples / divisor
            sample_rate = segment.frame_rate

        if sample_rate <= 0:
            raise ValueError("audio has an invalid sample rate")

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.reshape(-1)
        if audio.size == 0:
            raise ValueError("audio is empty")
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        duration = audio.size / float(sample_rate)
        if duration > settings.max_voice_sample_seconds:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Voice sample for speaker {speaker_id} is {duration:.1f}s; "
                    f"maximum is {settings.max_voice_sample_seconds:.1f}s"
                ),
            )
        if sample_rate != 24000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=24000)
        return np.ascontiguousarray(audio, dtype=np.float32)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode voice sample for speaker {speaker_id}",
        ) from exc


def _load_voice_samples(speakers, voices: VoiceManager):
    """Decode all requested voices off the event loop."""
    voice_samples: list[np.ndarray] = []
    voice_labels: list[str] = []
    for speaker in sorted(speakers, key=lambda item: item.speaker_id):
        if speaker.voice_sample_base64:
            audio = _decode_base64_voice(
                speaker.voice_sample_base64,
                speaker.speaker_id,
            )
            voice_labels.append(f"speaker{speaker.speaker_id}=base64_audio")
        else:
            audio = voices.load_voice_audio(
                speaker.voice_preset,
                is_openai_voice=False,
            )
            if audio is None:
                available = ", ".join(
                    voice["name"] for voice in voices.list_available_voices()
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Voice preset '{speaker.voice_preset}' not found. "
                        f"Available: {available}"
                    ),
                )
            voice_labels.append(
                f"speaker{speaker.speaker_id}={speaker.voice_preset}"
            )
        voice_samples.append(audio)
    return voice_samples, voice_labels


def _encode_response_audio(audio, response_format: str) -> bytes:
    return audio_to_bytes(audio, sample_rate=24000, format=response_format)


@router.post("/generate")
async def generate_speech(
    body: VibeVoiceGenerateRequest,
    request: Request,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager),
):
    """Generate bounded multi-speaker speech, optionally as SSE chunks."""
    try:
        voice_samples, voice_labels = await asyncio.to_thread(
            _load_voice_samples,
            body.speakers,
            voices,
        )

        try:
            chunks = await asyncio.to_thread(
                _parse_script_to_chunks,
                body.script,
                len(voice_samples),
                settings.vibevoice_max_chunk_chars,
                settings.vibevoice_min_chunk_chars,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not chunks:
            raise HTTPException(status_code=400, detail="No speakable text found in script")

        inference_steps = (
            body.inference_steps
            if body.inference_steps is not None
            else settings.vibevoice_inference_steps
        )
        preview_text = " ".join(text for _, text in chunks)
        preview = preview_text[:100] + ("..." if len(preview_text) > 100 else "")
        logger.info(
            "VibeVoice request: %d chunk(s), voices=%s, steps=%d, stream=%s, text=%r",
            len(chunks),
            ", ".join(voice_labels),
            inference_steps,
            body.stream,
            preview,
        )

        cancel_event = threading.Event()

        def generate_one(chunk_index: int, speaker_index: int, sentence: str):
            formatted = tts.format_script_for_single_speaker(sentence, speaker_id=0)
            if not formatted:
                return None
            chunk_seed = body.seed + chunk_index if body.seed is not None else None
            return tts.generate_speech(
                text=formatted,
                voice_samples=[voice_samples[speaker_index]],
                cfg_scale=body.cfg_scale,
                inference_steps=inference_steps,
                do_sample=body.do_sample,
                temperature=body.temperature,
                top_p=body.top_p,
                seed=chunk_seed,
                stream=False,
                cancel_event=cancel_event,
            )

        if body.stream:
            def audio_iterator():
                with tts.request_context():
                    for index, (speaker_index, sentence) in enumerate(chunks):
                        if cancel_event.is_set():
                            break
                        audio = generate_one(index, speaker_index, sentence)
                        if audio is not None and not cancel_event.is_set():
                            yield audio

            return create_streaming_response(
                audio_iterator(),
                format=body.response_format,
                sample_rate=24000,
                use_sse=True,
                cancel_event=cancel_event,
                request=request,
            )

        start = time.monotonic()

        def run_all():
            with tts.request_context():
                parts = []
                for index, (speaker_index, sentence) in enumerate(chunks):
                    if cancel_event.is_set():
                        break
                    audio = generate_one(index, speaker_index, sentence)
                    if audio is not None:
                        parts.append(audio)
                return concatenate_audio_chunks(parts) if parts else None

        task = asyncio.create_task(asyncio.to_thread(run_all))
        while not task.done():
            await asyncio.sleep(0.1)
            if await request.is_disconnected():
                logger.info("Native client disconnected; cancelling generation")
                cancel_event.set()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                return Response(status_code=499)

        audio = await task
        if audio is None:
            return Response(status_code=499)

        generation_time = time.monotonic() - start
        duration = get_audio_duration(audio, sample_rate=24000)
        logger.info(
            "Generated %.2fs audio in %.2fs (%d chunk(s))",
            duration,
            generation_time,
            len(chunks),
        )
        audio_bytes = await asyncio.to_thread(
            _encode_response_audio,
            audio,
            body.response_format,
        )
        return Response(
            content=audio_bytes,
            media_type=get_content_type(body.response_format),
            headers={
                "Content-Disposition": f"attachment; filename=vibevoice_output.{body.response_format}",
                "X-Audio-Duration": f"{duration:.3f}",
                "X-Audio-Format": body.response_format,
                "X-Audio-Sample-Rate": "24000",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error generating VibeVoice speech")
        raise HTTPException(status_code=500, detail="Speech generation failed") from exc


@router.get("/voices", response_model=VoiceListResponse)
async def list_vibevoice_voices(voices: VoiceManager = Depends(get_voice_manager)):
    try:
        available = voices.list_available_voices()
        return VoiceListResponse(voices=available, count=len(available))
    except Exception as exc:
        logger.exception("Failed to list VibeVoice voices")
        raise HTTPException(status_code=500, detail="Failed to list voices") from exc


@router.get("/health", response_model=HealthResponse)
async def vibevoice_health(tts: TTSService = Depends(get_tts_service_status)):
    if tts.is_loaded:
        status = "healthy"
    elif settings.vibevoice_lazy_load:
        status = "idle"
    else:
        status = "unavailable"
    return HealthResponse(
        status=status,
        model_loaded=tts.is_loaded,
        model_path=settings.vibevoice_model_path,
        device=str(tts.device or settings.vibevoice_device),
        busy=tts.is_busy,
    )
