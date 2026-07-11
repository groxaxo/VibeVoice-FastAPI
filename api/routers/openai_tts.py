"""OpenAI-compatible TTS endpoints."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from api.config import settings
from api.models import OpenAITTSRequest
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager
from api.utils.audio_utils import (
    adjust_audio_speed,
    audio_to_bytes,
    concatenate_audio_chunks,
    get_audio_duration,
    get_content_type,
)
from api.utils.text_chunking import split_text_chunks
from api.utils.text_sanitizer import is_speakable, sanitize_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/audio", tags=["OpenAI Compatible"])

tts_service: TTSService | None = None
voice_manager: VoiceManager | None = None

_STREAM_SAFE_FORMATS = {"mp3", "opus", "aac", "pcm"}


def get_tts_service() -> TTSService:
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    if not tts_service.is_loaded and not settings.vibevoice_lazy_load:
        raise HTTPException(status_code=503, detail="TTS model is unloaded; call /v1/vibevoice/preload")
    return tts_service


def get_voice_manager() -> VoiceManager:
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="Voice manager not initialized")
    return voice_manager


def _resolve_voice(voices: VoiceManager, voice_name: str):
    audio = voices.load_voice_audio(voice_name, is_openai_voice=True)
    if audio is None:
        audio = voices.load_voice_audio(voice_name, is_openai_voice=False)
    if audio is not None:
        return audio

    available_openai = ", ".join(voices.OPENAI_VOICE_MAPPING)
    available_presets = ", ".join(sorted(voices.voice_presets))
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice '{voice_name}' not found. OpenAI voices: {available_openai}. "
            f"VibeVoice presets: {available_presets}"
        ),
    )


def _generate_chunk(
    tts: TTSService,
    sentence: str,
    voice_audio,
    cancel_event: threading.Event,
):
    formatted = tts.format_script_for_single_speaker(sentence, speaker_id=0)
    if not formatted:
        return None
    return tts.generate_speech(
        text=formatted,
        voice_samples=[voice_audio],
        cfg_scale=settings.default_cfg_scale,
        stream=False,
        cancel_event=cancel_event,
    )


def _encode_response_audio(audio, speed: float, response_format: str):
    adjusted = adjust_audio_speed(audio, speed)
    encoded = audio_to_bytes(adjusted, sample_rate=24000, format=response_format)
    return adjusted, encoded


async def _sentence_stream_generator(
    sentences: list[str],
    voice_audio,
    body: OpenAITTSRequest,
    tts: TTSService,
    cancel_event: threading.Event,
    request: Request,
) -> AsyncIterator[bytes]:
    """Generate bounded utterances without blocking the event loop."""
    try:
        with tts.request_context():
            for index, sentence in enumerate(sentences):
                if cancel_event.is_set() or await request.is_disconnected():
                    cancel_event.set()
                    return

                task = asyncio.create_task(
                    asyncio.to_thread(_generate_chunk, tts, sentence, voice_audio, cancel_event)
                )
                while not task.done():
                    await asyncio.sleep(0.05)
                    if await request.is_disconnected():
                        cancel_event.set()
                        try:
                            await asyncio.wait_for(task, timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                        return

                audio = await task
                if audio is None:
                    continue
                _, chunk_bytes = await asyncio.to_thread(
                    _encode_response_audio,
                    audio,
                    body.speed,
                    body.response_format,
                )
                logger.debug("Streaming chunk %d/%d", index + 1, len(sentences))
                yield chunk_bytes
    finally:
        cancel_event.set()


@router.post("/speech")
async def create_speech(
    body: OpenAITTSRequest,
    request: Request,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager),
):
    """Generate speech through an OpenAI-compatible request shape."""
    try:
        sanitized = sanitize_text(body.input)
        if not sanitized or not is_speakable(sanitized):
            raise HTTPException(status_code=400, detail="Input contains no speakable text")

        if body.stream and body.response_format not in _STREAM_SAFE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Streaming supports mp3, opus, aac, or pcm. Container formats "
                    "wav, flac, and m4a require a finalized file header."
                ),
            )

        voice_audio = await asyncio.to_thread(_resolve_voice, voices, body.voice)
        sentences = split_text_chunks(sanitized, settings.vibevoice_max_chunk_chars)
        if not sentences:
            raise HTTPException(status_code=400, detail="Input contains no speakable text")

        preview = sanitized[:100] + ("..." if len(sanitized) > 100 else "")
        logger.info(
            "TTS request: %d chunk(s), voice=%s, format=%s, stream=%s, text=%r",
            len(sentences),
            body.voice,
            body.response_format,
            body.stream,
            preview,
        )
        cancel_event = threading.Event()

        if body.stream:
            return StreamingResponse(
                _sentence_stream_generator(
                    sentences, voice_audio, body, tts, cancel_event, request
                ),
                media_type=get_content_type(body.response_format),
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        start = time.monotonic()

        def run_all():
            with tts.request_context():
                parts = []
                for sentence in sentences:
                    if cancel_event.is_set():
                        break
                    audio = _generate_chunk(tts, sentence, voice_audio, cancel_event)
                    if audio is not None:
                        parts.append(audio)
                return concatenate_audio_chunks(parts) if parts else None

        task = asyncio.create_task(asyncio.to_thread(run_all))
        while not task.done():
            await asyncio.sleep(0.1)
            if await request.is_disconnected():
                logger.info("Client disconnected; cancelling TTS generation")
                cancel_event.set()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                return Response(status_code=499)

        audio = await task
        if audio is None:
            return Response(status_code=499)

        audio, audio_bytes = await asyncio.to_thread(
            _encode_response_audio,
            audio,
            body.speed,
            body.response_format,
        )
        generation_time = time.monotonic() - start
        duration = get_audio_duration(audio, sample_rate=24000)
        logger.info(
            "Generated %.2fs audio in %.2fs (%d chunk(s), voice=%s)",
            duration,
            generation_time,
            len(sentences),
            body.voice,
        )

        return Response(
            content=audio_bytes,
            media_type=get_content_type(body.response_format),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{body.response_format}",
                "X-Audio-Duration": f"{duration:.3f}",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error generating OpenAI-compatible speech")
        raise HTTPException(status_code=500, detail="Speech generation failed") from exc


@router.get("/voices")
async def list_voices(voices: VoiceManager = Depends(get_voice_manager)):
    """List OpenAI aliases and custom presets in OpenAI list format."""
    try:
        data = [
            {"id": alias, "object": "voice", "name": alias}
            for alias, preset in voices.OPENAI_VOICE_MAPPING.items()
            if preset in voices.voice_presets
        ]
        mapped_presets = set(voices.OPENAI_VOICE_MAPPING.values())
        data.extend(
            {"id": voice["name"], "object": "voice", "name": voice["name"]}
            for voice in voices.list_available_voices()
            if voice["name"] not in mapped_presets
        )
        return {"object": "list", "data": data}
    except Exception as exc:
        logger.exception("Failed to list voices")
        raise HTTPException(status_code=500, detail="Failed to list voices") from exc
