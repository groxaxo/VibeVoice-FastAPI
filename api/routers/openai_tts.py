"""OpenAI-compatible TTS endpoint with sentence-chunked streaming."""

import asyncio
import logging
import re
import threading
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import Response, StreamingResponse

from api.models import OpenAITTSRequest, ErrorResponse
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager
from api.utils.audio_utils import audio_to_bytes, get_content_type, get_audio_duration, concatenate_audio_chunks
from api.utils.streaming import create_streaming_response
from api.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["OpenAI Compatible"])

# Global service instances (initialized in main.py)
tts_service: TTSService = None
voice_manager: VoiceManager = None


def _split_sentences(text: str) -> list:
    """Split text at sentence boundaries (. ! ? ;) while preserving ellipsis (...)."""
    # Protect ellipsis from being treated as a sentence end
    text = re.sub(r'\.\.\.', '\x00ELP\x00', text)
    # Split after . ! ? ; followed by whitespace
    parts = re.split(r'(?<=[.!?;])\s+', text.strip())
    result = []
    for part in parts:
        part = part.replace('\x00ELP\x00', '...').strip()
        if part:
            result.append(part)
    return result if result else [text.strip()]


def get_tts_service() -> TTSService:
    """Dependency to get TTS service. Triggers lazy load on first request."""
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    if not tts_service.is_loaded:
        if not settings.vibevoice_lazy_load:
            raise HTTPException(status_code=503, detail="TTS service not ready")
        logger.info("Lazy-load: triggering model load on first request...")
        tts_service.load_model()
    return tts_service


def get_voice_manager() -> VoiceManager:
    """Dependency to get voice manager."""
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="Voice manager not initialized")
    return voice_manager


async def _sentence_stream_generator(sentences, voice_audio, body, tts, cancel_event, request):
    """Async generator: generate and yield audio bytes one sentence at a time."""
    for i, sentence in enumerate(sentences):
        if cancel_event.is_set():
            break
        if await request.is_disconnected():
            cancel_event.set()
            break

        formatted = tts.format_script_for_single_speaker(sentence, speaker_id=0)
        result_holder = {}

        def _gen(fmt=formatted):
            try:
                result_holder['audio'] = tts.generate_speech(
                    text=fmt,
                    voice_samples=[voice_audio],
                    cfg_scale=settings.default_cfg_scale,
                    stream=False,
                    cancel_event=cancel_event,
                )
            except Exception as e:
                result_holder['error'] = e

        thread = threading.Thread(target=_gen, daemon=True)
        thread.start()

        while thread.is_alive():
            await asyncio.sleep(0.05)
            if cancel_event.is_set() or await request.is_disconnected():
                cancel_event.set()
                thread.join(timeout=5.0)
                return

        if 'error' in result_holder:
            raise result_holder['error']

        audio = result_holder.get('audio')
        if audio is not None:
            logger.info(f"Streaming sentence {i+1}/{len(sentences)}: {sentence[:60]!r}")
            yield audio_to_bytes(audio, sample_rate=24000, format=body.response_format)


@router.post("/speech")
async def create_speech(
    body: OpenAITTSRequest,
    request: Request,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    Generate speech from text using OpenAI-compatible API.

    Text is auto-split at sentence boundaries (. ! ? ;) before generation.
    stream=True  -> each sentence audio yielded immediately (low TTFB).
    stream=False -> all sentences generated, merged, returned as one file.
    """
    try:
        voice_audio = voices.load_voice_audio(body.voice, is_openai_voice=True)
        if voice_audio is None:
            voice_audio = voices.load_voice_audio(body.voice, is_openai_voice=False)
        if voice_audio is None:
            available_openai = ', '.join(voices.OPENAI_VOICE_MAPPING.keys())
            available_presets = ', '.join(sorted(voices.voice_presets.keys()))
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Voice '{body.voice}' not found. OpenAI voices: {available_openai}. "
                    f"VibeVoice presets: {available_presets}"
                )
            )

        sentences = _split_sentences(body.input)
        logger.info(f"Split into {len(sentences)} sentence(s): {sentences}")
        cancel_event = threading.Event()

        if body.stream:
            # Streaming: yield sentence audio as each is generated (low TTFB)
            return StreamingResponse(
                _sentence_stream_generator(sentences, voice_audio, body, tts, cancel_event, request),
                media_type=get_content_type(body.response_format),
                headers={"Transfer-Encoding": "chunked", "Cache-Control": "no-cache"},
            )

        # Non-streaming: generate all sentences sequentially, merge, return merged audio
        result_holder = {}
        start_time = time.time()

        def _run_all():
            try:
                chunks = []
                for sentence in sentences:
                    if cancel_event.is_set():
                        break
                    fmt = tts.format_script_for_single_speaker(sentence, speaker_id=0)
                    audio = tts.generate_speech(
                        text=fmt,
                        voice_samples=[voice_audio],
                        cfg_scale=settings.default_cfg_scale,
                        stream=False,
                        cancel_event=cancel_event,
                    )
                    if audio is not None:
                        chunks.append(audio)
                result_holder['audio'] = concatenate_audio_chunks(chunks) if chunks else None
            except Exception as e:
                result_holder['error'] = e

        gen_thread = threading.Thread(target=_run_all, daemon=True)
        gen_thread.start()

        while gen_thread.is_alive():
            await asyncio.sleep(0.1)
            if await request.is_disconnected():
                logger.info("Client disconnected — cancelling generation")
                cancel_event.set()
                gen_thread.join(timeout=5.0)
                return Response(status_code=499)

        generation_time = time.time() - start_time

        if 'error' in result_holder:
            raise result_holder['error']

        audio = result_holder.get('audio')
        if audio is None:
            return Response(status_code=499)

        audio_duration = get_audio_duration(audio, sample_rate=24000)
        text_preview = body.input[:100] + "..." if len(body.input) > 100 else body.input
        logger.info(
            f"Generated speech ({len(sentences)} sentences) | Voice: {body.voice} | "
            f"CFG: {settings.default_cfg_scale} | Audio: {audio_duration:.2f}s | Gen: {generation_time:.2f}s"
        )

        audio_bytes = audio_to_bytes(audio, sample_rate=24000, format=body.response_format)
        return Response(
            content=audio_bytes,
            media_type=get_content_type(body.response_format),
            headers={"Content-Disposition": f"attachment; filename=speech.{body.response_format}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating speech: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices")
async def list_voices(voices: VoiceManager = Depends(get_voice_manager)):
    """List all available voices in OpenAI-compatible format."""
    try:
        voice_list = []
        for openai_name, vibevoice_preset in voices.OPENAI_VOICE_MAPPING.items():
            if vibevoice_preset in voices.voice_presets:
                voice_list.append({"id": openai_name, "object": "voice", "name": openai_name})
        all_voices = voices.list_available_voices()
        for voice in all_voices:
            if voice["name"] not in voices.OPENAI_VOICE_MAPPING.values():
                voice_list.append({"id": voice["name"], "object": "voice", "name": voice["name"]})
        return {"object": "list", "data": voice_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
