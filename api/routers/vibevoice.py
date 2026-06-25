"""VibeVoice-specific TTS endpoints with multi-speaker support."""

import asyncio
import base64
import logging
import re
import threading
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import Response

from api.models import (
    VibeVoiceGenerateRequest,
    VibeVoiceGenerateResponse,
    VoiceListResponse,
    HealthResponse
)
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager
from api.utils.audio_utils import audio_to_bytes, get_audio_duration, concatenate_audio_chunks, get_content_type
from api.utils.streaming import create_streaming_response
from api.config import settings

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/v1/vibevoice", tags=["VibeVoice Extended"])

# Global service instances (initialized in main.py)
tts_service: TTSService = None
voice_manager: VoiceManager = None


def get_tts_service() -> TTSService:
    """Dependency to get TTS service. When lazy-load is enabled the model is
    loaded here on the first request — this is the single entry point that
    triggers the actual load. Subsequent requests are a fast no-op."""
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    if not tts_service.is_loaded:
        if not settings.vibevoice_lazy_load:
            raise HTTPException(status_code=503, detail="TTS service not ready")
        logger.info("Lazy-load: triggering model load on first request...")
        # Synchronous load — happens once per idle window. The first request
        # pays the 30-40 s load cost, every subsequent request is fast.
        tts_service.load_model()
    return tts_service


def get_voice_manager() -> VoiceManager:
    """Dependency to get voice manager."""
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="Voice manager not initialized")
    return voice_manager


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


_SPEAKER_RE = re.compile(r'^\s*Speaker\s+(\d+)\s*:\s*(.*)$', re.IGNORECASE | re.DOTALL)


def _parse_script_to_chunks(script: str, num_speakers: int) -> list:
    """Split a (possibly multi-speaker) script into ordered (speaker_idx, sentence)
    chunks, one sentence per chunk.

    Each ``Speaker N:`` line is attributed to speaker N (clamped to the available
    voices); lines without a speaker label default to speaker 0. Every line's text
    is further split on sentence boundaries so each model.generate() call processes
    at most one sentence. This bounds the KV cache / VRAM and is the default
    anti-OOM behaviour for the native endpoint (mirrors the OpenAI-compatible one).
    """
    chunks = []
    for raw_line in script.replace('\r\n', '\n').split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        m = _SPEAKER_RE.match(line)
        if m:
            idx = int(m.group(1))
            text = m.group(2).strip()
        else:
            idx = 0
            text = line
        idx = max(0, min(idx, num_speakers - 1)) if num_speakers > 0 else 0
        for sentence in _split_sentences(text):
            if sentence:
                chunks.append((idx, sentence))
    return chunks


@router.post("/generate")
async def generate_speech(
    body: VibeVoiceGenerateRequest,
    request: Request,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    Generate multi-speaker speech with VibeVoice-specific features.

    The script is auto-split at sentence boundaries (. ! ? ;) and generated one
    sentence at a time (each bounded by VIBEVOICE_MAX_NEW_TOKENS), so a long
    script cannot preallocate a huge KV cache and OOM the GPU. Each sentence is
    rendered with its own speaker's voice and the pieces are streamed / merged in
    order.

    Supports:
    - Multi-speaker dialogue (up to 4 speakers; voice picked per sentence)
    - Custom voice samples via base64 or presets
    - CFG scale control
    - Inference step control
    - Real-time streaming via SSE (one sentence per event)
    - Cooperative cancellation on client disconnect via stop_check_fn
    """
    try:
        # Load voice samples for each speaker
        voice_samples = []

        for speaker_config in sorted(body.speakers, key=lambda s: s.speaker_id):
            if speaker_config.voice_sample_base64:
                try:
                    audio_bytes = base64.b64decode(speaker_config.voice_sample_base64)
                    import io
                    import soundfile as sf
                    audio_data, sr = sf.read(io.BytesIO(audio_bytes))

                    if sr != 24000:
                        import librosa
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=24000)

                    if len(audio_data.shape) > 1:
                        import numpy as np
                        audio_data = np.mean(audio_data, axis=1)

                    voice_samples.append(audio_data.astype('float32'))

                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to decode voice sample for speaker {speaker_config.speaker_id}: {str(e)}"
                    )

            elif speaker_config.voice_preset:
                audio_data = voices.load_voice_audio(speaker_config.voice_preset, is_openai_voice=False)

                if audio_data is None:
                    available_voices = [v["name"] for v in voices.list_available_voices()]
                    raise HTTPException(
                        status_code=400,
                        detail=f"Voice preset '{speaker_config.voice_preset}' not found. Available: {', '.join(available_voices)}"
                    )

                voice_samples.append(audio_data)

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Speaker {speaker_config.speaker_id} must have either voice_preset or voice_sample_base64"
                )

        voice_list = []
        for speaker_config in sorted(body.speakers, key=lambda s: s.speaker_id):
            if speaker_config.voice_preset:
                voice_list.append(f"speaker{speaker_config.speaker_id}={speaker_config.voice_preset}")
            else:
                voice_list.append(f"speaker{speaker_config.speaker_id}=base64_audio")
        voices_str = ", ".join(voice_list)

        actual_inference_steps = body.inference_steps if body.inference_steps is not None else settings.vibevoice_inference_steps

        # Default behaviour: split the script into per-speaker, per-sentence chunks
        # so each generation is short (bounded by VIBEVOICE_MAX_NEW_TOKENS) and the
        # GPU can't OOM on a long script.
        chunks = _parse_script_to_chunks(body.script, num_speakers=len(voice_samples))
        if not chunks:
            raise HTTPException(status_code=400, detail="No speakable text found in script")

        def _gen_one(speaker_idx: int, sentence: str, cancel_event: threading.Event):
            """Render a single sentence with its speaker's voice (bounded length)."""
            fmt = tts.format_script_for_single_speaker(sentence, speaker_id=0)
            return tts.generate_speech(
                text=fmt,
                voice_samples=[voice_samples[speaker_idx]],
                cfg_scale=body.cfg_scale,
                inference_steps=body.inference_steps,
                seed=body.seed,
                stream=False,
                cancel_event=cancel_event,
            )

        if body.stream:
            # Streaming path: generate one sentence at a time and yield each as its
            # own SSE event. cancel_event is tripped by create_streaming_response on
            # client disconnect and is also passed into every per-sentence generate.
            cancel_event = threading.Event()

            text_preview = body.script[:100] + "..." if len(body.script) > 100 else body.script
            logger.info(
                f"Generating speech (streaming, {len(chunks)} chunk(s)) - Text: {text_preview} | "
                f"Voices: {voices_str} | Model: {settings.vibevoice_model_path} | "
                f"CFG: {body.cfg_scale} | Steps: {actual_inference_steps} | "
                f"Seed: {body.seed if body.seed is not None else 'None'}"
            )

            def _sentence_audio_iter():
                for i, (sp_idx, sentence) in enumerate(chunks):
                    if cancel_event.is_set():
                        break
                    audio = _gen_one(sp_idx, sentence, cancel_event)
                    if cancel_event.is_set():
                        break
                    if audio is not None:
                        logger.info(
                            f"Streaming chunk {i+1}/{len(chunks)} (speaker {sp_idx}): {sentence[:60]!r}"
                        )
                        yield audio

            return create_streaming_response(
                _sentence_audio_iter(),
                format=body.response_format,
                sample_rate=24000,
                use_sse=True,
                cancel_event=cancel_event,
                request=request,
            )

        else:
            # Non-streaming: generate each sentence, concatenate, return one file.
            cancel_event = threading.Event()
            result_holder: dict = {}

            def _run_generation():
                try:
                    audio_parts = []
                    for sp_idx, sentence in chunks:
                        if cancel_event.is_set():
                            break
                        audio = _gen_one(sp_idx, sentence, cancel_event)
                        if audio is not None:
                            audio_parts.append(audio)
                    result_holder['audio'] = concatenate_audio_chunks(audio_parts) if audio_parts else None
                except Exception as e:
                    result_holder['error'] = e

            start_time = time.time()
            gen_thread = threading.Thread(target=_run_generation, daemon=True)
            gen_thread.start()

            while gen_thread.is_alive():
                await asyncio.sleep(0.1)
                if await request.is_disconnected():
                    logger.info("VibeVoice native client disconnected — cancelling generation")
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

            text_preview = body.script[:100] + "..." if len(body.script) > 100 else body.script
            logger.info(
                f"Generated speech ({len(chunks)} chunk(s)) - Text: {text_preview} | Voices: {voices_str} | "
                f"Model: {settings.vibevoice_model_path} | CFG: {body.cfg_scale} | "
                f"Steps: {actual_inference_steps} | Seed: {body.seed if body.seed is not None else 'None'} | "
                f"Audio Duration: {audio_duration:.2f}s | Generation Time: {generation_time:.2f}s"
            )

            audio_bytes = audio_to_bytes(
                audio,
                sample_rate=24000,
                format=body.response_format
            )

            return Response(
                content=audio_bytes,
                media_type=get_content_type(body.response_format),
                headers={
                    "Content-Disposition": f"attachment; filename=vibevoice_output.{body.response_format}",
                    "X-Audio-Duration": str(audio_duration),
                    "X-Audio-Format": body.response_format,
                    "X-Audio-Sample-Rate": "24000"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices", response_model=VoiceListResponse)
async def list_vibevoice_voices(
    voices: VoiceManager = Depends(get_voice_manager)
):
    """List all available voices for VibeVoice."""
    try:
        all_voices = voices.list_available_voices()
        return VoiceListResponse(voices=all_voices, count=len(all_voices))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def vibevoice_health(
    tts: TTSService = Depends(get_tts_service)
):
    """Health check for the VibeVoice service."""
    return HealthResponse(
        status="healthy" if tts.is_loaded else "loading",
        model_loaded=tts.is_loaded,
        model_path=settings.vibevoice_model_path,
        device=tts.device or "unknown"
    )
