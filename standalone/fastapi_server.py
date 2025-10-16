"""
FastAPI Server for VibeVoice TTS
Provides REST API endpoints for TTS generation
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import tempfile
import soundfile as sf
import os
import logging
from pathlib import Path
from vibevoice_core import VibeVoiceCore, get_optimal_device
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VibeVoiceAPI")

# Initialize FastAPI app
app = FastAPI(
    title="VibeVoice TTS API",
    description="Text-to-Speech API using Microsoft VibeVoice",
    version="1.0.0"
)

# Add CORS middleware - Enhanced for Open WebUI compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize VibeVoice Core
models_dir = os.environ.get("VIBEVOICE_MODELS_DIR", "./models")
core = VibeVoiceCore(models_dir=models_dir)

# Pydantic models
class TTSRequest(BaseModel):
    text: str
    model_name: str
    attention_type: str = "auto"
    quantize_llm: str = "full precision"
    diffusion_steps: int = 10  # Default steps for good quality/speed balance
    seed: int = 42
    cfg_scale: float = 1.3
    use_sampling: bool = False
    temperature: float = 0.95
    top_p: float = 0.95
    voice_speed_factor: float = 1.0
    free_memory: bool = True

class MultiSpeakerTTSRequest(BaseModel):
    text: str
    model_name: str
    attention_type: str = "auto"
    quantize_llm: str = "full precision"
    diffusion_steps: int = 10  # Default steps for good quality/speed balance
    seed: int = 42
    cfg_scale: float = 1.3
    use_sampling: bool = False
    temperature: float = 0.95
    top_p: float = 0.95
    voice_speed_factor: float = 1.0
    free_memory: bool = True

# OpenAI-compatible models
class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible TTS request"""
    model: Literal["tts-1", "tts-1-hd", "vibevoice-1.5b", "vibevoice-large"] = Field(
        default="tts-1-hd",
        description="Model to use: tts-1 (fast), tts-1-hd (quality), vibevoice-1.5b, vibevoice-large"
    )
    input: str = Field(..., description="The text to generate audio for", max_length=4096)
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = Field(
        default="alloy",
        description="Voice preset (mapped to seeds)"
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="Audio format"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed of generated audio")

# Voice presets mapping to seeds for consistency
VOICE_PRESETS = {
    "alloy": 42,
    "echo": 123,
    "fable": 456,
    "onyx": 789,
    "nova": 999,
    "shimmer": 1337
}

# Model mapping
MODEL_MAPPING = {
    "tts-1": "VibeVoice-1.5B",
    "tts-1-hd": "VibeVoice-Large",
    "vibevoice-1.5b": "VibeVoice-1.5B",
    "vibevoice-large": "VibeVoice-Large"
}

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "VibeVoice TTS API",
        "version": "1.0.0",
        "device": get_optimal_device(),
        "models_directory": models_dir,
        "openai_compatible": True,
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "tts_single": "/tts/single",
            "tts_multi": "/tts/multi",
            "openai_tts": "/v1/audio/speech",
            "openai_models": "/v1/models",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": get_optimal_device(),
        "models_loaded": core.model is not None
    }

@app.get("/models")
async def list_models():
    """List available models"""
    models = core.get_available_models()
    return {
        "models": models,
        "count": len(models),
        "models_directory": models_dir
    }

@app.post("/tts/single")
async def generate_single_speaker(
    request: TTSRequest,
    voice_file: Optional[UploadFile] = File(None)
):
    """Generate single speaker TTS"""
    try:
        # Validate request
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Load model
        core.load_model(
            model_name=request.model_name,
            attention_type=request.attention_type,
            quantize_llm=request.quantize_llm
        )
        
        # Handle voice file
        voice_samples = None
        if voice_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                content = await voice_file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            voice_sample = core._prepare_audio_from_file(
                tmp_path,
                speed_factor=request.voice_speed_factor
            )
            os.unlink(tmp_path)
            
            if voice_sample is not None:
                voice_samples = [voice_sample]
        
        # Generate speech
        audio_array, sample_rate = core.generate_speech(
            text=request.text,
            voice_samples=voice_samples,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            diffusion_steps=request.diffusion_steps,
            use_sampling=request.use_sampling,
            temperature=request.temperature,
            top_p=request.top_p,
            num_speakers=1
        )
        
        # Free memory if requested
        if request.free_memory:
            core.free_memory()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio_array, sample_rate)
            output_path = tmp.name
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_output.wav",
            headers={
                "X-Word-Count": str(len(request.text.split())),
                "X-Sample-Rate": str(sample_rate),
                "X-Duration-Seconds": str(len(audio_array) / sample_rate)
            }
        )
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/multi")
async def generate_multi_speaker(
    request: MultiSpeakerTTSRequest,
    speaker1_voice: Optional[UploadFile] = File(None),
    speaker2_voice: Optional[UploadFile] = File(None),
    speaker3_voice: Optional[UploadFile] = File(None),
    speaker4_voice: Optional[UploadFile] = File(None)
):
    """Generate multi-speaker TTS"""
    try:
        # Validate request
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Detect speakers from text
        import re
        bracket_pattern = r'\[(\d+)\]\s*:'
        speakers_numbers = sorted(list(set([int(m) for m in re.findall(bracket_pattern, request.text)])))
        
        if not speakers_numbers:
            raise HTTPException(
                status_code=400,
                detail="No speaker labels found. Use format: [1]: Text [2]: More text"
            )
        
        num_speakers = max(speakers_numbers)
        if num_speakers > 4:
            raise HTTPException(status_code=400, detail="Maximum 4 speakers supported")
        
        # Load model
        core.load_model(
            model_name=request.model_name,
            attention_type=request.attention_type,
            quantize_llm=request.quantize_llm
        )
        
        # Handle voice files
        voice_files_uploaded = [speaker1_voice, speaker2_voice, speaker3_voice, speaker4_voice]
        voice_samples = []
        
        for i, speaker_num in enumerate(speakers_numbers):
            idx = speaker_num - 1
            voice_sample = None
            
            if idx < len(voice_files_uploaded) and voice_files_uploaded[idx]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    content = await voice_files_uploaded[idx].read()
                    tmp.write(content)
                    tmp_path = tmp.name
                
                voice_sample = core._prepare_audio_from_file(
                    tmp_path,
                    speed_factor=request.voice_speed_factor
                )
                os.unlink(tmp_path)
            
            if voice_sample is None:
                voice_sample = core._create_synthetic_voice_sample(idx)
            
            voice_samples.append(voice_sample)
        
        # Convert [N]: format to Speaker N: format
        converted_text = request.text
        for speaker_num in sorted(speakers_numbers, reverse=True):
            pattern = f'\\[{speaker_num}\\]\\s*:'
            replacement = f'Speaker {speaker_num}:'
            converted_text = re.sub(pattern, replacement, converted_text)
        
        # Generate speech
        audio_array, sample_rate = core.generate_speech(
            text=converted_text,
            voice_samples=voice_samples,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            diffusion_steps=request.diffusion_steps,
            use_sampling=request.use_sampling,
            temperature=request.temperature,
            top_p=request.top_p,
            num_speakers=num_speakers
        )
        
        # Free memory if requested
        if request.free_memory:
            core.free_memory()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio_array, sample_rate)
            output_path = tmp.name
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_multi_output.wav",
            headers={
                "X-Num-Speakers": str(num_speakers),
                "X-Word-Count": str(len(request.text.split())),
                "X-Sample-Rate": str(sample_rate),
                "X-Duration-Seconds": str(len(audio_array) / sample_rate)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-speaker TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/free")
async def free_memory():
    """Free loaded models from memory"""
    try:
        core.free_memory()
        return {"status": "success", "message": "Memory freed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# OpenAI-compatible endpoints
@app.post("/v1/audio/speech")
async def openai_tts(request: OpenAITTSRequest):
    """
    OpenAI-compatible TTS endpoint
    Mimics OpenAI's /v1/audio/speech API
    Compatible with Open WebUI
    """
    try:
        # Validate input
        if not request.input or not request.input.strip():
            logger.error("Empty input text received")
            raise HTTPException(status_code=400, detail="Input text is required and cannot be empty")
        
        # Map model name
        model_name = MODEL_MAPPING.get(request.model, "VibeVoice-Large")
        logger.info(f"TTS Request - Model: {request.model} -> {model_name}, Voice: {request.voice}, Input length: {len(request.input)}")
        
        # Map voice to seed
        seed = VOICE_PRESETS.get(request.voice, 42)
        
        # Load model with error handling
        try:
            core.load_model(
                model_name=model_name,
                attention_type="auto",
                quantize_llm="full precision"
            )
        except Exception as model_error:
            logger.error(f"Model loading failed: {model_error}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(model_error)}")
        
        # Generate speech with error handling
        try:
            audio_array, sample_rate = core.generate_speech(
                text=request.input,
                voice_samples=None,
                cfg_scale=1.3,
                seed=seed,
                diffusion_steps=10,  # Consistent 10 steps for all models
                use_sampling=False,
                temperature=0.95,
                top_p=0.95,
                num_speakers=1
            )
        except Exception as gen_error:
            logger.error(f"Speech generation failed: {gen_error}")
            # Free memory on error
            try:
                core.free_memory()
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(gen_error)}")
        
        # Free memory after generation
        try:
            core.free_memory()
        except Exception as mem_error:
            logger.warning(f"Memory cleanup warning: {mem_error}")
        
        # Convert audio to requested format
        buffer = io.BytesIO()
        
        try:
            if request.response_format == "wav":
                sf.write(buffer, audio_array, sample_rate, format='WAV')
                media_type = "audio/wav"
            elif request.response_format == "flac":
                sf.write(buffer, audio_array, sample_rate, format='FLAC')
                media_type = "audio/flac"
            elif request.response_format == "pcm":
                # Raw PCM data
                buffer = io.BytesIO(audio_array.tobytes())
                media_type = "audio/pcm"
            else:
                # For mp3, opus, aac - default to WAV for Open WebUI compatibility
                sf.write(buffer, audio_array, sample_rate, format='WAV')
                media_type = "audio/wav"
                logger.info(f"Format {request.response_format} converted to WAV for compatibility")
        except Exception as format_error:
            logger.error(f"Audio format conversion failed: {format_error}")
            raise HTTPException(status_code=500, detail=f"Audio format conversion failed: {str(format_error)}")
        
        buffer.seek(0)
        
        logger.info(f"TTS generation successful - Size: {len(buffer.getvalue())} bytes")
        
        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={
                "Content-Disposition": f"inline; filename=speech.{request.response_format}",
                "X-Model": model_name,
                "X-Voice": request.voice,
                "X-Sample-Rate": str(sample_rate),
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected OpenAI TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/v1/models")
async def openai_list_models():
    """
    OpenAI-compatible models endpoint
    Lists available TTS models
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1699046400,
                "owned_by": "vibevoice",
                "description": "Fast TTS model (VibeVoice-1.5B)"
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1699046400,
                "owned_by": "vibevoice",
                "description": "High quality TTS model (VibeVoice-Large)"
            },
            {
                "id": "vibevoice-1.5b",
                "object": "model",
                "created": 1699046400,
                "owned_by": "microsoft",
                "description": "VibeVoice 1.5B model"
            },
            {
                "id": "vibevoice-large",
                "object": "model",
                "created": 1699046400,
                "owned_by": "microsoft",
                "description": "VibeVoice Large model"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="VibeVoice FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--models-dir", default="./models", help="Models directory path")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    if args.models_dir:
        os.environ["VIBEVOICE_MODELS_DIR"] = args.models_dir
    
    uvicorn.run(
        "fastapi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
