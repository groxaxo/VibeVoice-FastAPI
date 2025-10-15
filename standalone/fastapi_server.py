"""
FastAPI Server for VibeVoice TTS
Provides REST API endpoints for TTS generation
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import soundfile as sf
import os
import logging
from pathlib import Path
from vibevoice_core import VibeVoiceCore, get_optimal_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VibeVoiceAPI")

# Initialize FastAPI app
app = FastAPI(
    title="VibeVoice TTS API",
    description="Text-to-Speech API using Microsoft VibeVoice",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    diffusion_steps: int = 20
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
    diffusion_steps: int = 20
    seed: int = 42
    cfg_scale: float = 1.3
    use_sampling: bool = False
    temperature: float = 0.95
    top_p: float = 0.95
    voice_speed_factor: float = 1.0
    free_memory: bool = True

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "VibeVoice TTS API",
        "version": "1.0.0",
        "device": get_optimal_device(),
        "models_directory": models_dir,
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "tts_single": "/tts/single",
            "tts_multi": "/tts/multi",
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
