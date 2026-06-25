"""FastAPI application for VibeVoice TTS API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager
from api.routers import openai_tts, vibevoice


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.normalized_log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting VibeVoice API server...")
    logger.info(f"Model path: {settings.vibevoice_model_path}")
    logger.info(f"Device: {settings.vibevoice_device}")
    logger.info(f"Voices directory: {settings.voices_dir}")
    
    # Initialize voice manager
    logger.info("Initializing voice manager...")
    voice_manager = VoiceManager(
        voices_dir=settings.voices_dir,
        openai_voice_mapping=settings.openai_voice_mapping
    )
    
    # Initialize TTS service
    logger.info("Initializing TTS service...")
    tts_service = TTSService(settings)

    # Load model — skip on startup if lazy_load is enabled. The first
    # generate_speech() request will trigger the actual load.
    if settings.vibevoice_lazy_load:
        logger.info(
            "Lazy-load enabled: model will load on first request "
            f"(idle-unload after {settings.vibevoice_idle_timeout_seconds}s)."
        )
    else:
        logger.info("Loading VibeVoice model (this may take a few minutes)...")
        try:
            tts_service.load_model()
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise

    # Set global service instances in routers
    openai_tts.tts_service = tts_service
    openai_tts.voice_manager = voice_manager
    vibevoice.tts_service = tts_service
    vibevoice.voice_manager = voice_manager

    logger.info("API server ready!")

    yield

    # Shutdown
    logger.info("Shutting down VibeVoice API server...")
    try:
        tts_service.unload_model()
    except Exception as e:
        logger.warning(f"Error unloading model on shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="VibeVoice TTS API",
    description="OpenAI-compatible Text-to-Speech API powered by VibeVoice",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(openai_tts.router)
app.include_router(vibevoice.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "VibeVoice TTS API",
        "version": "0.1.0",
        "description": "OpenAI-compatible Text-to-Speech API powered by VibeVoice",
        "endpoints": {
            "openai_compatible": {
                "speech": "/v1/audio/speech",
                "voices": "/v1/audio/voices"
            },
            "vibevoice_extended": {
                "generate": "/v1/vibevoice/generate",
                "voices": "/v1/vibevoice/voices",
                "health": "/v1/vibevoice/health"
            },
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    from api.routers import vibevoice as _vibevoice_router
    loaded = bool(_vibevoice_router.tts_service and _vibevoice_router.tts_service.is_loaded)
    return {
        "status": "healthy",
        "model_loaded": loaded,
        "lazy_load": settings.vibevoice_lazy_load,
        "idle_timeout_seconds": settings.vibevoice_idle_timeout_seconds,
    }


@app.post("/v1/vibevoice/unload")
async def unload_now():
    """Manually unload the model from VRAM. Useful before long GPU-heavy
    jobs (training, gaming) or for testing the lazy-load path."""
    from api.routers import vibevoice as _vibevoice_router
    svc = _vibevoice_router.tts_service
    if svc is None:
        return {"status": "no_service"}
    was_loaded = svc.is_loaded
    svc.unload_model()
    return {
        "status": "unloaded" if was_loaded else "already_unloaded",
        "model_loaded": svc.is_loaded,
    }


@app.post("/v1/vibevoice/preload")
async def preload_now():
    """Force-load the model now (instead of waiting for the first request).
    Returns when the model is in VRAM and ready."""
    from api.routers import vibevoice as _vibevoice_router
    svc = _vibevoice_router.tts_service
    if svc is None:
        return JSONResponse(status_code=503, content={"status": "no_service"})
    if not svc.is_loaded:
        svc.load_model()
    return {"status": "loaded", "model_loaded": svc.is_loaded}


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": type(exc).__name__,
                "detail": str(exc)
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.normalized_log_level.lower()
    )

