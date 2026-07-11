"""FastAPI application for the VibeVoice TTS server."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.routers import openai_tts, vibevoice
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager

logging.basicConfig(
    level=getattr(logging, settings.normalized_log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting VibeVoice API server")
    logger.info("Model path: %s", settings.vibevoice_model_path)
    logger.info("Requested device: %s", settings.vibevoice_device)
    logger.info("Voices directory: %s", settings.voices_dir)

    manager = VoiceManager(
        voices_dir=settings.voices_dir,
        openai_voice_mapping=settings.openai_voice_mapping,
    )
    service = TTSService(settings)

    if settings.vibevoice_lazy_load:
        logger.info(
            "Lazy loading enabled; model loads on demand and unloads after %ss idle",
            settings.vibevoice_idle_timeout_seconds,
        )
    else:
        service.load_model()

    openai_tts.tts_service = service
    openai_tts.voice_manager = manager
    vibevoice.tts_service = service
    vibevoice.voice_manager = manager

    logger.info("VibeVoice API ready")
    try:
        yield
    finally:
        logger.info("Shutting down VibeVoice API server")
        try:
            await asyncio.to_thread(service.unload_model, reason="server shutdown")
        except Exception:
            logger.exception("Error unloading model during shutdown")


app = FastAPI(
    title="VibeVoice TTS API",
    description="OpenAI-compatible text-to-speech API powered by VibeVoice",
    version="0.2.0",
    lifespan=lifespan,
)

cors_origins = settings.cors_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    # Browsers reject wildcard origins combined with credentialed CORS.
    allow_credentials=cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_tts.router)
app.include_router(vibevoice.router)


@app.get("/")
async def root():
    return {
        "name": "VibeVoice TTS API",
        "version": "0.2.0",
        "description": "OpenAI-compatible Text-to-Speech API powered by VibeVoice",
        "endpoints": {
            "openai_compatible": {
                "speech": "/v1/audio/speech",
                "voices": "/v1/audio/voices",
            },
            "vibevoice_extended": {
                "generate": "/v1/vibevoice/generate",
                "voices": "/v1/vibevoice/voices",
                "health": "/v1/vibevoice/health",
                "preload": "/v1/vibevoice/preload",
                "unload": "/v1/vibevoice/unload",
            },
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get("/health")
async def health():
    service = vibevoice.tts_service
    loaded = bool(service and service.is_loaded)
    busy = bool(service and service.is_busy)
    if loaded:
        status = "healthy"
    elif settings.vibevoice_lazy_load:
        status = "idle"
    else:
        status = "unavailable"
    return {
        "status": status,
        "model_loaded": loaded,
        "busy": busy,
        "lazy_load": settings.vibevoice_lazy_load,
        "idle_timeout_seconds": settings.vibevoice_idle_timeout_seconds,
        "device": str(service.device if service and service.device else settings.vibevoice_device),
    }


@app.post("/v1/vibevoice/unload")
async def unload_now():
    service = vibevoice.tts_service
    if service is None:
        return JSONResponse(status_code=503, content={"status": "no_service"})

    was_loaded = service.is_loaded
    unloaded = await asyncio.to_thread(
        service.unload_model,
        blocking=False,
        reason="manual API request",
    )
    if not unloaded:
        return JSONResponse(
            status_code=409,
            content={
                "status": "busy",
                "message": "Model is serving or queueing a request; try unload again later",
            },
        )
    return {
        "status": "unloaded" if was_loaded else "already_unloaded",
        "model_loaded": service.is_loaded,
    }


@app.post("/v1/vibevoice/preload")
async def preload_now():
    service = vibevoice.tts_service
    if service is None:
        return JSONResponse(status_code=503, content={"status": "no_service"})

    await asyncio.to_thread(service.load_model)
    return {
        "status": "loaded",
        "model_loaded": service.is_loaded,
        "device": str(service.device),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled exception while processing %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_server_error",
            }
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.normalized_log_level.lower(),
    )
