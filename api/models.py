"""Pydantic request and response schemas for the API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "m4a"]


class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible speech request."""

    model: str = Field(default="tts-1", description="Model name accepted for OpenAI compatibility")
    input: str = Field(
        ...,
        min_length=1,
        description="Text to synthesize; long input is chunked and joined automatically",
    )
    voice: str = Field(..., min_length=1, description="OpenAI voice alias or VibeVoice preset")
    response_format: AudioFormat = Field(default="mp3", description="Audio response format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed")
    stream: bool = Field(default=False, description="Stream encoded audio as it becomes available")


class SpeakerConfig(BaseModel):
    """Voice source for one speaker."""

    speaker_id: int = Field(..., ge=0, le=3)
    voice_preset: Optional[str] = Field(default=None, min_length=1)
    voice_sample_base64: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_voice_source(self) -> "SpeakerConfig":
        sources = int(bool(self.voice_preset)) + int(bool(self.voice_sample_base64))
        if sources != 1:
            raise ValueError("Provide exactly one of voice_preset or voice_sample_base64")
        return self


class VibeVoiceGenerateRequest(BaseModel):
    """Extended generation request with up to four speakers."""

    script: str = Field(
        ...,
        min_length=1,
        description=(
            "Script using lines such as 'Speaker 0: Hello'. Long scripts are "
            "chunked and joined automatically."
        ),
    )
    speakers: list[SpeakerConfig] = Field(..., min_length=1, max_length=4)
    cfg_scale: float = Field(default=1.3, ge=1.0, le=2.0)
    inference_steps: Optional[int] = Field(default=None, ge=5, le=50)
    response_format: AudioFormat = Field(default="mp3")
    stream: bool = Field(default=False, description="Stream one encoded chunk per SSE event")
    seed: Optional[int] = Field(default=None)
    do_sample: Optional[bool] = Field(default=None)
    temperature: Optional[float] = Field(default=None, ge=0.1, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_speakers(self) -> "VibeVoiceGenerateRequest":
        speaker_ids = sorted(speaker.speaker_id for speaker in self.speakers)
        expected = list(range(len(self.speakers)))
        if speaker_ids != expected:
            raise ValueError(f"Speaker IDs must be unique and sequential from 0. Got: {speaker_ids}")
        if self.do_sample is False and (self.temperature is not None or self.top_p is not None):
            raise ValueError("temperature/top_p require do_sample=true or do_sample omitted")
        return self


class VibeVoiceGenerateResponse(BaseModel):
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    format: str
    sample_rate: int = 24000


class ErrorResponse(BaseModel):
    error: dict

    @staticmethod
    def from_exception(exc: Exception, status_code: int = 500) -> "ErrorResponse":
        return ErrorResponse(
            error={"message": str(exc), "type": type(exc).__name__, "code": status_code}
        )


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    device: str
    model_path: str
    busy: bool = False


class VoiceListResponse(BaseModel):
    voices: list[dict]
    count: int
