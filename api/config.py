"""Configuration management for the VibeVoice API."""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model configuration
    vibevoice_model_path: str = Field(default="microsoft/VibeVoice-1.5B")
    vibevoice_device: str = Field(default="cuda")
    vibevoice_inference_steps: int = Field(default=10, ge=1, le=100)
    vibevoice_dtype: Optional[str] = Field(default=None)
    vibevoice_attn_implementation: Optional[str] = Field(default=None)
    torch_compile: bool = Field(default=False)
    torch_compile_mode: str = Field(default="default")
    vibevoice_quantization: Optional[str] = Field(default=None)
    vibevoice_lazy_load: bool = Field(default=False)
    vibevoice_idle_timeout_seconds: int = Field(default=300, ge=0)

    # Voice configuration
    voices_dir: str = Field(default="/app/voices")
    openai_voice_mapping: str = Field(
        default=(
            '{"alloy": "en-Alice_woman", "echo": "en-Carter_man", '
            '"fable": "en-Maya_woman", "onyx": "en-Frank_man", '
            '"nova": "en-Mary_woman_bgm", "shimmer": "en-Alice_woman"}'
        )
    )
    max_voice_sample_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)
    max_voice_sample_seconds: float = Field(default=120.0, gt=0)

    # API server configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8001, ge=1, le=65535)
    api_workers: int = Field(
        default=1,
        ge=1,
        le=1,
        description="Use one worker so model weights and GPU memory are not duplicated",
    )
    api_cors_origins: str = Field(default="*")

    # Generation defaults and safety bounds
    default_cfg_scale: float = Field(default=1.3, ge=1.0, le=3.0)
    default_response_format: str = Field(default="mp3")
    max_generation_length: int = Field(default=90 * 60, ge=1)
    vibevoice_max_new_tokens: int = Field(default=256, ge=1)
    vibevoice_min_chunk_chars: int = Field(default=1000, ge=32, le=10000)
    vibevoice_max_chunk_chars: int = Field(default=2000, ge=32, le=10000)
    vibevoice_trim_silence: bool = Field(default=True)
    default_do_sample: bool = Field(default=False)
    default_temperature: float = Field(default=1.0, gt=0)
    default_top_p: float = Field(default=1.0, gt=0, le=1.0)
    default_top_k: int = Field(default=50, ge=0)
    default_repetition_penalty: float = Field(default=1.0, gt=0)

    # Logging
    log_level: str = Field(default="INFO")

    @property
    def normalized_log_level(self) -> str:
        level = self.log_level.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        return level if level in valid else "INFO"

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "Settings":
        if self.vibevoice_min_chunk_chars > self.vibevoice_max_chunk_chars:
            raise ValueError(
                "VIBEVOICE_MIN_CHUNK_CHARS cannot exceed VIBEVOICE_MAX_CHUNK_CHARS"
            )
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        if self.api_cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]

    def get_device(self) -> str:
        """Resolve and validate the requested inference device."""
        import torch

        requested = self.vibevoice_device.strip().lower()
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                print("WARNING: CUDA requested but unavailable; falling back to CPU")
                return "cpu"
            try:
                device = torch.device(requested)
            except (TypeError, RuntimeError) as exc:
                raise ValueError(f"Invalid VIBEVOICE_DEVICE={self.vibevoice_device!r}") from exc
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {device.index} is unavailable; "
                    f"detected {torch.cuda.device_count()} device(s)"
                )
            return str(device)

        if requested == "mps":
            if not torch.backends.mps.is_available():
                print("WARNING: MPS requested but unavailable; falling back to CPU")
                return "cpu"
            return "mps"

        if requested != "cpu":
            raise ValueError("VIBEVOICE_DEVICE must be cpu, mps, cuda, or cuda:N")
        return "cpu"

    def get_dtype(self):
        """Resolve a safe dtype for the selected device."""
        import torch

        if self.vibevoice_dtype:
            allowed = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            try:
                return allowed[self.vibevoice_dtype.lower()]
            except KeyError as exc:
                raise ValueError(
                    "VIBEVOICE_DTYPE must be bfloat16, float16, float32, or unset"
                ) from exc

        device = self.get_device()
        if device.startswith("cuda"):
            supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return torch.bfloat16 if supports_bf16 else torch.float16
        return torch.float32

    def get_attn_implementation(self) -> str:
        """Choose the configured attention backend with a safe default."""
        if self.vibevoice_attn_implementation:
            implementation = self.vibevoice_attn_implementation.lower()
            if implementation not in {"flash_attention_2", "sdpa", "eager"}:
                raise ValueError(
                    "VIBEVOICE_ATTN_IMPLEMENTATION must be flash_attention_2, sdpa, eager, or unset"
                )
            return implementation

        if self.get_device().startswith("cuda"):
            try:
                import flash_attn  # noqa: F401

                return "flash_attention_2"
            except ImportError:
                pass
        return "sdpa"


settings = Settings()
