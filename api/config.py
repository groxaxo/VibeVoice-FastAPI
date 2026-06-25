"""Configuration management for VibeVoice API."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model Configuration
    vibevoice_model_path: str = Field(
        default="microsoft/VibeVoice-1.5B",
        description="Path to VibeVoice model weights (local path or HuggingFace model ID)"
    )
    vibevoice_device: str = Field(
        default="cuda",
        description="Device for inference: cuda, cpu, or mps"
    )
    vibevoice_inference_steps: int = Field(
        default=10,
        description="Number of diffusion inference steps"
    )
    vibevoice_dtype: Optional[str] = Field(
        default=None,
        description="Model dtype: bfloat16, float16, or float32 (auto-detected if None)"
    )
    vibevoice_attn_implementation: Optional[str] = Field(
        default=None,
        description="Attention implementation: flash_attention_2, sdpa, or eager (auto-detected if None)"
    )
    torch_compile: bool = Field(
        default=False,
        description="Enable torch.compile for optimized inference (20-50% speedup, but first request is slower due to compilation)"
    )
    vibevoice_quantization: Optional[str] = Field(
        default=None,
        description="Quantization method: 'int8_torchao', 'int4_torchao', or None for full precision"
    )
    torch_compile_mode: str = Field(
        default="default",
        description="torch.compile mode: 'default', 'reduce-overhead', or 'max-autotune' (slower compile, faster inference)"
    )

    # Model loading: EAGER by default — the model loads at startup and stays
    # resident in VRAM (lowest latency, always ready). Opt into lazy loading by
    # setting VIBEVOICE_LAZY_LOAD=true (defer load to first request + idle-unload
    # to free VRAM between bursts on a shared GPU).
    vibevoice_lazy_load: bool = Field(
        default=False,
        description="If True, do NOT load the model at startup — load it on the first request and "
        "idle-unload after vibevoice_idle_timeout_seconds. Default is False (eager): the model "
        "loads at startup and stays resident. Set VIBEVOICE_LAZY_LOAD=true to opt into lazy loading."
    )
    vibevoice_idle_timeout_seconds: int = Field(
        default=300,
        description="Only used when lazy_load is True: unload the model after this many seconds of "
        "inactivity (0 = never unload). Ignored when eager (the default)."
    )

    # Voice Configuration
    voices_dir: str = Field(
        default="/app/voices",  # Docker default; override with VOICES_DIR=demo/voices for local dev
        description="Directory containing voice preset audio files"
    )
    openai_voice_mapping: str = Field(
        default='{"alloy": "en-Alice_woman", "echo": "en-Carter_man", "fable": "en-Maya_woman", "onyx": "en-Frank_man", "nova": "en-Mary_woman_bgm", "shimmer": "en-Alice_woman"}',
        description="JSON mapping of OpenAI voice names to VibeVoice preset names"
    )
    
    # API Server Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8001,
        description="API server port"
    )
    api_workers: int = Field(
        default=1,
        description="Number of API workers (keep at 1 for model loading)"
    )
    api_cors_origins: str = Field(
        default="*",
        description="CORS allowed origins (comma-separated)"
    )
    
    # Generation Defaults
    default_cfg_scale: float = Field(
        default=1.3,
        description="Default CFG scale for generation (1.0-3.0, higher = more faithful to prompt)"
    )
    default_response_format: str = Field(
        default="mp3",
        description="Default audio response format"
    )
    max_generation_length: int = Field(
        default=90 * 60,  # 90 minutes in seconds
        description="Maximum generation length in seconds"
    )
    vibevoice_max_new_tokens: int = Field(
        default=256,
        description=(
            "Hard cap on new acoustic tokens per model.generate() call. VibeVoice's "
            "tokenizer runs at ~7.5 Hz, so 256 tokens ≈ 34s of audio per chunk. Passed as "
            "max_new_tokens — this bounds the KV-cache preallocation (the model's "
            "max_position_embeddings is 32768, which with max_new_tokens=None would "
            "preallocate a 32k-token cache and OOM a 12GB GPU). Together with the default "
            "per-sentence chunking this keeps VRAM bounded. Env: VIBEVOICE_MAX_NEW_TOKENS."
        )
    )
    vibevoice_trim_silence: bool = Field(
        default=True,
        description=(
            "Trim trailing silence from generated audio. VibeVoice intermittently fails to "
            "emit its stop token (greedy argmax flipped by FP nondeterminism) and appends "
            "seconds of silent frames up to max_new_tokens; this removes that tail. Real "
            "speech sits far above the silence threshold so content is never clipped. "
            "Env: VIBEVOICE_TRIM_SILENCE."
        )
    )
    default_do_sample: bool = Field(
        default=False,
        description="Whether to use sampling for text generation (False = greedy decoding)"
    )
    default_temperature: float = Field(
        default=1.0,
        description="Temperature for sampling (only used if do_sample=True)"
    )
    default_top_p: float = Field(
        default=1.0,
        description="Top-p (nucleus) sampling (only used if do_sample=True)"
    )
    default_top_k: int = Field(
        default=50,
        description="Top-k sampling (only used if do_sample=True)"
    )
    default_repetition_penalty: float = Field(
        default=1.0,
        description="Repetition penalty (1.0 = no penalty)"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive)"
    )
    
    @property
    def normalized_log_level(self) -> str:
        """Get log level normalized to uppercase for logging module."""
        return self.log_level.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
        
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.api_cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.api_cors_origins.split(",")]
    
    def get_device(self) -> str:
        """Get the appropriate device, checking availability."""
        import torch
        
        if self.vibevoice_device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif self.vibevoice_device == "mps" and not torch.backends.mps.is_available():
            print("WARNING: MPS requested but not available, falling back to CPU")
            return "cpu"
        return self.vibevoice_device
    
    def get_dtype(self):
        """Get the appropriate dtype for the device."""
        import torch
        
        if self.vibevoice_dtype:
            return getattr(torch, self.vibevoice_dtype)
        
        device = self.get_device()
        if device == "cuda":
            return torch.bfloat16
        elif device == "mps":
            return torch.float32
        else:
            return torch.float32
    
    def get_attn_implementation(self) -> str:
        """Get the appropriate attention implementation."""
        if self.vibevoice_attn_implementation:
            return self.vibevoice_attn_implementation
        
        device = self.get_device()
        if device == "cuda":
            # Try flash_attention_2 first, fallback to sdpa
            try:
                import flash_attn
                return "flash_attention_2"
            except ImportError:
                return "sdpa"
        else:
            return "sdpa"


# Global settings instance
settings = Settings()


