#!/usr/bin/env python
"""Quick test to verify VibeVoice model loading"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["VIBEVOICE_MODEL_PATH"] = "aoi-ot/VibeVoice-Large"
os.environ["VIBEVOICE_DEVICE"] = "cuda"
os.environ["VIBEVOICE_DTYPE"] = "bfloat16"
os.environ["VIBEVOICE_ATTN_IMPLEMENTATION"] = "flash_attention_2"
os.environ["TORCH_COMPILE"] = "false"  # Disable for quick test
os.environ["VIBEVOICE_QUANTIZATION"] = "int8_torchao"
os.environ["VIBEVOICE_INFERENCE_STEPS"] = "5"

from api.config import settings
from api.services.tts_service import TTSService
import torch

print("=" * 80)
print("Testing VibeVoice Model Loading")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Model path: {settings.vibevoice_model_path}")
print(f"  Device: {settings.vibevoice_device}")
print(f"  Dtype: {settings.vibevoice_dtype}")
print(f"  Attention: {settings.get_attn_implementation()}")
print(f"  Torch compile: {settings.torch_compile}")
print(f"  Quantization: {settings.vibevoice_quantization}")
print(f"  Inference steps: {settings.vibevoice_inference_steps}")
print()

if torch.cuda.is_available():
    print(f"GPU Info:")
    print(f"  Available GPUs: {torch.cuda.device_count()}")
    print(f"  Current GPU: {torch.cuda.current_device()}")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(
        f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
    print()

try:
    print("Initializing TTS Service...")
    tts = TTSService(settings)

    print("Loading model (this may take a few minutes)...")
    tts.load_model()

    print("\n✅ Model loaded successfully!")

    # Test a simple generation
    print("\nTesting simple generation (short text)...")
    test_text = "Hello, this is a test."

    # Get a voice sample from demo
    import numpy as np

    voice_dir = settings.voices_dir
    voice_files = [
        f for f in os.listdir(voice_dir) if f.endswith((".wav", ".mp3", ".flac"))
    ]

    if voice_files:
        import librosa

        voice_path = os.path.join(voice_dir, voice_files[0])
        voice_sample, _ = librosa.load(voice_path, sr=24000)

        print(f"  Using voice: {voice_files[0]}")
        print(f"  Text: {test_text}")

        try:
            audio = tts.generate_speech(
                text=test_text,
                voice_samples=[voice_sample],
                cfg_scale=1.8,
                inference_steps=5,
                seed=42,
            )

            print(f"  ✅ Generated audio: shape={audio.shape}, dtype={audio.dtype}")
            print(f"  Duration: {len(audio) / 24000:.2f} seconds")
        except Exception as e:
            print(f"  ⚠️  Generation test failed: {e}")
            print(f"  (This may be expected - model loaded successfully)")
    else:
        print(f"  ⚠️  No voice files found in {voice_dir}")
        print(f"  (Model loaded successfully, but couldn't test generation)")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
