# VibeVoice API - Quick Start Guide

Get the VibeVoice TTS API running in 3 simple steps!

## Prerequisites

- **Python 3.10 or 3.11** (required for flash-attn)
- **CUDA 11.8+ or 12.x** (optional, for GPU acceleration)
- **8GB+ RAM** (16GB+ recommended for 7B model)
- **ffmpeg** (for audio format conversion)

## Installation

### Step 1: Run Setup Script

```bash
./setup.sh
```

This will:
- ✅ Check Python version (3.10 or 3.11 required)
- ✅ Create virtual environment
- ✅ Install PyTorch with CUDA support (if available)
- ✅ Install flash-attn (optional, for better performance)
- ✅ Install VibeVoice and API dependencies
- ✅ Create `.env` configuration file

**Expected time**: 5-10 minutes (depending on internet speed)

### Step 2: Configure (Optional)

Edit `.env` file if you want to change defaults:

```bash
# Use 1.5B model (default, faster)
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B

# Or use 7B model (better quality, requires more VRAM)
# VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-Large

# Device selection
VIBEVOICE_DEVICE=cuda  # or cpu, mps

# Server settings
API_PORT=8000
```

### Step 3: Start Server

```bash
./start.sh
```

The model will download automatically on first run (~3GB for 1.5B, ~14GB for 7B).

**Server will be ready at**: http://localhost:8000

## First API Call

### Test with curl (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This is VibeVoice speaking.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output hello.mp3
```

Play the audio:
```bash
# Linux
mpg123 hello.mp3

# macOS
afplay hello.mp3

# Windows
start hello.mp3
```

### Test with Python

```python
import requests

# Generate speech
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Welcome to VibeVoice API!",
        "voice": "alloy",
        "response_format": "mp3"
    }
)

# Save audio file
with open("output.mp3", "wb") as f:
    f.write(response.content)

print("✅ Audio generated: output.mp3")
```

## Multi-Speaker Example

```python
import requests

# Generate a 2-speaker dialogue
response = requests.post(
    "http://localhost:8000/v1/vibevoice/generate",
    json={
        "script": """Speaker 0: Welcome to our podcast about AI!
Speaker 1: Thanks for having me. I'm excited to discuss the latest developments.""",
        "speakers": [
            {"speaker_id": 0, "voice_preset": "en-Alice_woman"},
            {"speaker_id": 1, "voice_preset": "en-Carter_man"}
        ],
        "cfg_scale": 1.3,
        "response_format": "mp3"
    }
)

with open("podcast.mp3", "wb") as f:
    f.write(response.content)

print("✅ Podcast generated: podcast.mp3")
```

## Interactive API Documentation

Open your browser and visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Try the endpoints interactively!

## Available Voices

### OpenAI-Compatible Voices

- `alloy` - Female voice (Alice)
- `echo` - Male voice (Carter)
- `fable` - Female voice (Maya)
- `onyx` - Male voice (Frank)
- `nova` - Female voice with BGM (Mary)
- `shimmer` - Chinese female voice (Xinran)

### VibeVoice Presets

Check available voices:
```bash
curl http://localhost:8000/v1/vibevoice/voices
```

## Troubleshooting

### "Model not loaded" error

Wait for the model to finish loading (check terminal output). First load takes 30-60 seconds.

### CUDA out of memory

Use the 1.5B model instead:
```bash
# Edit .env
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B
```

Or reduce inference steps:
```bash
VIBEVOICE_INFERENCE_STEPS=5
```

### "flash-attn installation failed"

This is optional. The API will use SDPA fallback automatically. Performance will still be good.

### "ffmpeg not found"

Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Next Steps

- 📖 Read [API_README.md](API_README.md) for detailed documentation
- 🔧 Explore [API_IMPLEMENTATION_SUMMARY.md](API_IMPLEMENTATION_SUMMARY.md) for technical details
- 🌐 Visit http://localhost:8000/docs for interactive API testing
- 🎙️ Try different voices and parameters
- 🚀 Integrate with your application

## Support

- **Issues**: https://github.com/shijincai/VibeVoice/issues
- **Docs**: https://microsoft.github.io/VibeVoice

---

**Happy voice generation! 🎉**


