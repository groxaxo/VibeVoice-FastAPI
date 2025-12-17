# VibeVoice FastAPI Server

A production-ready FastAPI server that exposes the VibeVoice TTS model as an OpenAI-compatible API, with Docker support and comprehensive voice management.

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-orange?logo=openai)](https://platform.openai.com/docs/api-reference/audio)

</div>

## 🚀 Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's TTS API (`/v1/audio/speech`)
- **Unlimited Custom Voices**: Automatically load any voice from a directory - just drop audio files and restart
- **Multi-Format Support**: MP3, WAV, FLAC, AAC, M4A, Opus, PCM
- **Streaming Support**: Real-time audio streaming for long-form content
- **Docker Ready**: Complete Docker and docker-compose setup with GPU support
- **Voice Management**: Dynamic voice loading, OpenAI voice mapping, and custom voice presets
- **Production Ready**: Health checks, error handling, CORS support, and comprehensive logging

## 📋 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/ncoder-ai/VibeVoice-FastAPI.git
cd VibeVoice-FastAPI

# Copy and configure environment
cp docker-env.example .env
# Edit .env with your settings

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/ncoder-ai/VibeVoice-FastAPI.git
cd VibeVoice-FastAPI

# Run setup script
./setup.sh

# Configure environment
cp env.example .env
# Edit .env with your settings

# Start server
./start.sh
```

The API will be available at `http://localhost:8000`

## 📖 Documentation

- **[API README](API_README.md)** - Complete API documentation with examples
- **[Quick Start Guide](QUICKSTART.md)** - Get up and running in 5 minutes
- **[Docker Guide](DOCKER_GUIDE.md)** - Docker deployment instructions
- **[Voice Usage Guide](VOICE_USAGE_GUIDE.md)** - How to use and manage voices
- **[Custom Voices Guide](CUSTOM_VOICES_GUIDE.md)** - Adding your own voices
- **[Models Guide](MODELS_GUIDE.md)** - Available VibeVoice models

## 🎯 API Endpoints

### OpenAI-Compatible Endpoints

- `POST /v1/audio/speech` - Generate speech from text (OpenAI-compatible)
- `GET /v1/audio/voices` - List all available voices

### VibeVoice-Specific Endpoints

- `POST /v1/vibevoice/generate` - Advanced generation with multi-speaker support
- `GET /v1/vibevoice/voices` - List all voices with detailed info
- `GET /v1/vibevoice/health` - Detailed health check

### Example: Generate Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of the VibeVoice API",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Example: List Voices

```bash
# List all voices (141+ voices)
curl http://localhost:8000/v1/audio/voices

# List with OpenAI format
curl http://localhost:8000/v1/audio/voices | jq
```

## 🎤 Voice Management

### Using OpenAI-Compatible Voices

The API includes 6 OpenAI-compatible voice mappings:
- `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### Using Custom Voices

Simply place audio files (`.wav`, `.mp3`, `.flac`, `.m4a`, etc.) in your `VOICES_DIR` and restart the server. All files are automatically loaded as voice presets!

```bash
# Add a custom voice
cp my_voice.wav /path/to/voices/custom_voice.wav
# Restart server - voice is now available!
```

### Direct Voice Usage

You can use any voice name directly in API requests:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Testing custom voice",
    "voice": "custom_voice",
    "response_format": "wav"
  }'
```

## ⚙️ Configuration

Key environment variables (see `env.example` for full list):

```bash
# Model Configuration
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B  # or local path
VIBEVOICE_DEVICE=cuda                           # cuda, cpu, or mps
VIBEVOICE_INFERENCE_STEPS=10                    # 5-50, higher = better quality

# Voice Configuration
VOICES_DIR=demo/voices                           # Directory with voice files

# API Configuration
API_PORT=8000
API_CORS_ORIGINS=*

# Generation Defaults
DEFAULT_CFG_SCALE=1.3                            # 1.0-3.0
DEFAULT_RESPONSE_FORMAT=mp3
```

## 🐳 Docker Deployment

### Requirements

- Docker with NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with 8GB+ VRAM (for 1.5B model) or 16GB+ (for Large model)

### Quick Start

```bash
# Build image
docker build -t vibevoice-api:latest .

# Run container
docker run -d \
  --name vibevoice-api \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/voices:/app/voices:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface:rw \
  -e VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B \
  vibevoice-api:latest
```

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for detailed instructions.

## 🔧 Development

### Setup Development Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install -r requirements-api.txt

# Install PyTorch (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Optional: Install flash-attn for faster inference
pip install flash-attn --no-build-isolation
```

### Running Tests

```bash
# Start server
./start.sh

# Test API
curl http://localhost:8000/health
curl http://localhost:8000/v1/audio/voices
```

## 📊 Supported Models

| Model | Size | Context | Max Length | VRAM Required |
|-------|------|---------|------------|---------------|
| VibeVoice-1.5B | 1.5B | 64K | ~90 min | 8GB+ |
| VibeVoice-Large | 7B | 32K | ~45 min | 16GB+ |

Models are automatically downloaded from HuggingFace on first use.

## 🛠️ System Requirements

### Minimum
- **GPU**: NVIDIA GPU with 8GB VRAM
- **RAM**: 16GB
- **Storage**: 10GB (for model and dependencies)
- **OS**: Linux, macOS, or Windows (with WSL2)

### Recommended
- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **RAM**: 32GB
- **Storage**: 50GB (with model cache)
- **Python**: 3.10 or 3.11

## 🔐 Security Notes

- The API does not include authentication by default. For production use, add authentication middleware or deploy behind a reverse proxy with authentication.
- Voice files are loaded from the configured directory - ensure proper file permissions.
- Model weights are downloaded from HuggingFace - verify model integrity in production.

## 📝 License

This project maintains the original VibeVoice model codebase. Please refer to the original VibeVoice license for model usage terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- **VibeVoice Team** at Microsoft for the original model
- **FastAPI** for the excellent web framework
- **HuggingFace** for model hosting and transformers library

## 📚 Additional Resources

- [VibeVoice Original Paper](https://arxiv.org/pdf/2508.19205)
- [VibeVoice HuggingFace Collection](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

## ⚠️ Limitations

- **Language Support**: Primarily English and Chinese. Other languages may produce unexpected results.
- **Non-Speech Audio**: The model focuses on speech synthesis and may generate background music or sounds spontaneously.
- **Commercial Use**: This model is intended for research and development. Test thoroughly before production use.

## 🆘 Troubleshooting

### Server won't start
- Check GPU availability: `nvidia-smi`
- Verify Python version: `python3 --version` (should be 3.10 or 3.11)
- Check dependencies: `pip list | grep torch`

### Out of memory errors
- Use smaller model: `VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B`
- Reduce inference steps: `VIBEVOICE_INFERENCE_STEPS=5`
- Use CPU mode: `VIBEVOICE_DEVICE=cpu` (much slower)

### Voice not found errors
- Verify `VOICES_DIR` path in `.env`
- Check file permissions
- Ensure audio files are in supported formats

For more help, see the [API README](API_README.md) or open an issue.

---

**Made with ❤️ for the VibeVoice community**
