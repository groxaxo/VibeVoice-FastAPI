# VibeVoice FastAPI Server

A production-ready FastAPI server that exposes the VibeVoice TTS model as an OpenAI-compatible API, with Docker support and comprehensive voice management.

**🚀 Now with Ampere GPU Optimizations!** Optimized for RTX 3090, 3080, 3070, A100, and A40 with up to 50% performance improvements.

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-orange?logo=openai)](https://platform.openai.com/docs/api-reference/audio)
[![Ampere Optimized](https://img.shields.io/badge/Ampere-Optimized-green?logo=nvidia)](https://developer.nvidia.com/ampere-computing-architecture)

</div>

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-orange?logo=openai)](https://platform.openai.com/docs/api-reference/audio)

</div>

## 🚀 Features

- **Docker-First Deployment**: Production-ready Docker setup with GPU support (recommended)
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's TTS API (`/v1/audio/speech`)
- **Auto Language Detection**: Automatic language detection from input text with `language=auto` (supports 50+ languages)
- **Unlimited Custom Voices**: Automatically load any voice from a directory - just drop audio files and restart
- **Multi-Format Support**: MP3, WAV, FLAC, AAC, M4A, Opus, PCM
- **Streaming Support**: Real-time audio streaming for long-form content
- **Voice Management**: Dynamic voice loading, OpenAI voice mapping, and custom voice presets
- **Production Ready**: Health checks, error handling, CORS support, and comprehensive logging
- **Dependency Compatibility**: Automatic compatibility layer for different transformers versions
- **Smart GPU Autodetection**: Automatically selects the GPU with the most free memory to prevent OOM errors

## ⚡ Ampere GPU Optimizations

**Special optimizations for NVIDIA Ampere architecture (RTX 3090, 3080, 3070, A100, A40)** that achieve up to 50% performance improvements.

| Optimization | Impact | VRAM Impact | Status |
|-------------|---------|-------------|---------|
| **INT8 Quantization** | 2-3x faster memory access | -40% (10.8 GB vs 16GB) | ✅ Enabled |
| **Flash Attention 2** | 2-3x faster attention layers | - | ✅ Pre-built |
| **torch.compile (max-autotune)** | 20-50% speedup | Slight increase | ✅ Enabled |
| **bfloat16 Precision** | 2x faster than float32 | - | ✅ Native Ampere |
| **Reduced Steps (5)** | 30% faster | - | ✅ Configured |

### Performance Benchmarks (RTX 3090)

```
Model: VibeVoice-Large (7B parameters)
VRAM: 10.84 GB / 24 GB (INT8 quantized)
Real-time Factor: 0.77x (1.31x slower than real-time)

Generation Times:
• 10s audio  → ~13.1s to generate
• 1min audio → ~78.4s (1.3 min) to generate
• 5min audio → ~6.5 min to generate
```

### Quick Start with Optimizations

**Docker (Ampere Optimized):**
```bash
# Build optimized image
docker build -f Dockerfile.ampere -t vibevoice-ampere .

# Run with Docker Compose
docker-compose -f docker-compose.ampere.yml up -d
```

**Conda:**
```bash
# Create environment
conda create -n vibevoice python=3.11 -y
conda activate vibevoice

# Install optimized dependencies
pip install -r requirements-ampere.txt

# Copy optimized configuration
cp .env.ampere .env

# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

📖 **For complete Ampere optimization guide, see [AMPERE_OPTIMIZED_DEPLOYMENT.md](AMPERE_OPTIMIZED_DEPLOYMENT.md)**

## 📋 Quick Start

**Docker is the recommended deployment method** - it handles all dependencies, ensures consistent environments, and is production-ready.

### Dependency Requirements

The project requires specific versions of key dependencies:
- `transformers==4.51.3` - Required for FlashAttentionKwargs support
- `langdetect>=1.0.9` - For automatic language detection

If you encounter import errors, ensure you have the correct transformers version:
```bash
pip install transformers==4.51.3
```

### Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/ncoder-ai/VibeVoice-FastAPI.git
cd VibeVoice-FastAPI

# Copy and configure environment
cp docker-env.example .env
# Edit .env - set VOICES_DIR to your voice files path

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

The API will be available at `http://localhost:8001`

See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for detailed Docker instructions.

### Local Installation (Alternative)

For development or if you prefer bare-metal installation:

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

## 📖 Documentation

- **[AMPERE_OPTIMIZED_DEPLOYMENT.md](AMPERE_OPTIMIZED_DEPLOYMENT.md)** - ⭐ **NEW: Complete Ampere GPU optimization guide** with performance benchmarks, tuning options, and Docker support
- **[API README](API_README.md)** - Complete API documentation with examples, voice management, and troubleshooting
- **[Docker Quickstart](DOCKER_QUICKSTART.md)** - Docker deployment quickstart guide

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
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of the VibeVoice API",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Example: Auto-Detect Language

```bash
# Auto-detect language (default)
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Buenos días, ¿cómo estás?",
    "voice": "alloy",
    "response_format": "mp3",
    "language": "auto"
  }' \
  --output speech.mp3

# Explicit language specification
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Bonjour, comment allez-vous?",
    "voice": "alloy",
    "response_format": "mp3",
    "language": "fr"
  }' \
  --output speech.mp3
```

**Language Detection Notes:**
- Use `language="auto"` for automatic detection (default behavior)
- For best accuracy with short text, use explicit language codes: `es`, `en`, `fr`, `de`, `it`, `pt`, `zh`, `ja`, `ko`, `ar`, `hi`, `ru`, and more
- The detected language is returned in the `X-Detected-Language` response header
- Very short text (less than 10 characters) may have lower detection accuracy

### Example: List Voices

```bash
# List all available voices
curl http://localhost:8001/v1/audio/voices

# List with OpenAI format
curl http://localhost:8001/v1/audio/voices | jq
```
## MODEL MANAGEMENT
VibeeVoice Large: Huggingface: aoi-ot/VibeVoice-Large [https://huggingface.co/aoi-ot/VibeVoice-Large] ⭐ Optimized for Ampere GPUs

VibeVoice 1.5B: Huggingface microsoft/VibeVoice-1.5B [https://huggingface.co/microsoft/VibeVoice-1.5B]
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
curl -X POST http://localhost:8001/v1/audio/speech \
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
API_PORT=8001
API_CORS_ORIGINS=*

# Generation Defaults
DEFAULT_CFG_SCALE=1.3                            # 1.0-3.0
DEFAULT_RESPONSE_FORMAT=mp3
```

### API Request Parameters

- `model`: Model to use (`tts-1` or `tts-1-hd`)
- `input`: Text to convert to speech (max 4096 characters)
- `voice`: Voice name (OpenAI voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` or custom voices)
- `language`: Language of input text (`"auto"` for auto-detection, or explicit code like `"es"`, `"en"`, `"fr"`, `"de"`, `"it"`, `"pt"`, `"zh"`, `"ja"`, `"ko"`, `"ar"`, `"hi"`, `"ru"`, and more)
- `response_format`: Audio format (`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`, `m4a`)
- `speed`: Speed of generated audio (0.25 to 4.0, default 1.0)

## 🐳 Docker Deployment

Docker is the **recommended and preferred** deployment method. It provides:
- ✅ Consistent environment across all systems
- ✅ No dependency conflicts
- ✅ Easy GPU configuration
- ✅ Production-ready setup
- ✅ Simplified updates and maintenance

### Requirements

- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with 8GB+ VRAM (for 1.5B model) or 16GB+ (for Large model)

### Quick Start

```bash
# Copy and configure environment
cp docker-env.example .env
# Edit .env - set VOICES_DIR to your voice files path

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for complete Docker deployment guide.

## 🔧 Development

### Setup Development Environment

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install -r requirements-api.txt

# Install PyTorch (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Optional: Install flash-attn for faster inference
# See setup.sh for pre-built wheel installation
```

### Running Tests

```bash
# Start server
./start.sh

# Test API
curl http://localhost:8001/health
curl http://localhost:8001/v1/audio/voices
```

## 📊 Supported Models

| Model | Size | Context | Max Length | VRAM Required | Real-time Factor (Ampere) |
|-------|------|---------|------------|---------------|--------------------------|
| | VibeVoice-1.5B | 1.5B | 64K | ~90 min | 8GB+ | 1.5-2.5x |
| | VibeVoice-Large | 7B | 32K | ~45 min | 16GB+ (10.8GB optimized) | 0.77x |

*Real-time factor on RTX 3090 with all Ampere optimizations enabled.*

Models are automatically downloaded from HuggingFace on first use.

## 🛠️ System Requirements

**For Docker Deployment (Recommended):**
- **Docker**: Docker and Docker Compose installed
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for 1.5B model) or 16GB+ (for Large model)
- **NVIDIA Container Toolkit**: Required for GPU support
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB minimum, 50GB recommended (with model cache)
- **OS**: Linux (recommended), macOS, or Windows (with WSL2)

**For Local Installation:**
- **Python**: 3.12
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB minimum, 50GB recommended
- **OS**: Linux, macOS, or Windows (with WSL2)

## 🔐 Security Notes

- The API does not include authentication by default. For production use, add authentication middleware or deploy behind a reverse proxy with authentication.
- Voice files are loaded from the configured directory - ensure proper file permissions.
- Model weights are downloaded from HuggingFace - verify model integrity in production.

## 📝 License

This project maintains the original VibeVoice model codebase. Please refer to the original VibeVoice license for model usage terms.

## 🔧 Dependency Fixes

The project includes compatibility layers and fixes for common dependency issues:

### Transformers Version Compatibility
- Added compatibility layer for `FlashAttentionKwargs` import in both `modeling_vibevoice.py` and `modeling_vibevoice_inference.py`
- Supports both transformers 4.51.3+ and older versions
- Automatic fallback to stub implementation if `FlashAttentionKwargs` is not available

### Language Detection Dependencies
- Added `langdetect>=1.0.9` to `requirements-api.txt`
- Created `api/utils/language_utils.py` with robust language detection for 50+ languages
- Fallback to English if detection fails

### Installation Command
For a clean installation with all dependencies:
```bash
# Install core VibeVoice dependencies
pip install -e .

# Install API dependencies with language detection
pip install -r requirements-api.txt

# Ensure correct transformers version
pip install transformers==4.51.3
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- **VibeVoice Team at Microsoft** for the original model and groundbreaking research in text-to-speech
- **[shijincai](https://github.com/shijincai)** for maintaining a backup of the original VibeVoice codebase
- **[Dao-AILab](https://github.com/Dao-AILab)** for Flash Attention 2, enabling efficient attention computation
- **[PyTorch Team](https://github.com/pytorch)** for torch.compile and torchao optimization tools
- **[FastAPI](https://fastapi.tiangolo.com)** for the excellent web framework
- **[HuggingFace](https://huggingface.co)** for model hosting and transformers library

**Special thanks to the open-source community for making state-of-the-art TTS accessible to everyone.**

## 📚 Additional Resources

- [VibeVoice Original Paper](https://arxiv.org/pdf/2508.19205)
- [VibeVoice HuggingFace Collection](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

## ⚠️ Limitations

- **Language Support**: Primarily English and Chinese. Other languages may produce unexpected results. Auto-detection is available for many languages via `language="auto"`.
- **Non-Speech Audio**: The model focuses on speech synthesis and may generate background music or sounds spontaneously.
- **Commercial Use**: This model is intended for research and development. Test thoroughly before production use.

## 🆘 Troubleshooting

### Common Deployment Issues

**GPU "Out of Memory" (OOM):**
- The system now includes **Smart GPU Autodetection**. If you have multiple GPUs, the API will automatically select the one with the most free memory.
- To force a specific GPU, set `VIBEVOICE_DEVICE=cuda:0` (or `cuda:1`, etc.) in your `.env` file instead of just `cuda`.
- Use a smaller model: `VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B`
- Use quantization: `VIBEVOICE_QUANTIZATION=int8_torchao`

**Volume Mounting Errors:**
- Ensure your `.env` file has the correct relative path for `VOICES_DIR`.
- Correct: `VOICES_DIR=./demo/voices`
- Incorrect: `VOICES_DIR=demo/voices` (Docker treats this as a named volume)

### Dependency Issues

**Transformers Import Error:**
If you see `ImportError: cannot import name 'FlashAttentionKwargs'`, install the correct transformers version:
```bash
pip install transformers==4.51.3
```

**Package Conflicts:**
If you have package conflicts with chatterbox-tts or other packages:
```bash
# Uninstall conflicting packages
pip uninstall chatterbox-tts

# Reinstall with correct versions
pip install transformers==4.51.3 langdetect>=1.0.9
```

### Server won't start
- Check GPU availability: `nvidia-smi`
- Verify Python version: `python3 --version` (should be 3.12)
- Check dependencies: `pip list | grep transformers transformers` (should be 4.51.3)
- Check torch version: `pip list | grep torch`

### Out of memory errors
- Use smaller model: `VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B`
- Reduce inference steps: `VIBEVOICE_INFERENCE_STEPS=5`
- Use CPU mode: `VIBEVOICE_DEVICE=cpu` (much slower)

### Voice not found errors
- Verify `VOICES_DIR` path in `.env`
- Check file permissions
- Ensure audio files are in supported formats

### Language detection issues
- Very short text (<10 characters) may have lower accuracy
- For best results with short text, use explicit language parameter
- Use language codes like `es`, `en`, `fr`, `de`, `it`, `pt`, `zh`, `ja`, `ko`, `ar`, `hi`, `ru`

For more help, see the [API README](API_README.md) or open an issue.

---

**Made with ❤️ for the VibeVoice community**
