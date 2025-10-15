# VibeVoice Standalone TTS

Complete standalone implementation of Microsoft VibeVoice text-to-speech with Gradio UI and FastAPI, independent of ComfyUI.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Option A: Using existing Python (if dependencies installed)
python3 test_setup.py  # Verify setup

# Option B: Create conda environment
conda env create -f environment.yml
conda activate vibevoice-standalone

# Option C: Using pip/venv
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Models Setup

Models are symlinked from ComfyUI:
```bash
ls -la models/
# Should show:
# - VibeVoice-1.5B -> (linked to ComfyUI models)
# - VibeVoice-Large -> (linked to ComfyUI models)
# - tokenizer -> (linked to Qwen tokenizer)
```

If you need to download models manually:
```bash
cd models
git lfs clone https://huggingface.co/microsoft/VibeVoice-1.5B
git lfs clone https://huggingface.co/aoi-ot/VibeVoice-Large

# Download tokenizer
mkdir -p tokenizer && cd tokenizer
wget https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/{tokenizer_config.json,vocab.json,merges.txt,tokenizer.json}
```

### 3. Launch

```bash
# Gradio Web Interface (default GPU 0)
./run_gradio.sh

# Or on specific GPU
CUDA_VISIBLE_DEVICES=1 ./run_gradio.sh --port 8080

# FastAPI Server
./run_fastapi.sh
```

Access:
- **Gradio UI**: http://localhost:7860 (or custom port)
- **FastAPI Docs**: http://localhost:8000/docs

## ✨ Features

### Text-to-Speech Capabilities
- **Single Speaker**: Natural speech with voice cloning support
- **Multi-Speaker**: Up to 4 distinct speakers in conversations
- **Voice Cloning**: Clone voices from 20-30 second audio samples
- **Pause Control**: `[pause]` (1s) or `[pause:1500]` (custom ms)
- **Speed Control**: 0.8x - 1.2x voice speed adjustment

### Model Options
- **VibeVoice-1.5B**: Fast, ~6GB VRAM
- **VibeVoice-Large**: Best quality, ~20GB VRAM
- **Attention**: auto, eager, sdpa, flash_attention_2
- **Quantization**: 4-bit or 8-bit for memory savings

### Generation Control
- **Diffusion Steps**: 1-100 (20 recommended, 30-40 for quality)
- **CFG Scale**: 0.5-3.5 guidance (1.3 default)
- **Sampling Mode**: Deterministic or temperature/top_p sampling
- **Seed Control**: Reproducible results

## 📖 Usage Examples

### Gradio Interface

1. Open http://localhost:7860
2. Select model (VibeVoice-Large for best quality)
3. Enter text or upload voice sample
4. Click "Generate Speech"

### Multi-Speaker Format

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
[1]: That's wonderful to hear.
[3]: Can I join this conversation?
```

### FastAPI Examples

**Simple generation (curl):**
```bash
curl -X POST "http://localhost:8000/tts/single" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is VibeVoice!",
    "model_name": "VibeVoice-Large",
    "diffusion_steps": 20,
    "seed": 42
  }' --output output.wav
```

**Python client:**
```python
import requests

# Single speaker
response = requests.post(
    "http://localhost:8000/tts/single",
    json={
        "text": "Welcome to VibeVoice TTS!",
        "model_name": "VibeVoice-Large",
        "diffusion_steps": 20,
        "cfg_scale": 1.3,
        "seed": 42
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)

# Multi-speaker
response = requests.post(
    "http://localhost:8000/tts/multi",
    json={
        "text": "[1]: Hello! [2]: Hi there!",
        "model_name": "VibeVoice-Large",
        "diffusion_steps": 20
    }
)

with open("conversation.wav", "wb") as f:
    f.write(response.content)
```

**With voice cloning:**
```python
with open("reference_voice.wav", "rb") as voice:
    files = {"voice_file": voice}
    data = {
        "text": "This uses my cloned voice.",
        "model_name": "VibeVoice-Large",
        "diffusion_steps": 20
    }
    
    response = requests.post(
        "http://localhost:8000/tts/single",
        data=data,
        files=files
    )
```

## 🎯 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /tts/single` - Single speaker TTS
- `POST /tts/multi` - Multi-speaker TTS
- `POST /memory/free` - Free model memory
- `GET /docs` - Interactive API documentation

## ⚙️ Configuration

### GPU Selection
```bash
# Use GPU 0 (default)
./run_gradio.sh

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 ./run_gradio.sh

# Use GPU 2
CUDA_VISIBLE_DEVICES=2 ./run_gradio.sh
```

### Custom Ports
```bash
./run_gradio.sh --port 8080
./run_fastapi.sh --port 9000
```

### Models Directory
```bash
export VIBEVOICE_MODELS_DIR=/path/to/models
./run_gradio.sh
```

## 💾 System Requirements

| Model | VRAM | RAM | Storage | Speed |
|-------|------|-----|---------|-------|
| VibeVoice-1.5B | 6GB | 16GB | 5.4GB | Fast |
| VibeVoice-Large-Q8 | 12GB | 24GB | 11.6GB | Good |
| VibeVoice-Large | 20GB | 32GB | 18.7GB | Best |

**GPU**: CUDA 11.8+ (NVIDIA), MPS (Apple Silicon), or CPU (slow)

## 🔧 Advanced Parameters

### Quality vs Speed
```python
# Fast (3-5s per sentence)
diffusion_steps=10

# Balanced (5-8s)
diffusion_steps=20

# High Quality (8-12s)
diffusion_steps=40
```

### Sampling Mode
```python
# Deterministic (recommended)
use_sampling=False

# Creative (more varied)
use_sampling=True
temperature=0.95
top_p=0.95
```

### Memory Management
```python
# Auto-free after generation
free_memory=True

# Keep loaded (faster subsequent generations)
free_memory=False
```

## 🐛 Troubleshooting

### No models found
```bash
# Check models directory
ls -la models/

# Should see: VibeVoice-1.5B, VibeVoice-Large, tokenizer
# If missing, download or create symlinks
```

### Tokenizer error
```bash
# Verify tokenizer files
ls models/tokenizer/
# Required: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json
```

### CUDA out of memory
```python
# Use smaller model
model_name="VibeVoice-1.5B"

# Or enable quantization
quantize_llm="4bit"

# Or reduce steps
diffusion_steps=10
```

### Port already in use
```bash
# Find free port
python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])"

# Use that port
./run_gradio.sh --port 12345
```

### Import errors
```bash
# Verify all packages installed
python3 test_setup.py

# Reinstall if needed
pip install -r requirements.txt
```

## 📁 Project Structure

```
standalone/
├── vibevoice_core.py       # Core TTS engine
├── gradio_app.py            # Gradio web interface
├── fastapi_server.py        # REST API server
│
├── run_gradio.sh            # Launch Gradio
├── run_fastapi.sh           # Launch FastAPI
│
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
│
├── models/                  # Models directory
│   ├── VibeVoice-1.5B/     # (symlinked or downloaded)
│   ├── VibeVoice-Large/    # (symlinked or downloaded)
│   └── tokenizer/          # Qwen tokenizer (required)
│
└── README.md               # This file
```

## 🎓 Tips & Best Practices

1. **Seed Management**: Save seeds that produce good voices
2. **First Generation**: Takes longer (model loading)
3. **Batch Processing**: Keep `free_memory=False` for multiple generations
4. **Voice Cloning**: Use 20-30 seconds of clean audio
5. **Multi-Speaker**: Use VibeVoice-Large for better quality
6. **Long Text**: Automatically chunks at 250 words
7. **Pause Placement**: Use natural speech rhythm

## 🆚 Differences from ComfyUI Nodes

| Feature | ComfyUI Nodes | Standalone |
|---------|---------------|------------|
| **Dependency** | Requires ComfyUI | Independent ✓ |
| **Interface** | Node workflow | Gradio UI + API ✓ |
| **Setup** | ComfyUI install | Pip/Conda ✓ |
| **API** | None | REST API ✓ |
| **Features** | Full | All preserved ✓ |

All functionality from the ComfyUI nodes is preserved and working.

## 📝 License

MIT License - Inherits from VibeVoice-ComfyUI wrapper

**Note**: VibeVoice model is subject to Microsoft's research license.

## 🙏 Credits

This standalone implementation builds upon excellent work from the community:

- **VibeVoice Model**: [Microsoft Research](https://github.com/microsoft/VibeVoice) - Original TTS model and research
- **VibeVoice-ComfyUI**: [Fabio Sarracino](https://github.com/Enemyx-net/VibeVoice-ComfyUI) - ComfyUI integration (source of this adaptation)
- **Gradio**: Web interface framework
- **FastAPI**: REST API framework
- **Hugging Face Transformers**: Model infrastructure
- **PyTorch**: Deep learning framework

See [CREDITS.md](CREDITS.md) for detailed acknowledgments.

## 🔗 Links

- [VibeVoice (Microsoft)](https://github.com/microsoft/VibeVoice)
- [VibeVoice-ComfyUI](https://github.com/Enemyx-net/VibeVoice-ComfyUI)
- [Gradio](https://gradio.app)
- [FastAPI](https://fastapi.tiangolo.com)

---

**Quick Commands:**
```bash
# Launch Gradio on GPU 1
CUDA_VISIBLE_DEVICES=1 ./run_gradio.sh

# Test setup
python3 test_setup.py

# API health check
curl http://localhost:8000/health
```

For issues, check logs and verify models/tokenizer are properly installed.
