# VibeVoice API - Quick Reference

## 🚀 Getting Started

```bash
# Setup (one time)
./setup.sh

# Start server
./start.sh

# Server runs at: http://localhost:8001
```

---

## 🎤 Adding Custom Voices

### Quick Method
```bash
# 1. Copy your voice file
cp my-voice.wav demo/voices/my-custom-voice.wav

# 2. Restart server
./start.sh

# 3. Use it!
curl -X POST http://localhost:8001/v1/audio/speech \
  -d '{"voice":"my-custom-voice","input":"Hello!","response_format":"mp3"}' \
  --output speech.mp3
```

### Custom Directory
```bash
# In .env file
VOICES_DIR=/path/to/my/voices

# The API automatically loads ALL audio files from this directory
# Supported: .wav, .mp3, .flac, .ogg, .m4a, .aac
```

📖 **Full Guide**: [CUSTOM_VOICES_GUIDE.md](CUSTOM_VOICES_GUIDE.md)

---

## 🔊 Using Voices

### List All Voices
```bash
# OpenAI voices only (6 voices)
curl http://localhost:8001/v1/audio/voices

# All voices including custom (9+ voices)
curl http://localhost:8001/v1/audio/voices?show_all=true
```

### Generate Speech
```bash
# OpenAI-compatible endpoint
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Your text here",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Available Voices

| Voice Name | Language | Access Via |
|------------|----------|------------|
| en-Alice_woman | English | `alloy` or `en-Alice_woman` |
| en-Carter_man | English | `echo` or `en-Carter_man` |
| en-Maya_woman | English | `fable` or `en-Maya_woman` |
| en-Frank_man | English | `onyx` or `en-Frank_man` |
| en-Mary_woman_bgm | English | `nova` or `en-Mary_woman_bgm` |
| in-Samuel_man | Indian English | `in-Samuel_man` |
| zh-Anchen_man_bgm | Chinese | `zh-Anchen_man_bgm` |
| zh-Bowen_man | Chinese | `zh-Bowen_man` |
| zh-Xinran_woman | Chinese | `shimmer` or `zh-Xinran_woman` |
| **+ your custom voices** | Any | **filename without extension** |

📖 **Full Guide**: [VOICE_USAGE_GUIDE.md](VOICE_USAGE_GUIDE.md)

---

## 🎭 Multi-Speaker Conversations

```bash
curl -X POST http://localhost:8001/v1/vibevoice/generate \
  -H "Content-Type: application/json" \
  -d '{
    "script": "Speaker 0: Hello!\nSpeaker 1: Hi there!",
    "speakers": [
      {"speaker_id": 0, "voice_preset": "en-Alice_woman"},
      {"speaker_id": 1, "voice_preset": "en-Carter_man"}
    ],
    "response_format": "wav"
  }' \
  --output conversation.wav
```

---

## ⚙️ Configuration (.env)

```bash
# Model
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B
VIBEVOICE_DEVICE=cuda
VIBEVOICE_INFERENCE_STEPS=10

# Voices (KEY SETTING!)
VOICES_DIR=demo/voices

# Server
API_HOST=0.0.0.0
API_PORT=8001

# Defaults
DEFAULT_CFG_SCALE=1.3
DEFAULT_RESPONSE_FORMAT=mp3
```

📖 **Model Guide**: [MODELS_GUIDE.md](MODELS_GUIDE.md)

---

## 🐍 Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"
)

# Generate speech
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",  # or any custom voice name
    input="Hello from VibeVoice!"
)
response.stream_to_file("speech.mp3")
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Generate speech (OpenAI-compatible) |
| `/v1/audio/voices` | GET | List voices (`?show_all=true` for all) |
| `/v1/vibevoice/generate` | POST | Multi-speaker generation |
| `/v1/vibevoice/voices` | GET | List VibeVoice presets |
| `/v1/vibevoice/health` | GET | Model status |
| `/health` | GET | Basic health check |
| `/docs` | GET | Interactive API documentation |

---

## 🎯 Common Tasks

### Add a Voice
```bash
# Just copy the file and restart
cp voice.wav demo/voices/my-voice.wav
./start.sh
```

### Change Model
```bash
# Edit .env
VIBEVOICE_MODEL_PATH=/path/to/model
# or
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-Large

# Restart
./start.sh
```

### Change Port
```bash
# Edit .env
API_PORT=8080

# Restart
./start.sh
```

### Use Custom Voice Directory
```bash
# Edit .env
VOICES_DIR=/my/custom/voices

# Restart
./start.sh
```

---

## 🔍 Troubleshooting

### Voice not found
```bash
# List all voices to see what's available
curl http://localhost:8001/v1/audio/voices?show_all=true

# Check voices directory
ls -la demo/voices/

# Restart server to reload voices
./start.sh
```

### Model not loading
```bash
# Check model path
cat .env | grep VIBEVOICE_MODEL_PATH

# Check if model exists
ls -la ~/.cache/huggingface/hub/ | grep VibeVoice

# Check server logs
# Look for "Model loaded successfully!" message
```

### Out of memory
```bash
# Use smaller model in .env
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B

# Or use CPU
VIBEVOICE_DEVICE=cpu
```

---

## 📚 Documentation

- **[API_README.md](API_README.md)** - Complete API documentation
- **[CUSTOM_VOICES_GUIDE.md](CUSTOM_VOICES_GUIDE.md)** - How to add unlimited custom voices
- **[VOICE_USAGE_GUIDE.md](VOICE_USAGE_GUIDE.md)** - Using voices in API requests
- **[MODELS_GUIDE.md](MODELS_GUIDE.md)** - Available models and switching
- **[QUICKSTART.md](QUICKSTART.md)** - Quick setup guide
- **[example_add_voice.sh](example_add_voice.sh)** - Script to add voices

---

## 💡 Key Features

✅ **Automatic Voice Discovery** - Drop files in folder, restart, done!  
✅ **Unlimited Voices** - Add as many as you want  
✅ **Any Format** - WAV, MP3, FLAC, OGG, M4A, AAC  
✅ **OpenAI Compatible** - Works with existing OpenAI clients  
✅ **Multi-Speaker** - Mix different voices in conversations  
✅ **No Code Changes** - Everything configured via `.env`

---

## 🎉 Quick Example

```bash
# 1. Add your voice
cp my-recording.wav demo/voices/my-voice.wav

# 2. Restart
./start.sh

# 3. Generate speech
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "This is my custom voice!",
    "voice": "my-voice",
    "response_format": "mp3"
  }' \
  --output output.mp3

# 4. Listen!
# (play output.mp3 with your media player)
```

**That's it!** 🚀


