# Open WebUI Integration Guide

## VibeVoice as OpenAI-Compatible TTS Provider for Open WebUI

This guide explains how to configure Open WebUI to use VibeVoice as a TTS provider.

## Quick Setup

### 1. Start VibeVoice Server

```bash
cd /home/op/ComfyUI/custom_nodes/VibeVoice-ComfyUI/standalone
./autolauncher_gpu1.sh
```

Server will start on: `http://0.0.0.0:8000`

### 2. Configure Open WebUI

Open WebUI should be configured to use the OpenAI-compatible endpoint:

**Base URL:** `http://localhost:8000/v1`

**Or if Open WebUI is on a different machine:**
**Base URL:** `http://YOUR_SERVER_IP:8000/v1`

### 3. Settings in Open WebUI Admin Panel

1. Go to **Admin Panel** → **Settings** → **Audio**
2. Set **TTS Engine** to **OpenAI**
3. Set **API Base URL** to `http://localhost:8000/v1` (or `http://0.0.0.0:8000/v1`)
4. **API Key**: Can be anything (not required, but field might be mandatory in UI - use "sk-dummy")
5. **TTS Model**: Select `tts-1` (fast) or `tts-1-hd` (high quality)
6. **TTS Voice**: Choose from: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### 4. Environment Variables (Alternative Configuration)

If Open WebUI uses environment variables, you can set:

```bash
export OPENAI_API_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="not-needed"
export TTS_MODEL="tts-1-hd"
export TTS_VOICE="alloy"
```

## Available Models

- **tts-1** → VibeVoice-1.5B (Fast, ~6GB VRAM, ~10 diffusion steps)
- **tts-1-hd** → VibeVoice-Large (High Quality, ~20GB VRAM, ~20 diffusion steps)
- **vibevoice-1.5b** → Direct access to 1.5B model
- **vibevoice-large** → Direct access to Large model

## Available Voices

Each voice is mapped to a unique seed for consistent voice generation:

- **alloy** (seed: 42) - Default balanced voice
- **echo** (seed: 123) - Alternative voice style
- **fable** (seed: 456) - Narrative style
- **onyx** (seed: 789) - Deep voice
- **nova** (seed: 999) - Bright voice
- **shimmer** (seed: 1337) - Smooth voice

## Testing the Connection

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "device": "cuda", "models_loaded": false}
```

### Test 2: Models List
```bash
curl http://localhost:8000/v1/models
```

Expected: List of available models in OpenAI format

### Test 3: Generate Speech
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1-hd",
    "input": "Hello from VibeVoice!",
    "voice": "nova"
  }' \
  --output test_speech.wav
```

### Test 4: Using OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",
    input="Testing Open WebUI integration with VibeVoice!"
)

response.stream_to_file("output.wav")
```

## Troubleshooting

### Error: Connection Refused
- Ensure VibeVoice server is running: `ps aux | grep fastapi_server.py`
- Check if port 8000 is accessible: `curl http://localhost:8000/health`
- Restart server: `./autolauncher_gpu1.sh`

### Error: 500 Internal Server Error
- Check server logs in the terminal where fastapi_server.py is running
- Ensure GPU 1 is available: `nvidia-smi`
- Verify models are accessible: `ls -la models/`

### Error: Model Not Found
- Verify model files exist in `standalone/models/`
- Check symlinks are valid: `ls -la models/`
- Models required:
  - `models/VibeVoice-1.5B/`
  - `models/VibeVoice-Large/`
  - `models/tokenizer/`

### Error: Empty or Invalid Audio
- Check the input text is not empty
- Try with a simple test: `{"model":"tts-1","input":"test","voice":"alloy"}`
- Verify audio format compatibility (WAV is most compatible)

### Error: CORS Issues
- Server has CORS enabled for all origins (`*`)
- Check browser console for specific CORS errors
- Ensure using correct URL scheme (http:// not https://)

## Performance Tuning

### Fast Generation (tts-1)
- Uses VibeVoice-1.5B
- 10 diffusion steps
- ~3-5 seconds per sentence
- ~6GB VRAM

### High Quality (tts-1-hd)
- Uses VibeVoice-Large
- 10 diffusion steps (same speed, better quality model)
- ~5-8 seconds per sentence
- ~20GB VRAM

### Adjust Speed (if needed)
Default is 10 steps for all models (good balance).
To customize, modify `fastapi_server.py`:
```python
diffusion_steps=5   # Faster but lower quality
diffusion_steps=10  # Default - good balance
diffusion_steps=20  # Higher quality
diffusion_steps=40  # Maximum quality but slower
```

## Server Info

- **API Documentation:** http://localhost:8000/docs
- **Health Endpoint:** http://localhost:8000/health
- **OpenAI Endpoint:** http://localhost:8000/v1/audio/speech
- **Models Endpoint:** http://localhost:8000/v1/models

## Network Configuration

### Local Access Only
Server binds to `0.0.0.0:8000` - accessible from all network interfaces

### Remote Access
If Open WebUI is on a different machine:
1. Use server's IP instead of localhost
2. Ensure firewall allows port 8000
3. Update Open WebUI config: `http://SERVER_IP:8000/v1`

### Docker/Container Setup
If running in Docker, ensure port mapping:
```bash
docker run -p 8000:8000 ...
```

## Features

✅ **OpenAI API Compatible** - Drop-in replacement  
✅ **No API Key Required** - Free and open-source  
✅ **Multiple Voice Presets** - 6 different voices  
✅ **GPU Accelerated** - Fast generation on GPU 1  
✅ **Automatic Memory Management** - Frees GPU memory after generation  
✅ **Detailed Logging** - Easy debugging  
✅ **CORS Enabled** - Works with web interfaces  
✅ **Format Support** - WAV, FLAC, PCM (MP3 fallback to WAV)

## Support

For issues or questions:
1. Check server logs in terminal
2. Verify all dependencies installed
3. Test endpoint directly with curl
4. Check Open WebUI logs for specific errors

## Example Open WebUI Configuration

If using configuration file or environment:

```yaml
# Open WebUI Config
audio:
  tts:
    engine: openai
    openai:
      api_base_url: http://localhost:8000/v1
      api_key: not-needed
      model: tts-1-hd
      voice: alloy
```

Or environment variables:
```bash
export AUDIO_TTS_ENGINE=openai
export AUDIO_TTS_OPENAI_API_BASE_URL=http://localhost:8000/v1
export AUDIO_TTS_OPENAI_API_KEY=not-needed
export AUDIO_TTS_MODEL=tts-1-hd
export AUDIO_TTS_VOICE=alloy
```

---

**Server Status:** Running on GPU 1  
**Endpoint:** http://0.0.0.0:8000/v1/audio/speech  
**Compatible with:** Open WebUI, any OpenAI SDK client
