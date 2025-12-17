# VibeVoice OpenAI-Compatible TTS API

A production-ready REST API for VibeVoice text-to-speech that provides both OpenAI-compatible endpoints and extended VibeVoice-specific features.

## Features

- ✅ **OpenAI-Compatible API** - Drop-in replacement for OpenAI's TTS API
- ✅ **Unlimited Custom Voices** - Automatically loads all audio files from voices folder
- ✅ **Multi-Speaker Support** - Generate dialogues with up to 4 distinct speakers
- ✅ **Streaming Audio** - Real-time audio generation with chunked streaming
- ✅ **Multiple Audio Formats** - Support for MP3, WAV, FLAC, Opus, AAC, and PCM
- ✅ **9 Built-in Voices** - English, Chinese, and Indian English presets included
- ✅ **CFG Scale Control** - Fine-tune generation quality and adherence
- ✅ **Auto Device Detection** - Automatically uses CUDA, MPS, or CPU

## Quick Start

### 1. Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/shijincai/VibeVoice.git
cd VibeVoice

# Run the setup script (installs dependencies, PyTorch, flash-attn)
./setup.sh
```

The setup script will:
- Detect and validate Python 3.10 or 3.11
- Create a virtual environment
- Install PyTorch with appropriate CUDA support
- Install flash-attn (optional, for better performance)
- Install VibeVoice and API dependencies
- Create a `.env` configuration file

### 2. Configuration

Edit the `.env` file to configure the API:

```bash
# Model configuration
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B  # or microsoft/VibeVoice-Large
VIBEVOICE_DEVICE=cuda  # cuda, cpu, or mps
VIBEVOICE_INFERENCE_STEPS=10  # 5-50, higher = better quality

# Voice configuration
VOICES_DIR=demo/voices  # Directory containing voice preset audio files

# Server configuration
API_HOST=0.0.0.0
API_PORT=8000
```

**Adding Custom Voices:**
Simply drop audio files (WAV, MP3, FLAC, etc.) into the `VOICES_DIR` folder and restart the server. The API will automatically discover and load them as voice presets. See [CUSTOM_VOICES_GUIDE.md](CUSTOM_VOICES_GUIDE.md) for details.

### 3. Start the Server

```bash
./start.sh
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### OpenAI-Compatible Endpoints

#### POST `/v1/audio/speech`

Generate speech from text using OpenAI-compatible API.

**Request:**
```json
{
  "model": "tts-1",
  "input": "Hello, this is a test of the VibeVoice API.",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

**Supported Voices:**
- `alloy` → en-Alice_woman (English, Female)
- `echo` → en-Carter_man (English, Male)
- `fable` → en-Maya_woman (English, Female)
- `onyx` → en-Frank_man (English, Male)
- `nova` → en-Mary_woman_bgm (English, Female with BGM)
- `shimmer` → zh-Xinran_woman (Chinese, Female)

**Plus 3 additional voices** (use preset names directly):
- `in-Samuel_man` (Indian English, Male)
- `zh-Anchen_man_bgm` (Chinese, Male with BGM)
- `zh-Bowen_man` (Chinese, Male)

📖 See [VOICE_USAGE_GUIDE.md](VOICE_USAGE_GUIDE.md) for complete voice documentation.

**Example (curl):**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello from VibeVoice!",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

**Example (Python):**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Hello from VibeVoice!",
        "voice": "alloy",
        "response_format": "mp3"
    }
)

with open("speech.mp3", "wb") as f:
    f.write(response.content)
```

#### GET `/v1/audio/voices`

List available OpenAI-compatible voices.

### VibeVoice Extended Endpoints

#### POST `/v1/vibevoice/generate`

Generate multi-speaker dialogue with advanced features.

**Request:**
```json
{
  "script": "Speaker 0: Welcome to our podcast!\nSpeaker 1: Thanks for having me!",
  "speakers": [
    {
      "speaker_id": 0,
      "voice_preset": "en-Alice_woman"
    },
    {
      "speaker_id": 1,
      "voice_preset": "en-Carter_man"
    }
  ],
  "cfg_scale": 1.3,
  "inference_steps": 10,
  "response_format": "mp3",
  "stream": false
}
```

**Example (Python with streaming):**
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/vibevoice/generate",
    json={
        "script": "Speaker 0: Hello!\nSpeaker 1: Hi there!",
        "speakers": [
            {"speaker_id": 0, "voice_preset": "en-Alice_woman"},
            {"speaker_id": 1, "voice_preset": "en-Carter_man"}
        ],
        "stream": True
    },
    stream=True
)

# Process Server-Sent Events
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if 'done' in data:
                print("Generation complete!")
                break
            elif 'audio' in data:
                # Process audio chunk (base64 encoded)
                print(f"Received chunk {data['chunk_id']}")
```

#### GET `/v1/vibevoice/voices`

List all available VibeVoice voice presets.

#### GET `/v1/vibevoice/health`

Check service health and model status.

## Audio Formats

Supported output formats:
- **mp3** - MPEG Audio Layer III (default)
- **opus** - Opus codec (efficient for streaming)
- **aac** - Advanced Audio Coding
- **flac** - Free Lossless Audio Codec
- **wav** - Waveform Audio File Format
- **pcm** - Raw PCM audio data

## Voice Presets

Place voice sample files (WAV, MP3, FLAC, etc.) in the `demo/voices/` directory. The API will automatically detect and load them.

Voice file naming convention:
- `en-Name_gender.wav` - English voices
- `zh-Name_gender.wav` - Chinese voices
- `in-Name_gender.wav` - Indian English voices

## Performance Tips

1. **Use CUDA with flash-attn** for best performance:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Adjust inference steps** based on quality/speed tradeoff:
   - 5 steps: Fast, lower quality
   - 10 steps: Balanced (default)
   - 20+ steps: High quality, slower

3. **Use streaming** for real-time applications:
   ```json
   {"stream": true}
   ```

4. **Keep workers=1** when using GPU (model loading limitation)

## Troubleshooting

### Model Download Issues

If the model fails to download automatically:
```bash
# Download manually using huggingface-cli
pip install huggingface-hub
huggingface-cli download microsoft/VibeVoice-1.5B
```

### CUDA Out of Memory

Try the 1.5B model instead of 7B:
```bash
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B
```

Or reduce inference steps:
```bash
VIBEVOICE_INFERENCE_STEPS=5
```

### Flash Attention Installation Failed

The API will automatically fall back to SDPA attention. For better performance, try:
```bash
source venv/bin/activate
pip install flash-attn --no-build-isolation
```

### Audio Format Conversion Issues

Ensure ffmpeg is installed:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Development

### Running Tests

```bash
source venv/bin/activate
pytest tests/
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

This project follows the VibeVoice license. See the main repository for details.

## Contributing

Contributions are welcome! Please see the main VibeVoice repository for contribution guidelines.

## Support

For issues and questions:
- GitHub Issues: https://github.com/shijincai/VibeVoice/issues
- Documentation: https://microsoft.github.io/VibeVoice

