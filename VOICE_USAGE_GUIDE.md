# Voice Usage Guide

## Overview

The VibeVoice API supports **9 voice presets** across multiple languages. You can use them in two ways:

1. **OpenAI-compatible names** (6 voices mapped)
2. **Direct VibeVoice preset names** (all 9 voices)

---

## Listing Available Voices

### OpenAI-Compatible Voices (Default)

```bash
curl http://localhost:8001/v1/audio/voices
```

**Response:**
```json
{
  "voices": [
    {
      "name": "alloy",
      "vibevoice_preset": "en-Alice_woman",
      "available": true
    },
    {
      "name": "echo",
      "vibevoice_preset": "en-Carter_man",
      "available": true
    },
    ...
  ],
  "total": 6,
  "note": "Add ?show_all=true to see all available VibeVoice presets"
}
```

### All VibeVoice Presets

```bash
curl http://localhost:8001/v1/audio/voices?show_all=true
```

**Response:**
```json
{
  "voices": [
    {
      "name": "en-Alice_woman",
      "path": "demo/voices/en-Alice_woman.wav",
      "language": "English",
      "openai_alias": "alloy"
    },
    {
      "name": "en-Carter_man",
      "path": "demo/voices/en-Carter_man.wav",
      "language": "English",
      "openai_alias": "echo"
    },
    {
      "name": "en-Frank_man",
      "path": "demo/voices/en-Frank_man.wav",
      "language": "English",
      "openai_alias": "onyx"
    },
    {
      "name": "en-Mary_woman_bgm",
      "path": "demo/voices/en-Mary_woman_bgm.wav",
      "language": "English",
      "openai_alias": "nova"
    },
    {
      "name": "en-Maya_woman",
      "path": "demo/voices/en-Maya_woman.wav",
      "language": "English",
      "openai_alias": "fable"
    },
    {
      "name": "in-Samuel_man",
      "path": "demo/voices/in-Samuel_man.wav",
      "language": "Indian English"
    },
    {
      "name": "zh-Anchen_man_bgm",
      "path": "demo/voices/zh-Anchen_man_bgm.wav",
      "language": "Chinese"
    },
    {
      "name": "zh-Bowen_man",
      "path": "demo/voices/zh-Bowen_man.wav",
      "language": "Chinese"
    },
    {
      "name": "zh-Xinran_woman",
      "path": "demo/voices/zh-Xinran_woman.wav",
      "language": "Chinese",
      "openai_alias": "shimmer"
    }
  ],
  "total": 9,
  "note": "Use 'name' field directly or 'openai_alias' (if available) in /v1/audio/speech requests"
}
```

---

## Using Voices in Speech Generation

### Method 1: OpenAI Voice Names

Use the standard OpenAI voice names (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`):

```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Method 2: Direct VibeVoice Preset Names

Use any of the 9 VibeVoice preset names directly:

```bash
# Indian English voice (not mapped to OpenAI)
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test.",
    "voice": "in-Samuel_man",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

```bash
# Chinese voice with background music
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "你好，这是一个测试。",
    "voice": "zh-Anchen_man_bgm",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

---

## Voice Mapping Reference

| OpenAI Name | VibeVoice Preset | Language | Gender | Notes |
|-------------|------------------|----------|--------|-------|
| `alloy` | `en-Alice_woman` | English | Female | Clear, neutral |
| `echo` | `en-Carter_man` | English | Male | Deep, resonant |
| `fable` | `en-Maya_woman` | English | Female | Warm, expressive |
| `onyx` | `en-Frank_man` | English | Male | Strong, authoritative |
| `nova` | `en-Mary_woman_bgm` | English | Female | With background music |
| `shimmer` | `zh-Xinran_woman` | Chinese | Female | Mandarin Chinese |

### Additional Voices (No OpenAI Mapping)

| VibeVoice Preset | Language | Gender | Notes |
|------------------|----------|--------|-------|
| `in-Samuel_man` | Indian English | Male | Indian accent |
| `zh-Anchen_man_bgm` | Chinese | Male | With background music |
| `zh-Bowen_man` | Chinese | Male | Clear Mandarin |

---

## Multi-Speaker Support

For multi-speaker conversations, use the VibeVoice extended endpoint:

```bash
curl -X POST http://localhost:8001/v1/vibevoice/generate \
  -H "Content-Type: application/json" \
  -d '{
    "script": "Speaker 0: Hello!\nSpeaker 1: Hi there!\nSpeaker 2: 你好！",
    "speakers": [
      {"speaker_id": 0, "voice_preset": "en-Alice_woman"},
      {"speaker_id": 1, "voice_preset": "in-Samuel_man"},
      {"speaker_id": 2, "voice_preset": "zh-Xinran_woman"}
    ],
    "response_format": "wav"
  }' \
  --output conversation.wav
```

---

## Adding Custom Voices

To add your own voice presets:

1. Place audio files (WAV, MP3, FLAC, etc.) in the `demo/voices/` directory
2. Name them with a descriptive pattern (e.g., `en-YourName_gender.wav`)
3. Restart the API server
4. The new voices will be automatically available

**Requirements for voice samples:**
- Duration: 3-10 seconds recommended
- Quality: Clear speech, minimal background noise
- Format: Any common audio format (WAV, MP3, FLAC, OGG, M4A, AAC)
- Sample rate: Will be automatically resampled to 24kHz

---

## Python SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"  # API key not required for local server
)

# Using OpenAI voice name
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello, this is a test."
)
response.stream_to_file("speech1.mp3")

# Using direct VibeVoice preset name
response = client.audio.speech.create(
    model="tts-1",
    voice="in-Samuel_man",  # Indian English voice
    input="Hello, this is a test with Indian accent."
)
response.stream_to_file("speech2.mp3")
```

---

## Tips

1. **Language Matching**: For best results, match the voice language to your text language
2. **Background Music**: Voices with `_bgm` suffix include subtle background music
3. **Voice Discovery**: Use `?show_all=true` to discover all available voices
4. **Fallback**: The API will try OpenAI names first, then VibeVoice preset names
5. **Error Messages**: If a voice is not found, the error will list all available options

---

## Summary

✅ **6 OpenAI-compatible voices** (alloy, echo, fable, onyx, nova, shimmer)  
✅ **9 total VibeVoice presets** (including Indian English and Chinese)  
✅ **Flexible naming** - use either OpenAI names or direct preset names  
✅ **Multi-speaker support** - mix different voices in one conversation  
✅ **Easy discovery** - use `?show_all=true` to see all options


