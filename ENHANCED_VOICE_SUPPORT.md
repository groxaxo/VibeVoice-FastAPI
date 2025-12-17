# Enhanced Voice Support - Implementation Summary

## What Changed

The OpenAI-compatible API has been enhanced to support **all 9 VibeVoice presets**, not just the 6 mapped OpenAI voices.

---

## Changes Made

### 1. Enhanced `/v1/audio/voices` Endpoint

**File**: `api/routers/openai_tts.py`

Added optional `show_all` query parameter:

```python
@router.get("/voices")
async def list_voices(show_all: bool = False, ...):
    if show_all:
        # Returns all 9 VibeVoice presets with language info
    else:
        # Returns 6 OpenAI-compatible voices (default)
```

**Usage:**
```bash
# Default: OpenAI voices only
curl http://localhost:8001/v1/audio/voices

# All voices including extras
curl http://localhost:8001/v1/audio/voices?show_all=true
```

### 2. Direct Preset Name Support in `/v1/audio/speech`

**File**: `api/routers/openai_tts.py`

The speech endpoint now accepts both:
- OpenAI voice names (`alloy`, `echo`, etc.)
- Direct VibeVoice preset names (`in-Samuel_man`, `zh-Anchen_man_bgm`, etc.)

**Implementation:**
```python
# Try OpenAI voice first
voice_audio = voices.load_voice_audio(request.voice, is_openai_voice=True)

# If not found, try as direct VibeVoice preset
if voice_audio is None:
    voice_audio = voices.load_voice_audio(request.voice, is_openai_voice=False)
```

**Usage:**
```bash
# Using OpenAI name
curl -X POST http://localhost:8001/v1/audio/speech \
  -d '{"voice": "alloy", ...}'

# Using direct preset name
curl -X POST http://localhost:8001/v1/audio/speech \
  -d '{"voice": "in-Samuel_man", ...}'
```

### 3. Documentation

Created comprehensive documentation:
- **VOICE_USAGE_GUIDE.md** - Complete guide to using all voices
- Updated **API_README.md** - Added reference to additional voices
- Updated **MODELS_GUIDE.md** - Added Realtime-0.5B incompatibility warning

---

## Available Voices

### OpenAI-Compatible (6 voices)

| OpenAI Name | VibeVoice Preset | Language | Gender |
|-------------|------------------|----------|--------|
| alloy | en-Alice_woman | English | Female |
| echo | en-Carter_man | English | Male |
| fable | en-Maya_woman | English | Female |
| onyx | en-Frank_man | English | Male |
| nova | en-Mary_woman_bgm | English | Female |
| shimmer | zh-Xinran_woman | Chinese | Female |

### Additional Presets (3 voices)

| VibeVoice Preset | Language | Gender | Notes |
|------------------|----------|--------|-------|
| in-Samuel_man | Indian English | Male | Indian accent |
| zh-Anchen_man_bgm | Chinese | Male | With BGM |
| zh-Bowen_man | Chinese | Male | Clear |

**Total: 9 voices** across English, Chinese, and Indian English

---

## Benefits

1. ✅ **Full Voice Access** - All 9 voices available via OpenAI-compatible API
2. ✅ **Backward Compatible** - Existing OpenAI voice names still work
3. ✅ **Flexible** - Use either OpenAI names or direct preset names
4. ✅ **Discoverable** - `?show_all=true` parameter shows all options
5. ✅ **Better Errors** - Error messages list both OpenAI and preset names
6. ✅ **Multi-Language** - Easy access to Chinese and Indian English voices

---

## Example Usage

### List All Voices

```bash
curl http://localhost:8001/v1/audio/voices?show_all=true | jq
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
      "name": "in-Samuel_man",
      "path": "demo/voices/in-Samuel_man.wav",
      "language": "Indian English"
    },
    ...
  ],
  "total": 9
}
```

### Use Indian English Voice

```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is Samuel speaking.",
    "voice": "in-Samuel_man",
    "response_format": "mp3"
  }' \
  --output indian_accent.mp3
```

### Use Chinese Voice with BGM

```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "你好，欢迎使用VibeVoice。",
    "voice": "zh-Anchen_man_bgm",
    "response_format": "mp3"
  }' \
  --output chinese_bgm.mp3
```

### Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"
)

# Use any voice - OpenAI name or direct preset
response = client.audio.speech.create(
    model="tts-1",
    voice="in-Samuel_man",  # Direct preset name
    input="Hello from India!"
)
response.stream_to_file("speech.mp3")
```

---

## Testing

To test the changes:

1. **List voices (default)**:
   ```bash
   curl http://localhost:8001/v1/audio/voices
   ```

2. **List all voices**:
   ```bash
   curl http://localhost:8001/v1/audio/voices?show_all=true
   ```

3. **Generate with OpenAI voice**:
   ```bash
   curl -X POST http://localhost:8001/v1/audio/speech \
     -H "Content-Type: application/json" \
     -d '{"model":"tts-1","input":"Test","voice":"alloy","response_format":"wav"}' \
     --output test1.wav
   ```

4. **Generate with direct preset**:
   ```bash
   curl -X POST http://localhost:8001/v1/audio/speech \
     -H "Content-Type: application/json" \
     -d '{"model":"tts-1","input":"Test","voice":"in-Samuel_man","response_format":"wav"}' \
     --output test2.wav
   ```

---

## Files Modified

1. `api/routers/openai_tts.py` - Enhanced voice listing and speech generation
2. `API_README.md` - Updated voice documentation
3. `MODELS_GUIDE.md` - Added Realtime model warning

## Files Created

1. `VOICE_USAGE_GUIDE.md` - Comprehensive voice usage documentation
2. `ENHANCED_VOICE_SUPPORT.md` - This file

---

## Backward Compatibility

✅ **100% Backward Compatible**

- Existing code using OpenAI voice names continues to work
- Default behavior unchanged (returns 6 OpenAI voices)
- New features are opt-in via query parameters or direct preset names
- No breaking changes to API contracts

---

## Future Enhancements

Potential future improvements:
- [ ] Voice preview samples in API response
- [ ] Voice characteristics metadata (pitch, speed, emotion)
- [ ] Custom voice upload endpoint
- [ ] Voice mixing/blending capabilities
- [ ] Voice emotion/style controls


