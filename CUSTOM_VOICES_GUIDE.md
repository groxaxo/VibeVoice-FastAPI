# Custom Voices Guide

## Overview

The VibeVoice API **automatically discovers and loads all audio files** from your voices directory. You can add as many custom voices as you want - just drop audio files in the folder and restart the server!

---

## How It Works

### 1. Automatic Voice Discovery

The API scans the voices directory on startup and automatically creates a preset for **every audio file** it finds.

**Supported Formats:**
- `.wav` (recommended)
- `.mp3`
- `.flac`
- `.ogg`
- `.m4a`
- `.aac`

**Preset Naming:**
- The filename (without extension) becomes the voice preset name
- Example: `my-custom-voice.wav` → preset name: `my-custom-voice`

### 2. Configure Voices Directory

In your `.env` file:

```bash
# Default location
VOICES_DIR=demo/voices

# Or use a custom directory (absolute or relative path)
VOICES_DIR=/path/to/my/custom/voices

# Or relative to project root
VOICES_DIR=./my_voices
```

---

## Adding Custom Voices

### Step 1: Prepare Voice Samples

**Requirements:**
- **Duration**: 3-10 seconds recommended
  - Too short: May not capture voice characteristics well
  - Too long: Unnecessary, increases memory usage
- **Quality**: Clear speech, minimal background noise
- **Content**: Natural speech (not singing or whispering)
- **Format**: Any supported format (WAV recommended for quality)

**Tips:**
- Use a consistent speaking style
- Avoid background music (unless you want it in the output)
- Record at 16kHz or higher (will be resampled to 24kHz automatically)
- Mono or stereo both work (stereo will be converted to mono)

### Step 2: Name Your Files

Use a descriptive naming pattern:

```bash
# Recommended pattern: language-name_gender.wav
en-John_man.wav
en-Sarah_woman.wav
es-Maria_woman.wav
fr-Pierre_man.wav
de-Hans_man.wav

# Or any pattern you prefer
customer_service_voice.wav
narrator_deep.wav
excited_announcer.wav
```

### Step 3: Add to Voices Directory

```bash
# Copy files to the voices directory
cp my-voice-sample.wav demo/voices/

# Or create a custom directory
mkdir my_voices
cp *.wav my_voices/

# Update .env to point to your directory
echo "VOICES_DIR=my_voices" >> .env
```

### Step 4: Restart the Server

```bash
./start.sh
```

The server will automatically:
1. Scan the voices directory
2. Load all audio files
3. Create presets with names matching the filenames
4. Print a list of available voices

**Example Output:**
```
Loaded 15 voice presets from my_voices
Available voices: customer_service, en-Alice_woman, en-Carter_man, en-John_man, en-Sarah_woman, es-Maria_woman, excited_announcer, fr-Pierre_man, narrator_deep, ...
```

---

## Using Custom Voices

### Via OpenAI-Compatible API

```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is my custom voice!",
    "voice": "en-John_man",
    "response_format": "mp3"
  }' \
  --output output.mp3
```

### Via VibeVoice Extended API (Multi-Speaker)

```bash
curl -X POST http://localhost:8001/v1/vibevoice/generate \
  -H "Content-Type: application/json" \
  -d '{
    "script": "Speaker 0: Welcome!\nSpeaker 1: Thanks for having me!",
    "speakers": [
      {"speaker_id": 0, "voice_preset": "customer_service"},
      {"speaker_id": 1, "voice_preset": "en-John_man"}
    ],
    "response_format": "wav"
  }' \
  --output conversation.wav
```

### List All Available Voices

```bash
# See all voices including your custom ones
curl http://localhost:8001/v1/audio/voices?show_all=true
```

---

## Directory Structure Examples

### Example 1: Single Directory

```
demo/voices/
├── en-Alice_woman.wav
├── en-Carter_man.wav
├── en-John_man.wav          # Your custom voice
├── en-Sarah_woman.wav       # Your custom voice
└── customer_service.wav     # Your custom voice
```

`.env`:
```bash
VOICES_DIR=demo/voices
```

### Example 2: Custom Directory

```
my_company_voices/
├── agent_friendly.wav
├── agent_professional.wav
├── narrator_calm.wav
├── narrator_energetic.wav
└── ivr_system.wav
```

`.env`:
```bash
VOICES_DIR=my_company_voices
```

### Example 3: Organized by Language

```
voices/
├── english/
│   ├── john.wav
│   ├── sarah.wav
│   └── mike.wav
├── spanish/
│   ├── maria.wav
│   └── carlos.wav
└── french/
    └── pierre.wav
```

**Note**: The API only scans the top-level directory, not subdirectories. You'll need to flatten it:

```bash
voices/
├── en-john.wav
├── en-sarah.wav
├── en-mike.wav
├── es-maria.wav
├── es-carlos.wav
└── fr-pierre.wav
```

`.env`:
```bash
VOICES_DIR=voices
```

---

## Advanced: Multiple Voice Directories

If you want to support multiple voice directories (e.g., for different projects), you can:

### Option 1: Use Environment Variables

```bash
# Project A
VOICES_DIR=/path/to/project_a/voices ./start.sh

# Project B
VOICES_DIR=/path/to/project_b/voices ./start.sh
```

### Option 2: Symlinks

```bash
# Create a voices directory with symlinks
mkdir all_voices
ln -s /path/to/project_a/voices/*.wav all_voices/
ln -s /path/to/project_b/voices/*.wav all_voices/

# Use the combined directory
echo "VOICES_DIR=all_voices" >> .env
```

### Option 3: Copy/Merge

```bash
# Merge multiple directories
mkdir merged_voices
cp project_a/voices/*.wav merged_voices/
cp project_b/voices/*.wav merged_voices/

echo "VOICES_DIR=merged_voices" >> .env
```

---

## Voice Sample Recording Tips

### Using Audacity (Free)

1. **Record**:
   - File → New
   - Click red record button
   - Speak naturally for 5-10 seconds
   - Click stop

2. **Edit**:
   - Select silent parts at start/end → Delete
   - Effect → Normalize → OK
   - Effect → Noise Reduction (optional)

3. **Export**:
   - File → Export → Export as WAV
   - Save as `your-voice-name.wav`

### Using Python (Programmatic)

```python
import sounddevice as sd
import soundfile as sf
import numpy as np

# Record 5 seconds at 24kHz
duration = 5  # seconds
sample_rate = 24000

print("Recording...")
audio = sd.rec(int(duration * sample_rate), 
               samplerate=sample_rate, 
               channels=1, 
               dtype='float32')
sd.wait()
print("Done!")

# Save
sf.write('my-voice.wav', audio, sample_rate)
```

### Using FFmpeg (Convert Existing Audio)

```bash
# Extract 5 seconds from an audio file
ffmpeg -i input.mp3 -ss 00:00:10 -t 00:00:05 -ar 24000 -ac 1 output.wav

# Convert any format to WAV
ffmpeg -i input.m4a -ar 24000 -ac 1 output.wav

# Trim silence from start/end
ffmpeg -i input.wav -af silenceremove=1:0:-50dB output.wav
```

---

## Troubleshooting

### Voice Not Appearing

1. **Check file format**:
   ```bash
   file demo/voices/my-voice.wav
   # Should show: RIFF (little-endian) data, WAVE audio
   ```

2. **Check file permissions**:
   ```bash
   ls -la demo/voices/
   # Files should be readable
   ```

3. **Check server logs**:
   ```bash
   # Look for "Loaded X voice presets" message
   # Should list your voice name
   ```

4. **Restart server**:
   ```bash
   # Voices are loaded on startup only
   ./start.sh
   ```

### Voice Quality Issues

1. **Audio too quiet**: Normalize the audio
   ```bash
   ffmpeg -i input.wav -af loudnorm output.wav
   ```

2. **Background noise**: Use noise reduction
   ```bash
   ffmpeg -i input.wav -af "highpass=f=200, lowpass=f=3000" output.wav
   ```

3. **Wrong sample rate**: Resample to 24kHz
   ```bash
   ffmpeg -i input.wav -ar 24000 output.wav
   ```

### Voice Not Loading

Check the server startup logs:
```
Loaded 9 voice presets from demo/voices
Available voices: en-Alice_woman, en-Carter_man, ...
```

If your voice isn't listed:
- Ensure the file has a supported extension
- Check that the file isn't corrupted
- Verify the voices directory path in `.env`

---

## Best Practices

### ✅ DO:
- Use descriptive, unique names
- Keep samples 3-10 seconds long
- Use clear, natural speech
- Test voices after adding them
- Use WAV format for best quality
- Normalize audio levels

### ❌ DON'T:
- Use special characters in filenames (stick to letters, numbers, hyphens, underscores)
- Use very long samples (>30 seconds)
- Include multiple speakers in one sample
- Use low-quality recordings
- Forget to restart the server after adding voices

---

## Example Workflow

### Adding a New Voice

```bash
# 1. Record or obtain voice sample
# (Use Audacity, phone recording, etc.)

# 2. Process the audio (optional but recommended)
ffmpeg -i raw_recording.m4a \
  -ss 00:00:05 -t 00:00:05 \
  -ar 24000 -ac 1 \
  -af "loudnorm,silenceremove=1:0:-50dB" \
  demo/voices/en-NewVoice_woman.wav

# 3. Verify the file
file demo/voices/en-NewVoice_woman.wav
ls -lh demo/voices/en-NewVoice_woman.wav

# 4. Restart the server
./start.sh

# 5. List voices to confirm
curl http://localhost:8001/v1/audio/voices?show_all=true | jq '.voices[] | select(.name | contains("NewVoice"))'

# 6. Test the voice
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Testing my new custom voice!",
    "voice": "en-NewVoice_woman",
    "response_format": "mp3"
  }' \
  --output test.mp3

# 7. Listen and adjust if needed
```

---

## Summary

✅ **Automatic Discovery** - Just drop files in the folder  
✅ **Any Format** - WAV, MP3, FLAC, OGG, M4A, AAC  
✅ **Unlimited Voices** - Add as many as you want  
✅ **Custom Directory** - Configure via `VOICES_DIR` in `.env`  
✅ **No Code Changes** - Everything is automatic  
✅ **Instant Access** - Available via API immediately after restart

**The system is already set up for unlimited custom voices!** 🎉

Just:
1. Add audio files to your voices directory
2. Restart the server
3. Use the filename (without extension) as the voice name in API requests


