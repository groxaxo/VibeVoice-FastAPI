# Unlimited Custom Voices Feature

## 🎉 Overview

The VibeVoice API **already supports unlimited custom voices** out of the box! The system automatically discovers and loads all audio files from your configured voices directory.

---

## ✨ How It Works

### Automatic Discovery

The API scans the `VOICES_DIR` folder on startup and:

1. **Finds all audio files** (WAV, MP3, FLAC, OGG, M4A, AAC)
2. **Creates voice presets** using the filename (without extension)
3. **Makes them available** via both OpenAI and VibeVoice endpoints
4. **No configuration needed** - it just works!

### Configuration

In your `.env` file:

```bash
# Default location
VOICES_DIR=demo/voices

# Or specify any directory (absolute or relative)
VOICES_DIR=/path/to/your/voices
VOICES_DIR=./my_custom_voices
VOICES_DIR=/mnt/shared/company_voices
```

---

## 🚀 Quick Start

### 1. Add a Voice File

```bash
# Copy your voice sample to the voices directory
cp my-recording.wav demo/voices/my-custom-voice.wav
```

**File Requirements:**
- **Format**: WAV, MP3, FLAC, OGG, M4A, or AAC
- **Duration**: 3-10 seconds recommended
- **Quality**: Clear speech, minimal background noise
- **Naming**: Use descriptive names (letters, numbers, hyphens, underscores)

### 2. Restart the Server

```bash
./start.sh
```

The server will automatically:
- Scan the voices directory
- Load your new voice file
- Print: `Loaded X voice presets from demo/voices`
- List all available voices

### 3. Use Your Voice

```bash
# Via OpenAI-compatible API
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello from my custom voice!",
    "voice": "my-custom-voice",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

**That's it!** No code changes, no configuration files, no database updates.

---

## 📁 Directory Structure

### Example 1: Default Setup

```
demo/voices/
├── en-Alice_woman.wav          # Built-in
├── en-Carter_man.wav           # Built-in
├── my-custom-voice.wav         # Your voice
├── customer-service.wav        # Your voice
└── narrator-deep.wav           # Your voice
```

All 5 voices are automatically available!

### Example 2: Custom Directory

```bash
# Create your own voices folder
mkdir /opt/tts_voices

# Add your voices
cp voice1.wav /opt/tts_voices/
cp voice2.mp3 /opt/tts_voices/
cp voice3.flac /opt/tts_voices/

# Configure in .env
echo "VOICES_DIR=/opt/tts_voices" >> .env

# Restart
./start.sh
```

### Example 3: Organized by Purpose

```
company_voices/
├── sales-john.wav
├── sales-sarah.wav
├── support-mike.wav
├── support-lisa.wav
├── ivr-system.wav
├── narrator-calm.wav
└── narrator-energetic.wav
```

```bash
# In .env
VOICES_DIR=company_voices
```

All 7 voices are automatically loaded!

---

## 🎯 Use Cases

### 1. Customer Service

```
voices/
├── agent-friendly.wav
├── agent-professional.wav
├── agent-empathetic.wav
└── manager-escalation.wav
```

### 2. E-Learning

```
voices/
├── instructor-main.wav
├── instructor-enthusiastic.wav
├── student-curious.wav
└── narrator-educational.wav
```

### 3. Audiobook Production

```
voices/
├── narrator-male-deep.wav
├── narrator-female-warm.wav
├── character-hero.wav
├── character-villain.wav
└── character-sidekick.wav
```

### 4. Multi-Language Support

```
voices/
├── en-narrator.wav
├── es-narrator.wav
├── fr-narrator.wav
├── de-narrator.wav
├── zh-narrator.wav
└── ja-narrator.wav
```

### 5. Brand Voices

```
voices/
├── brand-commercial-male.wav
├── brand-commercial-female.wav
├── brand-explainer.wav
└── brand-testimonial.wav
```

---

## 🔄 Workflow

### Adding Voices at Scale

```bash
#!/bin/bash
# Script to add multiple voices

VOICES_DIR="demo/voices"

# Add voices from a collection
for voice_file in /path/to/voice/collection/*.wav; do
    # Extract name
    name=$(basename "$voice_file" .wav)
    
    # Copy to voices directory
    cp "$voice_file" "$VOICES_DIR/${name}.wav"
    
    echo "Added: $name"
done

echo "Restart the server to load new voices"
```

### Voice Management

```bash
# List current voices
ls -1 demo/voices/*.{wav,mp3,flac,ogg,m4a,aac} 2>/dev/null

# Count voices
ls -1 demo/voices/*.{wav,mp3,flac,ogg,m4a,aac} 2>/dev/null | wc -l

# Remove a voice
rm demo/voices/unwanted-voice.wav
# (restart server to update)

# Backup voices
tar -czf voices-backup-$(date +%Y%m%d).tar.gz demo/voices/

# Restore voices
tar -xzf voices-backup-20250101.tar.gz
```

---

## 🎤 Recording Voice Samples

### Quick Recording with FFmpeg

```bash
# Record 5 seconds from microphone
ffmpeg -f alsa -i default -t 5 -ar 24000 -ac 1 my-voice.wav

# Or on macOS
ffmpeg -f avfoundation -i ":0" -t 5 -ar 24000 -ac 1 my-voice.wav
```

### Extract from Existing Audio

```bash
# Extract 5 seconds starting at 10 seconds
ffmpeg -i source.mp3 -ss 00:00:10 -t 00:00:05 -ar 24000 -ac 1 voice.wav

# Trim silence and normalize
ffmpeg -i input.wav \
  -af "silenceremove=1:0:-50dB,loudnorm" \
  -ar 24000 -ac 1 \
  output.wav
```

### Batch Processing

```bash
# Convert all MP3s to optimized WAV
for file in *.mp3; do
    ffmpeg -i "$file" \
      -ar 24000 -ac 1 \
      -af "loudnorm" \
      "demo/voices/$(basename "$file" .mp3).wav"
done
```

---

## 🔍 Discovery and Testing

### List All Voices via API

```bash
# Get all voices with details
curl http://localhost:8001/v1/audio/voices?show_all=true | jq

# Get just the names
curl -s http://localhost:8001/v1/audio/voices?show_all=true | jq -r '.voices[].name'

# Count voices
curl -s http://localhost:8001/v1/audio/voices?show_all=true | jq '.total'
```

### Test a Voice

```bash
# Quick test function
test_voice() {
    local voice_name="$1"
    local text="${2:-Hello, this is a test of the $voice_name voice.}"
    
    curl -X POST http://localhost:8001/v1/audio/speech \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"tts-1\",\"input\":\"$text\",\"voice\":\"$voice_name\",\"response_format\":\"mp3\"}" \
      --output "test_${voice_name}.mp3"
    
    echo "Generated: test_${voice_name}.mp3"
}

# Usage
test_voice "my-custom-voice"
test_voice "customer-service" "Welcome to our support line"
```

### Test All Voices

```bash
# Test every voice
for voice in $(curl -s http://localhost:8001/v1/audio/voices?show_all=true | jq -r '.voices[].name'); do
    echo "Testing: $voice"
    test_voice "$voice"
done
```

---

## 📊 Monitoring

### Check Voice Loading on Startup

Watch the server logs for:

```
Loaded 15 voice presets from demo/voices
Available voices: customer-service, en-Alice_woman, en-Carter_man, my-custom-voice, narrator-deep, ...
```

### Verify Voice Availability

```bash
# Check if specific voice is loaded
curl -s http://localhost:8001/v1/audio/voices?show_all=true | \
  jq '.voices[] | select(.name == "my-custom-voice")'
```

---

## 🛠️ Advanced Features

### Voice Metadata (Future Enhancement)

While not currently implemented, you could extend the system to support metadata files:

```
voices/
├── narrator.wav
├── narrator.json          # Metadata
├── customer-service.wav
└── customer-service.json  # Metadata
```

`narrator.json`:
```json
{
  "name": "Professional Narrator",
  "language": "en-US",
  "gender": "male",
  "age_range": "30-40",
  "characteristics": ["deep", "authoritative", "clear"],
  "best_for": ["audiobooks", "documentaries", "corporate"]
}
```

### Voice Aliases

You could create symlinks for voice aliases:

```bash
# Create alias
ln -s en-Alice_woman.wav demo/voices/alice.wav

# Now accessible as both:
# - en-Alice_woman
# - alice
```

### Voice Collections

Organize voices in separate directories and switch between them:

```bash
# Production voices
VOICES_DIR=/opt/production_voices ./start.sh

# Development voices
VOICES_DIR=/opt/dev_voices ./start.sh

# Client-specific voices
VOICES_DIR=/opt/client_abc_voices ./start.sh
```

---

## 💡 Best Practices

### ✅ DO:

1. **Use descriptive names**: `customer-service-friendly` not `voice1`
2. **Consistent naming**: `lang-name_gender` pattern (e.g., `en-John_man`)
3. **Optimize audio**: 24kHz, mono, normalized volume
4. **Test voices**: Generate samples before production use
5. **Backup regularly**: Keep copies of your voice files
6. **Document voices**: Maintain a list of what each voice is for

### ❌ DON'T:

1. **Use special characters**: Stick to `a-z`, `0-9`, `-`, `_`
2. **Use very long files**: Keep samples 3-10 seconds
3. **Mix multiple speakers**: One voice per file
4. **Forget to restart**: Server loads voices on startup only
5. **Use low quality**: Poor input = poor output

---

## 🎓 Examples

### Example 1: Simple Addition

```bash
# You have a recording
ls my-voice.wav

# Add it
cp my-voice.wav demo/voices/

# Restart
./start.sh

# Use it
curl -X POST http://localhost:8001/v1/audio/speech \
  -d '{"voice":"my-voice","input":"Test","response_format":"mp3"}' \
  --output test.mp3
```

### Example 2: Batch Addition

```bash
# You have multiple recordings
ls recordings/
# voice1.mp3
# voice2.mp3
# voice3.mp3

# Convert and add all
for f in recordings/*.mp3; do
    name=$(basename "$f" .mp3)
    ffmpeg -i "$f" -ar 24000 -ac 1 "demo/voices/${name}.wav"
done

# Restart
./start.sh

# All 3 voices now available!
```

### Example 3: Custom Directory

```bash
# Create dedicated directory
mkdir /var/tts/voices

# Add voices
cp voice*.wav /var/tts/voices/

# Configure
echo "VOICES_DIR=/var/tts/voices" >> .env

# Restart
./start.sh
```

---

## 📚 Related Documentation

- **[CUSTOM_VOICES_GUIDE.md](CUSTOM_VOICES_GUIDE.md)** - Comprehensive guide
- **[VOICE_USAGE_GUIDE.md](VOICE_USAGE_GUIDE.md)** - Using voices in API
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick commands
- **[example_add_voice.sh](example_add_voice.sh)** - Helper script

---

## 🎉 Summary

### The Feature You Asked For Is Already Built-In!

✅ **Unlimited voices** - Add as many as you want  
✅ **Automatic discovery** - Just drop files and restart  
✅ **Any format** - WAV, MP3, FLAC, OGG, M4A, AAC  
✅ **Custom directory** - Configure via `VOICES_DIR` in `.env`  
✅ **No code changes** - Everything is automatic  
✅ **OpenAI compatible** - Works with existing clients  

### Simple Workflow:

1. **Add** audio file to `VOICES_DIR`
2. **Restart** server
3. **Use** filename (without extension) as voice name

**That's it!** 🚀

The system is already set up exactly as you requested - you can specify a folder in the `.env` file (`VOICES_DIR`) and it will automatically map to as many voices as you want!


