# VibeVoice Models Guide

## Available Models in Your Cache

You have three VibeVoice models cached locally in `~/.cache/huggingface/hub/`:

### 1. VibeVoice-1.5B (Default)
- **Size**: 1.5 billion parameters
- **Best for**: Balanced quality and speed
- **VRAM**: ~6GB
- **Local Path**:
```
/home/nishant/.cache/huggingface/hub/models--microsoft--VibeVoice-1.5B/snapshots/1904eae38036e9c780d28e27990c27748984eafe
```
- **HuggingFace ID**: `microsoft/VibeVoice-1.5B`

### 2. VibeVoice-Realtime-0.5B ⚠️ **NOT COMPATIBLE**
- **Size**: 0.5 billion parameters
- **Status**: ❌ **Not compatible with this API**
- **Reason**: This is a `vibevoice_streaming` model type with different architecture
- **VRAM**: ~2-3GB
- **Local Path**:
```
/home/nishant/.cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B/snapshots/6bce5f06044837fe6d2c5d7a71a84f0416bd57e4
```
- **HuggingFace ID**: `microsoft/VibeVoice-Realtime-0.5B`
- **Note**: This model requires a different inference implementation and is not supported by the current API.

### 3. VibeVoice-Large (7B)
- **Size**: 7 billion parameters
- **Best for**: Highest quality, most expressive
- **VRAM**: ~14-16GB
- **Local Path**:
```
/home/nishant/.cache/huggingface/hub/models--rsxdalv--VibeVoice-Large/snapshots/01d40dfb38f6eb1c87f924b7ba87deaa3dca48ba
```
- **HuggingFace ID**: `rsxdalv/VibeVoice-Large` or `microsoft/VibeVoice-Large`

---

## How to Switch Models

### Method 1: Edit `.env` file

Open `.env` and change the `VIBEVOICE_MODEL_PATH`:

```bash
# For 1.5B model (default)
VIBEVOICE_MODEL_PATH=/home/nishant/.cache/huggingface/hub/models--microsoft--VibeVoice-1.5B/snapshots/1904eae38036e9c780d28e27990c27748984eafe

# For Realtime 0.5B model (NOT COMPATIBLE - DO NOT USE)
# VIBEVOICE_MODEL_PATH=/home/nishant/.cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B/snapshots/6bce5f06044837fe6d2c5d7a71a84f0416bd57e4

# For Large 7B model (best quality)
VIBEVOICE_MODEL_PATH=/home/nishant/.cache/huggingface/hub/models--rsxdalv--VibeVoice-Large/snapshots/01d40dfb38f6eb1c87f924b7ba87deaa3dca48ba
```

Then restart the server:
```bash
./start.sh
```

### Method 2: Use Environment Variable

```bash
# Start with specific model
VIBEVOICE_MODEL_PATH=/home/nishant/.cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B/snapshots/6bce5f06044837fe6d2c5d7a71a84f0416bd57e4 ./start.sh
```

### Method 3: Use HuggingFace Model ID (Auto-uses cache)

```bash
# In .env file
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-1.5B
# or
VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-Realtime-0.5B
# or
VIBEVOICE_MODEL_PATH=rsxdalv/VibeVoice-Large
```

**Note**: When using HuggingFace model IDs, the transformers library will automatically use your local cache if available, so no re-download is needed.

---

## Model Comparison

| Model | Parameters | VRAM | Speed | Quality | Best Use Case | Status |
|-------|-----------|------|-------|---------|---------------|--------|
| Realtime-0.5B | 500M | ~2-3GB | ⚡⚡⚡ Fast | ⭐⭐ Good | Real-time apps, low VRAM | ❌ Not Compatible |
| 1.5B | 1.5B | ~6GB | ⚡⚡ Medium | ⭐⭐⭐ Great | Balanced production use | ✅ Recommended |
| Large-7B | 7B | ~14-16GB | ⚡ Slower | ⭐⭐⭐⭐ Excellent | Highest quality needed | ✅ Supported |

---

## Performance Tips

### For Faster Generation
1. Use the Realtime-0.5B model
2. Reduce inference steps: `VIBEVOICE_INFERENCE_STEPS=5`
3. Use flash_attention_2 if available

### For Better Quality
1. Use the Large-7B model
2. Increase inference steps: `VIBEVOICE_INFERENCE_STEPS=20`
3. Adjust CFG scale: `DEFAULT_CFG_SCALE=1.5`

### For Lower VRAM
1. Use Realtime-0.5B model
2. Use float16 instead of bfloat16: `VIBEVOICE_DTYPE=float16`
3. Consider CPU inference for very low VRAM: `VIBEVOICE_DEVICE=cpu`

---

## Current Configuration

Your `.env` file is currently set to use:
- **Model**: VibeVoice-1.5B (local cache)
- **Device**: CUDA
- **Inference Steps**: 10
- **Port**: 8001

To check which model is loaded, visit:
```bash
curl http://localhost:8001/v1/vibevoice/health
```

---

## Troubleshooting

### Model Not Found Error
If you get a "model not found" error with a local path, verify the path exists:
```bash
ls -la /home/nishant/.cache/huggingface/hub/models--microsoft--VibeVoice-1.5B/snapshots/
```

### Out of Memory Error
- Switch to a smaller model (Realtime-0.5B)
- Use float16: `VIBEVOICE_DTYPE=float16`
- Reduce batch size or use CPU

### Slow Generation
- Switch to Realtime-0.5B model
- Reduce inference steps to 5-7
- Ensure flash_attention_2 is installed and being used

