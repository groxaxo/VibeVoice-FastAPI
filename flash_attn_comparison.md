# Flash Attention 2 vs SDPA Performance Comparison

## Environment
- **GPU**: RTX 3060 12GB (sm_86, Ampere)
- **Model**: VibeVoice-Large-AWQ (AutoAWQ quantized, 4-bit)
- **Framework**: torch 2.4.1+cu121, transformers 4.51.3
- **flash-attn**: 2.8.3.post1 (prebuilt wheel)
- **Test**: 5 iterations, fixed text, `alloy` voice, WAV format

## Results Summary

| Metric | SDPA (Baseline) | Flash Attention 2 | Delta |
|--------|----------------|-------------------|-------|
| **Avg Wall Clock** | 27.179s | 25.736s | -5.3% |
| **Avg Audio Duration** | 17.627s | 15.974s | -9.4% |
| **Avg RTF** | **1.542x** | **1.608x** | +4.3% (worse) |
| **RTF Std Dev** | ±0.007 | ±0.080 | Higher variance |

## Per-Iteration Details

### SDPA (Baseline)
| Iter | Wall Clock | Audio | RTF |
|------|------------|-------|-----|
| 1 | 26.79s | 17.33s | 1.545x |
| 2 | 27.18s | 17.73s | 1.532x |
| 3 | 28.58s | 18.53s | 1.542x |
| 4 | 25.68s | 16.53s | 1.553x |
| 5 | 27.67s | 18.00s | 1.537x |

### Flash Attention 2
| Iter | Wall Clock | Audio | RTF |
|------|------------|-------|-----|
| 1 | 30.40s | 17.20s | 1.767x ⚠️ |
| 2 | 26.58s | 17.07s | 1.557x |
| 3 | 25.07s | 16.00s | 1.567x |
| 4 | 23.27s | 14.80s | 1.572x |
| 5 | 23.37s | 14.80s | 1.579x |

## Analysis

### Key Findings
1. **FA2 is ~4.3% slower in RTF** on this AWQ-quantized VibeVoice model
2. **Higher variance** with FA2 (std 0.080 vs 0.007)
3. **First iteration outlier**: FA2 iteration 1 took significantly longer (30.40s vs ~26-28s for others)
4. **Stable after warmup**: Ignoring iteration 1, FA2 RTF ~1.57x (closer to SDPA 1.54x)

### Possible Explanations
- **AWQ quantization + FA2 interaction**: AutoAWQ's 4-bit quantization may not benefit from FA2's fused kernels, or may even add overhead
- **Model architecture**: VibeVoice is a diffusion TTS model with complex multi-modal attention (acoustic + semantic tokenizers + diffusion head) — the attention pattern may not map cleanly to FA2's optimizations
- **Sequence length**: Generated audio varies (14.8-18.5s), affecting attention patterns; shorter sequences with FA2 may see less benefit
- **RTX 3060 mid-range**: Ampere but with limited memory bandwidth; FA2 benefits are more pronounced on higher-end GPUs (A100/H100)

### Server Logs Confirmation
Server-reported timing matches client-side measurements:
```
FA2 Iteration 1: Audio 17.20s, Gen 30.38s → RTF 1.766x
FA2 Iteration 5: Audio 14.80s, Gen 23.37s → RTF 1.579x
```

## Recommendation

**For this specific setup (VibeVoice-Large-AWQ on RTX 3060), SDPA appears preferable:**
- More stable performance (lower variance)
- Slightly better average RTF
- No additional dependency (flash-attn)

**FA2 may still be worth testing on:**
- Higher-end GPUs (A100/H100)
- Non-quantized (full precision) models
- Models with longer sequences / more attention-heavy architectures

## Artifacts
- Baseline results: `benchmark_results_sdpa_baseline.json`
- FA2 results: `benchmark_results_fa2.json`
- Server log: `/tmp/vibevoice-fa2.log`
- Benchmark script: `benchmark_tts.py`
