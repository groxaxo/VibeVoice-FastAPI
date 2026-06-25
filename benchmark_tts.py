#!/usr/bin/env python3
"""Benchmark VibeVoice TTS realtime performance (RTF = generation_time / audio_duration)."""
import requests
import time
import numpy as np
import subprocess
import json
import sys

# Configuration
SERVER_URL = "http://localhost:6969"
VOICE = "alloy"  # OpenAI-compatible name, maps to es-ar-Female_warm
TEXT = (
    "The quick brown fox jumps over the lazy dog. Artificial intelligence is transforming "
    "every industry. VibeVoice generates high-quality speech with diffusion models."
)
ITERATIONS = 5
RESPONSE_FORMAT = "wav"

def get_vram_gb():
    """Get current GPU VRAM usage in GB via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip()) / 1024  # MB to GB
    except Exception as e:
        print(f"Failed to get VRAM: {e}")
        return None

def run_benchmark():
    """Run TTS benchmark iterations."""
    print(f"=== VibeVoice TTS Benchmark ({ITERATIONS} iterations) ===")
    print(f"Voice: {VOICE}, Format: {RESPONSE_FORMAT}")
    print(f"Text: {TEXT}\n")

    url = f"{SERVER_URL}/v1/audio/speech"
    results = []

    for i in range(ITERATIONS):
        print(f"\n--- Iteration {i+1}/{ITERATIONS} ---")
        vram_before = get_vram_gb()

        start = time.time()
        try:
            response = requests.post(
                url,
                json={
                    "model": "tts-1",
                    "input": TEXT,
                    "voice": VOICE,
                    "response_format": RESPONSE_FORMAT,
                },
                timeout=60,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Request failed: {e}")
            continue

        wall_clock = time.time() - start
        audio_bytes = len(response.content)

        # Estimate audio duration: WAV (16-bit PCM, 24kHz) = bytes / (24000 * 2)
        # For other formats, this is approximate
        if RESPONSE_FORMAT == "wav":
            audio_duration = audio_bytes / (24000 * 2)
        else:
            audio_duration = audio_bytes / (24000 * 2 * 0.7)  # rough estimate for compressed

        rtf = wall_clock / audio_duration if audio_duration > 0 else None

        vram_after = get_vram_gb()
        vram_used = vram_after - vram_before if vram_before and vram_after else None

        result = {
            "iteration": i+1,
            "wall_clock_s": round(wall_clock, 3),
            "audio_duration_s": round(audio_duration, 3),
            "rtf": round(rtf, 3) if rtf else None,
            "vram_used_gb": round(vram_used, 2) if vram_used else None,
        }
        results.append(result)

        print(f"Wall clock: {wall_clock:.3f}s | Audio: {audio_duration:.3f}s | RTF: {rtf:.3f}x")
        if vram_used:
            print(f"VRAM delta: {vram_used:.2f} GB")

    # Summary statistics
    if results:
        rtfs = [r["rtf"] for r in results if r["rtf"]]
        avg_rtf = np.mean(rtfs) if rtfs else None
        std_rtf = np.std(rtfs) if rtfs else None
        avg_wall = np.mean([r["wall_clock_s"] for r in results])
        avg_audio = np.mean([r["audio_duration_s"] for r in results])

        print(f"\n=== Summary ===")
        print(f"Avg wall clock: {avg_wall:.3f}s")
        print(f"Avg audio duration: {avg_audio:.3f}s")
        print(f"Avg RTF: {avg_rtf:.3f}x (±{std_rtf:.3f})")

        # Save results
        output = {
            "config": {"voice": VOICE, "format": RESPONSE_FORMAT, "iterations": ITERATIONS},
            "results": results,
            "summary": {
                "avg_wall_clock_s": round(avg_wall, 3),
                "avg_audio_duration_s": round(avg_audio, 3),
                "avg_rtf": round(avg_rtf, 3) if avg_rtf else None,
                "std_rtf": round(std_rtf, 3) if std_rtf else None,
            }
        }
        with open("benchmark_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
