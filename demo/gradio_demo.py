"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import json
import os
import sys
import tempfile
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Iterator
from datetime import datetime
import threading
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import os
import traceback

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed
from api.utils.text_chunking import split_text_chunks

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        """Initialize the VibeVoice demo with model loading."""
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        self.load_model()
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts
        
    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"Loading processor & model from {self.model_path}")
        # Normalize potential 'mpx'
        if self.device.lower() == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            self.device = "mps"
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            self.device = "cpu"
        print(f"Using device: {self.device}")
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"
        print(f"Using device: {self.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            if self.device == "mps":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl_primary,
                    device_map=None,
                )
                self.model.to("mps")
            elif self.device == "cuda":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl_primary,
                )
            else:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl_primary,
                )
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print(f"[ERROR] : {type(e).__name__}: {e}")
                print(traceback.format_exc())
                fallback_attn = "sdpa"
                print(f"Falling back to attention implementation: {fallback_attn}")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation=fallback_attn,
                )
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise e
        self.model.eval()
        
        # Use SDE solver by default
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
    
    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all WAV files in the voices directory
        self.voice_presets = {}
        
        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            print("No bundled voice presets found; Studio will use uploaded references.")
            return
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")
    
    def read_audio(
        self,
        audio_path: str,
        target_sr: int = 24000,
        trim_start: float = 0.0,
        trim_end: float = 0.0,
        normalize: bool = True,
    ) -> np.ndarray:
        """Read, trim, resample, and optionally normalize a voice reference."""
        try:
            wav, sr = sf.read(audio_path, dtype="float32")
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            duration = len(wav) / float(sr)
            start = max(0.0, float(trim_start or 0.0))
            end = float(trim_end or 0.0)
            end = duration if end <= 0 else min(end, duration)
            if start >= duration or end <= start:
                raise ValueError(
                    f"Invalid trim range {start:.2f}s-{end:.2f}s for a {duration:.2f}s voice sample"
                )
            wav = wav[int(start * sr) : int(end * sr)]
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            if normalize and wav.size:
                peak = float(np.max(np.abs(wav)))
                if peak > 0:
                    wav = wav * (0.95 / peak)
            return np.ascontiguousarray(wav)
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])

    @staticmethod
    def _format_script(script: str, num_speakers: int) -> str:
        """Normalize labels and auto-assign unlabeled lines in speaker rotation."""
        raw_lines = [
            line.strip()
            for line in script.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        ]
        labelled_ids = [
            int(match.group(1))
            for line in raw_lines
            if (match := re.match(r"^Speaker\s+(\d+)\s*:", line, re.IGNORECASE))
        ]
        one_based = bool(labelled_ids) and 0 not in labelled_ids and all(
            1 <= speaker_id <= num_speakers for speaker_id in labelled_ids
        )
        formatted = []
        for line in raw_lines:
            match = re.match(r"^Speaker\s+(\d+)\s*:\s*(.*)$", line, re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1)) - (1 if one_based else 0)
                if speaker_id >= num_speakers:
                    raise gr.Error(
                        f"Script references Speaker {speaker_id}, but only {num_speakers} speaker(s) are enabled."
                    )
                formatted.append(f"Speaker {speaker_id}: {match.group(2).strip()}")
            else:
                speaker_id = len(formatted) % num_speakers
                formatted.append(f"Speaker {speaker_id}: {line}")
        return "\n".join(formatted)

    @staticmethod
    def _chunk_script(script: str, min_chars: int, max_chars: int) -> List[str]:
        """Pack labelled turns into sentence-aware, model-safe script chunks."""
        if min_chars > max_chars:
            raise gr.Error("Minimum chunk size cannot exceed maximum chunk size.")

        units = []
        for line in script.splitlines():
            match = re.match(r"^(Speaker\s+\d+\s*:)(.*)$", line, re.IGNORECASE)
            if not match:
                continue
            label, utterance = match.group(1), match.group(2).strip()
            text_budget = max_chars - len(label) - 1
            if text_budget < 32:
                raise gr.Error("Maximum chunk size is too small for the speaker label.")
            for part in split_text_chunks(
                utterance,
                min_chars=min(min_chars, text_budget),
                max_chars=text_budget,
            ):
                units.append(f"{label} {part}")

        chunks = []
        current = ""
        for unit in units:
            candidate = f"{current}\n{unit}" if current else unit
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = unit
        if current:
            chunks.append(current)
        return chunks

    def generate_podcast_streaming(
        self,
        num_speakers: int,
        script: str,
        speaker_presets: List[str],
        uploaded_voices: List[str],
        trim_starts: List[float],
        trim_ends: List[float],
        cfg_scale: float = 1.3,
        inference_steps: int = 10,
        seed: int = 42,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_chunk_chars: int = 1000,
        max_chunk_chars: int = 2000,
        chunk_pause_seconds: float = 0.15,
        normalize_voices: bool = True,
    ) -> Iterator[tuple]:
        """Generate unlimited text as bounded chunks and concatenate their audio."""
        try:
            self.stop_generation = False
            self.is_generating = True
            if not script.strip():
                raise gr.Error("Error: Please provide a script.")
            script = script.replace("’", "'")
            if num_speakers < 1 or num_speakers > 4:
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")

            min_chunk_chars = int(min_chunk_chars)
            max_chunk_chars = int(max_chunk_chars)
            formatted_script = self._format_script(script, int(num_speakers))
            script_chunks = self._chunk_script(
                formatted_script,
                min_chars=min_chunk_chars,
                max_chars=max_chunk_chars,
            )
            if not script_chunks:
                raise gr.Error("Error: The script contains no speakable text.")

            voice_samples = []
            voice_labels = []
            for index in range(int(num_speakers)):
                uploaded = uploaded_voices[index] if index < len(uploaded_voices) else None
                preset = speaker_presets[index] if index < len(speaker_presets) else None
                if uploaded:
                    audio_path = uploaded
                    label = f"upload:{Path(uploaded).stem}"
                elif preset and preset in self.available_voices:
                    audio_path = self.available_voices[preset]
                    label = preset
                else:
                    raise gr.Error(
                        f"Choose a preset or upload a reference for Speaker {index}."
                    )
                audio_data = self.read_audio(
                    audio_path,
                    trim_start=trim_starts[index],
                    trim_end=trim_ends[index],
                    normalize=normalize_voices,
                )
                if len(audio_data) == 0:
                    raise gr.Error(f"Failed to load or trim voice reference for Speaker {index}.")
                voice_samples.append(audio_data)
                voice_labels.append(label)

            log = f"🎙️ Generating with {num_speakers} speaker(s)\n"
            log += (
                f"📊 CFG={cfg_scale}, steps={int(inference_steps)}, seed={int(seed)}, "
                f"sampling={bool(do_sample)}\n"
            )
            log += f"🎭 Voices: {', '.join(voice_labels)}\n"
            log += (
                f"🧩 Unlimited mode: {len(script_chunks)} chunk(s), "
                f"targeting {min_chunk_chars}-{max_chunk_chars} characters at periods\n"
            )

            start_time = time.time()
            sample_rate = 24000
            all_audio_chunks = []
            stream_piece_count = 0

            for script_index, script_chunk in enumerate(script_chunks):
                if self.stop_generation:
                    break

                inputs = self.processor(
                    text=[script_chunk],
                    voice_samples=[voice_samples],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                target_device = self.device if self.device in ("cuda", "mps") else "cpu"
                for key, value in inputs.items():
                    if torch.is_tensor(value):
                        inputs[key] = value.to(target_device)

                audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
                self.current_streamer = audio_streamer
                errors = []
                generation_thread = threading.Thread(
                    target=self._generate_with_streamer,
                    args=(
                        inputs,
                        cfg_scale,
                        audio_streamer,
                        int(inference_steps),
                        int(seed) + script_index,
                        bool(do_sample),
                        float(temperature),
                        float(top_p),
                        errors,
                    ),
                    daemon=True,
                )
                generation_thread.start()

                chunk_audio = []
                for audio_chunk in audio_streamer.get_stream(0):
                    if self.stop_generation:
                        audio_streamer.end()
                        break
                    if torch.is_tensor(audio_chunk):
                        audio_chunk = audio_chunk.float().cpu().numpy()
                    audio_16bit = convert_to_16_bit_wav(
                        np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
                    )
                    if not audio_16bit.size:
                        continue
                    chunk_audio.append(audio_16bit)
                    all_audio_chunks.append(audio_16bit)
                    stream_piece_count += 1
                    total_duration = sum(len(part) for part in all_audio_chunks) / sample_rate
                    progress = log + (
                        f"🔄 Text chunk {script_index + 1}/{len(script_chunks)}\n"
                        f"🎵 {total_duration:.1f}s generated ({stream_piece_count} audio pieces)\n"
                    )
                    yield (sample_rate, audio_16bit), None, progress, gr.update(visible=True)

                generation_thread.join(timeout=10.0)
                if generation_thread.is_alive():
                    self.stop_generation = True
                    audio_streamer.end()
                    generation_thread.join(timeout=5.0)
                    raise RuntimeError("Generation worker did not stop cleanly")
                if errors and not self.stop_generation:
                    raise errors[0]
                if not chunk_audio and not self.stop_generation:
                    raise RuntimeError(
                        f"No audio received for text chunk {script_index + 1}/{len(script_chunks)}"
                    )

                if script_index < len(script_chunks) - 1 and chunk_pause_seconds > 0:
                    pause = np.zeros(
                        int(sample_rate * float(chunk_pause_seconds)), dtype=np.int16
                    )
                    all_audio_chunks.append(pause)

            self.current_streamer = None
            self.is_generating = False
            if self.stop_generation:
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            if not all_audio_chunks:
                raise RuntimeError("No audio was generated")

            complete_audio = np.concatenate(all_audio_chunks)
            generation_time = time.time() - start_time
            final_duration = len(complete_audio) / sample_rate
            final_log = log + (
                f"✅ Joined {len(script_chunks)} text chunk(s) in {generation_time:.2f}s\n"
                f"🎵 Final audio duration: {final_duration:.2f}s\n"
                "✨ Complete audio is ready to play or download."
            )
            yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)

        except gr.Error as e:
            # Handle Gradio-specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)
    
    def _generate_with_streamer(
        self,
        inputs,
        cfg_scale,
        audio_streamer,
        inference_steps,
        seed,
        do_sample,
        temperature,
        top_p,
        errors,
    ):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return
                
            self.model.set_ddpm_inference_steps(num_steps=inference_steps)
            set_seed(seed)
            generation_config = {'do_sample': do_sample}
            if do_sample:
                generation_config.update(temperature=temperature, top_p=top_p)
            self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config=generation_config,
                audio_streamer=audio_streamer,
                stop_check_fn=lambda: self.stop_generation,
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
            )
            
        except Exception as e:
            errors.append(e)
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")
    
    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        
        # Check if text_examples directory exists
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return
        
        # Get all .txt files in the text_examples directory
        txt_files = sorted([f for f in os.listdir(examples_dir) 
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(examples_dir, f))])
        
        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)
            
            import re
            # Check if filename contains a time pattern like "45min", "90min", etc.
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    print(f"Skipping {txt_file}: duration {minutes} minutes exceeds 15-minute limit")
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                
                # Remove empty lines and lines with only whitespace
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                
                if not script_content:
                    continue
                
                # Parse the script to determine number of speakers
                num_speakers = self._get_num_speakers_from_script(script_content)
                
                # Add to examples list as [num_speakers, script_content]
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")
                
            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")
        
        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")
    
    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()
        
        lines = script.strip().split('\n')
        for line in lines:
            # Use regex to find speaker patterns
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)
        
        # If no speakers found, default to 1
        if not speakers:
            return 1
        
        # Return the maximum speaker ID + 1 (assuming 0-based indexing)
        # or the count of unique speakers if they're 1-based
        max_speaker = max(speakers)
        min_speaker = min(speakers)
        
        if min_speaker == 0:
            return max_speaker + 1
        else:
            # Assume 1-based indexing, return the count
            return len(speakers)
    

def create_demo_interface(demo_instance: VibeVoiceDemo):
    """Create the Gradio interface with streaming support."""
    
    # Custom CSS for high-end aesthetics with lighter theme
    custom_css = """
    /* Modern light theme with gradients */
    .gradio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .settings-card, .generation-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Speaker selection styling */
    .speaker-grid {
        display: grid;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .speaker-item {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        border: 1px solid rgba(148, 163, 184, 0.4);
        border-radius: 12px;
        padding: 1rem;
        color: #374151;
        font-weight: 500;
    }
    
    /* Streaming indicator */
    .streaming-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Queue status styling */
    .queue-status {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 0.9rem;
        color: #0369a1;
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
        transition: all 0.3s ease;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(5, 150, 105, 0.6);
    }
    
    .stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        transition: all 0.3s ease;
    }
    
    .stop-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(239, 68, 68, 0.6);
    }
    
    /* Audio player styling */
    .audio-output {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .complete-audio-section {
        margin-top: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
    }
    
    /* Text areas */
    .script-input, .log-output {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .script-input::placeholder {
        color: #64748b !important;
    }
    
    /* Sliders */
    .slider-container {
        background: rgba(248, 250, 252, 0.8);
        border: 1px solid rgba(226, 232, 240, 0.6);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Labels and text */
    .gradio-container label {
        color: #374151 !important;
        font-weight: 600 !important;
    }
    
    .gradio-container .markdown {
        color: #1f2937 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .settings-card, .generation-card { padding: 1rem; }
    }
    
    /* Random example button styling - more subtle professional color */
    .random-btn {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(100, 116, 139, 0.3);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .random-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(100, 116, 139, 0.4);
        background: linear-gradient(135deg, #475569 0%, #334155 100%);
    }
    """
    
    with gr.Blocks(
        title="VibeVoice Studio - Unlimited AI Audio",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        )
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ VibeVoice Studio</h1>
            <p>Unlimited long-form, multi-speaker audio with custom voice references</p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1, elem_classes="settings-card"):
                gr.Markdown("### 🎛️ **Podcast Settings**")
                
                # Number of speakers
                num_speakers = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="Number of Speakers",
                    elem_classes="slider-container"
                )
                
                # Speaker selection
                gr.Markdown("### 🎭 **Speaker Selection**")
                
                available_speaker_names = list(demo_instance.available_voices.keys())
                preferred_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']
                default_speakers = [
                    preferred_speakers[i]
                    if preferred_speakers[i] in available_speaker_names
                    else (available_speaker_names[i % len(available_speaker_names)] if available_speaker_names else None)
                    for i in range(4)
                ]

                speaker_selections = []
                speaker_uploads = []
                speaker_trim_starts = []
                speaker_trim_ends = []
                speaker_panels = []
                for i in range(4):
                    default_value = default_speakers[i] if i < len(default_speakers) else None
                    with gr.Column(visible=(i < 2)) as speaker_panel:
                        gr.Markdown(f"#### Speaker {i}")
                        speaker = gr.Dropdown(
                            choices=available_speaker_names,
                            value=default_value,
                            label="Preset voice (used when no upload is supplied)",
                            elem_classes="speaker-item",
                        )
                        upload = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Upload or record a voice reference (overrides preset)",
                        )
                        with gr.Row():
                            trim_start = gr.Number(
                                value=0.0,
                                minimum=0.0,
                                label="Trim start (seconds)",
                            )
                            trim_end = gr.Number(
                                value=0.0,
                                minimum=0.0,
                                label="Trim end (0 = full file)",
                            )
                    speaker_panels.append(speaker_panel)
                    speaker_selections.append(speaker)
                    speaker_uploads.append(upload)
                    speaker_trim_starts.append(trim_start)
                    speaker_trim_ends.append(trim_end)
                
                # Advanced settings
                gr.Markdown("### ⚙️ **Advanced Settings**")
                
                # Sampling parameters (contains all generation settings)
                with gr.Accordion("Generation Parameters", open=False):
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.3,
                        step=0.05,
                        label="CFG Scale (Guidance Strength)",
                        # info="Higher values increase adherence to text",
                        elem_classes="slider-container"
                    )
                    inference_steps = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=demo_instance.inference_steps,
                        step=1,
                        label="Inference Steps",
                    )
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    do_sample = gr.Checkbox(value=False, label="Enable sampling")
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.05, label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.05, maximum=1.0, value=0.95, step=0.05, label="Top P"
                    )
                    normalize_voices = gr.Checkbox(
                        value=True,
                        label="Normalize uploaded/preset voice references",
                    )
                with gr.Accordion("Unlimited Text Chunking", open=False):
                    gr.Markdown(
                        "Text is split at sentence periods, generated sequentially, and joined into one file."
                    )
                    min_chunk_chars = gr.Slider(
                        minimum=1000,
                        maximum=1900,
                        value=1000,
                        step=100,
                        label="Target minimum characters",
                    )
                    max_chunk_chars = gr.Slider(
                        minimum=1100,
                        maximum=2000,
                        value=2000,
                        step=100,
                        label="Hard maximum characters",
                    )
                    chunk_pause_seconds = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.15,
                        step=0.05,
                        label="Pause between joined chunks (seconds)",
                    )
                
            # Right column - Generation
            with gr.Column(scale=2, elem_classes="generation-card"):
                gr.Markdown("### 📝 **Script Input**")
                
                script_input = gr.Textbox(
                    label="Conversation Script",
                    placeholder="""Enter your podcast script here. You can format it as:

Speaker 1: Welcome to our podcast today!
Speaker 2: Thanks for having me. I'm excited to discuss...

Or paste text directly and it will auto-assign speakers.""",
                    lines=12,
                    max_lines=20,
                    elem_classes="script-input"
                )
                
                # Button row with Random Example on the left and Generate on the right
                with gr.Row():
                    # Random example button (now on the left)
                    random_example_btn = gr.Button(
                        "🎲 Random Example",
                        size="lg",
                        variant="secondary",
                        elem_classes="random-btn",
                        scale=1  # Smaller width
                    )
                    
                    # Generate button (now on the right)
                    generate_btn = gr.Button(
                        "🚀 Generate Podcast",
                        size="lg",
                        variant="primary",
                        elem_classes="generate-btn",
                        scale=2  # Wider than random button
                    )
                
                # Stop button
                stop_btn = gr.Button(
                    "🛑 Stop Generation",
                    size="lg",
                    variant="stop",
                    elem_classes="stop-btn",
                    visible=False
                )
                
                # Streaming status indicator
                streaming_status = gr.HTML(
                    value="""
                    <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                border: 1px solid rgba(34, 197, 94, 0.3); 
                                border-radius: 8px; 
                                padding: 0.75rem; 
                                margin: 0.5rem 0;
                                text-align: center;
                                font-size: 0.9rem;
                                color: #166534;">
                        <span class="streaming-indicator"></span>
                        <strong>LIVE STREAMING</strong> - Audio is being generated in real-time
                    </div>
                    """,
                    visible=False,
                    elem_id="streaming-status"
                )
                
                # Output section
                gr.Markdown("### 🎵 **Generated Podcast**")
                
                # Streaming audio output (outside of tabs for simpler handling)
                audio_output = gr.Audio(
                    label="Streaming Audio (Real-time)",
                    type="numpy",
                    elem_classes="audio-output",
                    streaming=True,  # Enable streaming mode
                    autoplay=True,
                    show_download_button=False,  # Explicitly show download button
                    visible=True
                )
                
                # Complete audio output (non-streaming)
                complete_audio_output = gr.Audio(
                    label="Complete Podcast (Download after generation)",
                    type="numpy",
                    elem_classes="audio-output complete-audio-section",
                    streaming=False,  # Non-streaming mode
                    autoplay=False,
                    show_download_button=True,  # Explicitly show download button
                    visible=False  # Initially hidden, shown when audio is ready
                )
                
                gr.Markdown("""
                *💡 **Streaming**: Audio plays as it's being generated (may have slight pauses)  
                *💡 **Complete Audio**: Will appear below after generation finishes*
                """)
                
                # Generation log
                log_output = gr.Textbox(
                    label="Generation Log",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    elem_classes="log-output"
                )
        
        def update_speaker_visibility(num_speakers):
            updates = []
            for i in range(4):
                updates.append(gr.update(visible=(i < num_speakers)))
            return updates
        
        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_panels
        )
        
        # Main generation function with streaming
        def generate_podcast_wrapper(num_speakers, script, *speaker_and_params):
            """Wrapper function to handle the streaming generation call."""
            try:
                presets = list(speaker_and_params[0:4])
                uploads = list(speaker_and_params[4:8])
                trim_starts = list(speaker_and_params[8:12])
                trim_ends = list(speaker_and_params[12:16])
                (
                    cfg_value,
                    steps_value,
                    seed_value,
                    sample_value,
                    temperature_value,
                    top_p_value,
                    normalize_value,
                    min_chars_value,
                    max_chars_value,
                    pause_value,
                ) = speaker_and_params[16:26]
                
                # Clear outputs and reset visibility at start
                yield None, gr.update(value=None, visible=False), "🎙️ Starting generation...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                
                # The generator will yield multiple times
                final_log = "Starting generation..."
                
                for streaming_audio, complete_audio, log, streaming_visible in demo_instance.generate_podcast_streaming(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_presets=presets,
                    uploaded_voices=uploads,
                    trim_starts=trim_starts,
                    trim_ends=trim_ends,
                    cfg_scale=cfg_value,
                    inference_steps=int(steps_value),
                    seed=int(seed_value),
                    do_sample=bool(sample_value),
                    temperature=temperature_value,
                    top_p=top_p_value,
                    min_chunk_chars=int(min_chars_value),
                    max_chunk_chars=int(max_chars_value),
                    chunk_pause_seconds=pause_value,
                    normalize_voices=bool(normalize_value),
                ):
                    final_log = log
                    
                    # Check if we have complete audio (final yield)
                    if complete_audio is not None:
                        # Final state: clear streaming, show complete audio
                        yield None, gr.update(value=complete_audio, visible=True), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    else:
                        # Streaming state: update streaming audio only
                        if streaming_audio is not None:
                            yield streaming_audio, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)
                        else:
                            # No new audio, just update status
                            yield None, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)

            except Exception as e:
                error_msg = f"❌ A critical error occurred in the wrapper: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # Reset button states on error
                yield None, gr.update(value=None, visible=False), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def stop_generation_handler():
            """Handle stopping generation."""
            demo_instance.stop_audio_generation()
            # Return values for: log_output, streaming_status, generate_btn, stop_btn
            return "🛑 Generation stopped.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Add a clear audio function
        def clear_audio_outputs():
            """Clear both audio outputs before starting new generation."""
            return None, gr.update(value=None, visible=False)

        # Connect generation button with streaming outputs
        generate_btn.click(
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        ).then(  # Immediate UI update to hide Generate, show Stop (non-queued)
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=[],
            outputs=[generate_btn, stop_btn],
            queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=(
                [num_speakers, script_input]
                + speaker_selections
                + speaker_uploads
                + speaker_trim_starts
                + speaker_trim_ends
                + [
                    cfg_scale,
                    inference_steps,
                    seed,
                    do_sample,
                    temperature,
                    top_p,
                    normalize_voices,
                    min_chunk_chars,
                    max_chunk_chars,
                    chunk_pause_seconds,
                ]
            ),
            outputs=[audio_output, complete_audio_output, log_output, streaming_status, generate_btn, stop_btn],
            queue=True  # Enable Gradio's built-in queue
        )
        
        # Connect stop button
        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, generate_btn, stop_btn],
            queue=False  # Don't queue stop requests
        ).then(
            # Clear both audio outputs after stopping
            fn=lambda: (None, None),
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        )
        
        # Function to randomly select an example
        def load_random_example():
            """Randomly select and load an example script."""
            import random
            
            # Get available examples
            if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
                example_scripts = demo_instance.example_scripts
            else:
                # Fallback to default
                example_scripts = [
                    [2, "Speaker 0: Welcome to our AI podcast demonstration!\nSpeaker 1: Thanks for having me. This is exciting!"]
                ]
            
            # Randomly select one
            if example_scripts:
                selected = random.choice(example_scripts)
                num_speakers_value = selected[0]
                script_value = selected[1]
                
                # Return the values to update the UI
                return num_speakers_value, script_value
            
            # Default values if no examples
            return 2, ""
        
        # Connect random example button
        random_example_btn.click(
            fn=load_random_example,
            inputs=[],
            outputs=[num_speakers, script_input],
            queue=False  # Don't queue this simple operation
        )
        
        # Add usage tips
        gr.Markdown("""
        ### 💡 **Usage Tips**
        
        - Click **🚀 Generate Podcast** to start audio generation
        - **Live Streaming** tab shows audio as it's generated (may have slight pauses)
        - **Complete Audio** tab provides the full, uninterrupted podcast after generation
        - During generation, you can click **🛑 Stop Generation** to interrupt the process
        - The streaming indicator shows real-time generation progress
        - Upload or record a voice for any speaker; uploaded audio overrides its preset
        - Use trim start/end to isolate the cleanest portion of each voice reference
        - Long scripts are chunked at periods around 1,000-2,000 characters and joined automatically
        """)
        
        # Add example scripts
        gr.Markdown("### 📚 **Example Scripts**")
        
        # Use dynamically loaded examples if available, otherwise provide a default
        if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
            example_scripts = demo_instance.example_scripts
        else:
            # Fallback to a simple default example if no scripts loaded
            example_scripts = [
                [1, "Speaker 1: Welcome to our AI podcast demonstration! This is a sample script showing how VibeVoice can generate natural-sounding speech."]
            ]
        
        gr.Examples(
            examples=example_scripts,
            inputs=[num_speakers, script_input],
            label="Try these example scripts:"
        )

        # --- Risks & limitations (footer) ---
        gr.Markdown(
            """
## Risks and limitations

While efforts have been made to optimize it through various techniques, it may still produce outputs that are unexpected, biased, or inaccurate. VibeVoice inherits any biases, errors, or omissions produced by its base model (specifically, Qwen2.5 1.5b in this release).
Potential for Deepfakes and Disinformation: High-quality synthetic speech can be misused to create convincing fake audio content for impersonation, fraud, or spreading disinformation. Users must ensure transcripts are reliable, check content accuracy, and avoid using generated content in misleading ways. Users are expected to use the generated content and to deploy the models in a lawful manner, in full compliance with all applicable laws and regulations in the relevant jurisdictions. It is best practice to disclose the use of AI when sharing AI-generated content.
            """,
            elem_classes="generation-card",  # 可选：复用卡片样式
        )
    return interface


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/tmp/vibevoice-model",
        help="Path to the VibeVoice model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
        help="Device for inference: cuda | mps | cpu",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=10,
        help="Default number of DDPM inference steps shown in Studio",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the demo publicly via Gradio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on",
    )
    
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    
    set_seed(42)  # Set a fixed seed for reproducibility

    print("🎙️ Initializing VibeVoice Demo with Streaming Support...")
    
    # Initialize demo instance
    demo_instance = VibeVoiceDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps
    )
    
    # Create interface
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 Launching demo on port {args.port}")
    print(f"📁 Model path: {args.model_path}")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    print(f"🔴 Streaming mode: ENABLED")
    print(f"🔒 Session isolation: ENABLED")
    
    # Launch the interface
    try:
        interface.queue(
            max_size=20,  # Maximum queue size
            default_concurrency_limit=1  # Process one request at a time
        ).launch(
            share=args.share,
            server_port=args.port,
            server_name="0.0.0.0" if args.share else "127.0.0.1",
            show_error=True,
            show_api=False  # Hide API docs for cleaner interface
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()
