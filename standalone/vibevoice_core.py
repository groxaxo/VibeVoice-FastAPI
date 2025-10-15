"""
Standalone VibeVoice Core Module
Adapted from ComfyUI VibeVoice nodes - independent of ComfyUI
"""

import logging
import os
import sys
import tempfile
import torch
import numpy as np
import re
import gc
import json
from typing import List, Optional, Tuple, Any, Dict
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VibeVoice")

def get_optimal_device():
    """Get the best available device (cuda, mps, or cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_device_map():
    """Get device map for model loading"""
    device = get_optimal_device()
    return device if device != "mps" else "mps"

class VibeVoiceCore:
    """Standalone VibeVoice TTS Core"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.processor = None
        self.current_model = None
        self.current_attention_type = None
        self.current_quantize_llm = "full precision"
        self.current_lora_path = None
        
    def free_memory(self):
        """Free model and processor from memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            self.current_model = None
            self.current_quantize_llm = "full precision"
            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Model and processor memory freed successfully")
            
        except Exception as e:
            logger.error(f"Error freeing memory: {e}")
    
    def get_available_models(self) -> List[str]:
        """Scan models directory and return available models"""
        models = []
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory does not exist: {self.models_dir}")
            return models
            
        for item in self.models_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != 'loras' and item.name != 'tokenizer':
                # Check if it's a valid model folder
                if (item / "config.json").exists():
                    models.append(item.name)
        
        models.sort()
        logger.info(f"Found {len(models)} VibeVoice model(s)")
        return models
    
    def find_tokenizer_path(self) -> Optional[Path]:
        """Find Qwen tokenizer using priority system"""
        # Priority 1: Check tokenizer folder
        tokenizer_dir = self.models_dir / "tokenizer"
        if tokenizer_dir.exists():
            required_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
            if all((tokenizer_dir / f).exists() for f in required_files):
                logger.info(f"Found Qwen tokenizer in: {tokenizer_dir}")
                return tokenizer_dir
        
        # Priority 2: Check models--Qwen--Qwen2.5-1.5B folder
        qwen_model_dir = self.models_dir / "models--Qwen--Qwen2.5-1.5B"
        if qwen_model_dir.exists():
            snapshots_dir = qwen_model_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir() and (snapshot / "tokenizer_config.json").exists():
                        logger.info(f"Found Qwen tokenizer in model cache: {snapshot}")
                        return snapshot
        
        # Priority 3: Check HuggingFace cache
        hf_cache_paths = [
            Path.home() / ".cache/huggingface/hub",
        ]
        
        if "HF_HOME" in os.environ:
            hf_cache_paths.insert(0, Path(os.environ["HF_HOME"]) / "hub")
        
        for cache_path in hf_cache_paths:
            if cache_path.exists():
                qwen_cache = cache_path / "models--Qwen--Qwen2.5-1.5B"
                if qwen_cache.exists():
                    snapshots_dir = qwen_cache / "snapshots"
                    if snapshots_dir.exists():
                        for snapshot in snapshots_dir.iterdir():
                            if snapshot.is_dir() and (snapshot / "tokenizer_config.json").exists():
                                logger.info(f"Found Qwen tokenizer in HF cache: {snapshot}")
                                return snapshot
        
        return None
    
    def load_model(self, model_name: str, attention_type: str = "auto", 
                   quantize_llm: str = "full precision", lora_path: Optional[str] = None):
        """Load VibeVoice model"""
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            raise Exception(f"Model not found: {model_path}")
        
        # Check if we need to reload
        lora_changed = (self.current_lora_path or "") != (lora_path or "")
        quantize_changed = self.current_quantize_llm != quantize_llm
        
        if (self.model is None or
            self.current_model != model_name or
            self.current_attention_type != attention_type or
            quantize_changed or lora_changed):
            
            if self.model is not None:
                logger.info(f"Freeing existing model before loading with new settings")
                self.free_memory()
            
            try:
                # Add vvembed to path
                vvembed_path = Path(__file__).parent.parent / 'vvembed'
                if str(vvembed_path) not in sys.path:
                    sys.path.insert(0, str(vvembed_path))
                
                # Import from embedded version
                from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                from processor.vibevoice_processor import VibeVoiceProcessor
                
                logger.info(f"Using embedded VibeVoice from {vvembed_path}")
                
                import time
                start_time = time.time()
                
                # Suppress verbose logs
                import transformers
                import warnings
                transformers.logging.set_verbosity_error()
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Prepare model kwargs
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "device_map": get_device_map(),
                    "local_files_only": True,
                }
                
                # Handle quantization
                if quantize_llm == "4bit":
                    if not torch.cuda.is_available():
                        raise Exception("LLM quantization requires a CUDA GPU")
                    
                    from transformers import BitsAndBytesConfig
                    logger.info("Quantizing LLM component to 4-bit...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4',
                        llm_int8_skip_modules=["lm_head", "prediction_head", "acoustic_connector", "semantic_connector", "diffusion_head"]
                    )
                    model_kwargs["quantization_config"] = bnb_config
                    model_kwargs["device_map"] = "auto"
                
                elif quantize_llm == "8bit":
                    if not torch.cuda.is_available():
                        raise Exception("LLM quantization requires a CUDA GPU")
                    
                    from transformers import BitsAndBytesConfig
                    logger.info("Quantizing LLM component to 8-bit...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.bfloat16,
                        llm_int8_skip_modules=[
                            "lm_head", "prediction_head", "acoustic_connector",
                            "semantic_connector", "acoustic_tokenizer", "semantic_tokenizer"
                        ],
                        llm_int8_threshold=3.0,
                        llm_int8_has_fp16_weight=False,
                    )
                    model_kwargs["quantization_config"] = bnb_config
                    model_kwargs["device_map"] = "auto"
                
                # Set attention implementation
                if attention_type != "auto":
                    model_kwargs["attn_implementation"] = attention_type
                    logger.info(f"Using {attention_type} attention implementation")
                else:
                    logger.info("Using auto attention implementation selection")
                
                # Load model
                logger.info(f"Loading model from: {model_path}")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    str(model_path),
                    **model_kwargs
                )
                
                elapsed = time.time() - start_time
                logger.info(f"Model loaded in {elapsed:.2f} seconds")
                
                # Load processor
                logger.info("Loading VibeVoice processor...")
                tokenizer_path = self.find_tokenizer_path()
                if not tokenizer_path:
                    raise Exception(
                        "Qwen tokenizer not found. Please download it manually.\\n"
                        "Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main\\n"
                        "Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json\\n"
                        f"Place files in: {self.models_dir / 'tokenizer'}/"
                    )
                
                processor_kwargs = {
                    "trust_remote_code": True,
                    "local_files_only": True,
                    "language_model_pretrained_name": str(tokenizer_path),
                }
                
                self.processor = VibeVoiceProcessor.from_pretrained(
                    str(model_path),
                    **processor_kwargs
                )
                
                # Move to device if needed
                is_llm_quantized = quantize_llm != "full precision"
                if not is_llm_quantized:
                    device = get_optimal_device()
                    if device == "cuda":
                        self.model = self.model.cuda()
                    elif device == "mps":
                        self.model = self.model.to("mps")
                
                self.current_model = model_name
                self.current_attention_type = attention_type
                self.current_quantize_llm = quantize_llm
                self.current_lora_path = lora_path
                
                logger.info("Model and processor loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load VibeVoice model: {str(e)}")
                raise Exception(f"Model loading failed: {str(e)}")
    
    def _create_synthetic_voice_sample(self, speaker_idx: int) -> np.ndarray:
        """Create synthetic voice sample for a specific speaker"""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples, False)
        
        base_frequencies = [120, 180, 140, 200]
        base_freq = base_frequencies[speaker_idx % len(base_frequencies)]
        
        formant1 = 800 + speaker_idx * 100
        formant2 = 1200 + speaker_idx * 150
        
        voice_sample = (
            0.6 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * base_freq * 3 * t) +
            0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2) +
            0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3) +
            0.02 * np.random.normal(0, 1, len(t))
        )
        
        vibrato_freq = 4 + speaker_idx * 0.3
        envelope = (np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t)))
        voice_sample *= envelope * 0.08
        
        return voice_sample.astype(np.float32)
    
    def _prepare_audio_from_file(self, audio_path: str, target_sample_rate: int = 24000,
                                  speed_factor: float = 1.0) -> Optional[np.ndarray]:
        """Prepare audio from file path"""
        if not audio_path or not os.path.exists(audio_path):
            return None
        
        try:
            import librosa
            
            # Load audio
            audio_np, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
            
            # Ensure audio is in correct range [-1, 1]
            audio_max = np.abs(audio_np).max()
            if audio_max > 0:
                audio_np = audio_np / max(audio_max, 1.0)
            
            # Apply speed adjustment if requested
            if speed_factor != 1.0:
                original_length = len(audio_np)
                target_length = int(original_length / speed_factor)
                original_indices = np.arange(original_length)
                target_indices = np.linspace(0, original_length - 1, target_length)
                audio_np = np.interp(target_indices, original_indices, audio_np)
                logger.info(f"Adjusted voice speed by factor {speed_factor:.2f}")
            
            return audio_np.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error loading audio from {audio_path}: {e}")
            return None
    
    def _parse_pause_keywords(self, text: str) -> List[Tuple[str, Any]]:
        """Parse [pause] and [pause:ms] keywords from text"""
        segments = []
        pattern = r'\[pause(?::(\d+))?\]'
        
        last_end = 0
        for match in re.finditer(pattern, text):
            if match.start() > last_end:
                text_segment = text[last_end:match.start()].strip()
                if text_segment:
                    segments.append(('text', text_segment))
            
            duration_ms = int(match.group(1)) if match.group(1) else 1000
            segments.append(('pause', duration_ms))
            last_end = match.end()
        
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                segments.append(('text', remaining_text))
        
        if not segments:
            segments.append(('text', text))
        
        return segments
    
    def _generate_silence(self, duration_ms: int, sample_rate: int = 24000) -> torch.Tensor:
        """Generate silence audio tensor"""
        num_samples = int(sample_rate * duration_ms / 1000.0)
        silence_waveform = torch.zeros(1, 1, num_samples, dtype=torch.float32)
        return silence_waveform
    
    def _split_text_into_chunks(self, text: str, max_words: int = 250) -> List[str]:
        """Split long text into manageable chunks at sentence boundaries
        
        Args:
            text: The text to split
            max_words: Maximum words per chunk (default 250 for safety)
        
        Returns:
            List of text chunks
        """
        # Split into sentences (handling common abbreviations)
        # This regex tries to split on sentence endings while avoiding common abbreviations
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # If regex split didn't work well, fall back to simple split
        if len(sentences) == 1 and len(text.split()) > max_words:
            # Fall back to splitting on any period followed by space
            sentences = text.replace('. ', '.|').split('|')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If single sentence is too long, split it further
            if sentence_word_count > max_words:
                # Split long sentence at commas or semicolons
                sub_parts = re.split(r'[,;]', sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part:
                        continue
                    part_words = part.split()
                    part_word_count = len(part_words)
                    
                    if current_word_count + part_word_count > max_words and current_chunk:
                        # Save current chunk
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [part]
                        current_word_count = part_word_count
                    else:
                        current_chunk.append(part)
                        current_word_count += part_word_count
            else:
                # Check if adding this sentence would exceed the limit
                if current_word_count + sentence_word_count > max_words and current_chunk:
                    # Save current chunk and start new one
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # If no chunks were created, return the original text
        if not chunks:
            chunks = [text]
        
        logger.info(f"Split text into {len(chunks)} chunks (max {max_words} words each)")
        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            logger.debug(f"Chunk {i+1}: {word_count} words")
        
        return chunks
    
    def generate_speech(self, text: str, voice_samples: Optional[List[np.ndarray]] = None,
                       cfg_scale: float = 1.3, seed: int = 42, diffusion_steps: int = 20,
                       use_sampling: bool = False, temperature: float = 0.95, top_p: float = 0.95,
                       num_speakers: int = 1, max_words_per_chunk: int = 250) -> Tuple[np.ndarray, int]:
        """Generate speech from text with automatic chunking for long texts
        
        Args:
            text: Input text to synthesize
            voice_samples: Optional voice samples for cloning
            cfg_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            diffusion_steps: Number of diffusion steps
            use_sampling: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            num_speakers: Number of speakers
            max_words_per_chunk: Maximum words per chunk (default 250)
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self.model is None or self.processor is None:
            raise Exception("Model not loaded. Call load_model() first.")
        
        # Parse pause keywords
        segments = self._parse_pause_keywords(text)
        
        # Prepare voice samples
        if voice_samples is None:
            voice_samples = [self._create_synthetic_voice_sample(i) for i in range(num_speakers)]
        
        sample_rate = 24000
        all_audio_segments = []
        
        for seg_idx, (seg_type, seg_content) in enumerate(segments):
            if seg_type == 'pause':
                duration_ms = seg_content
                logger.info(f"Adding {duration_ms}ms pause")
                silence = self._generate_silence(duration_ms, sample_rate)
                all_audio_segments.append(silence)
            else:
                # Check if text needs to be chunked
                word_count = len(seg_content.split())
                
                if word_count > max_words_per_chunk:
                    # Split long text into chunks
                    logger.info(f"Text segment {seg_idx+1} has {word_count} words, splitting into chunks...")
                    text_chunks = self._split_text_into_chunks(seg_content, max_words_per_chunk)
                    
                    for chunk_idx, chunk in enumerate(text_chunks):
                        logger.info(f"Processing chunk {chunk_idx+1}/{len(text_chunks)} of segment {seg_idx+1}...")
                        chunk_audio = self._generate_chunk(
                            chunk, voice_samples, cfg_scale, seed, diffusion_steps,
                            use_sampling, temperature, top_p, num_speakers
                        )
                        all_audio_segments.append(chunk_audio)
                else:
                    # Process as single chunk
                    logger.info(f"Processing text segment {seg_idx+1} ({word_count} words)")
                    chunk_audio = self._generate_chunk(
                        seg_content, voice_samples, cfg_scale, seed, diffusion_steps,
                        use_sampling, temperature, top_p, num_speakers
                    )
                    all_audio_segments.append(chunk_audio)
        
        # Concatenate all segments
        if all_audio_segments:
            combined_waveform = torch.cat(all_audio_segments, dim=-1)
            # Convert to numpy (1, 1, samples) -> (samples,)
            audio_array = combined_waveform.squeeze().numpy()
            return audio_array, sample_rate
        else:
            raise Exception("No audio segments generated")
    
    def _generate_chunk(self, text: str, voice_samples: List[np.ndarray],
                       cfg_scale: float, seed: int, diffusion_steps: int,
                       use_sampling: bool, temperature: float, top_p: float,
                       num_speakers: int) -> torch.Tensor:
        """Generate audio for a single text chunk"""
        # Set seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # Set diffusion steps
        self.model.set_ddpm_inference_steps(diffusion_steps)
        
        # Format text
        formatted_text = text
        if num_speakers == 1:
            formatted_text = f"Speaker 1: {text}"
        
        # Prepare inputs
        inputs = self.processor(
            [formatted_text],
            voice_samples=[voice_samples],
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        logger.info(f"Generating audio with {diffusion_steps} diffusion steps...")
        
        # Generate
        with torch.no_grad():
            if use_sampling:
                output = self.model.generate(
                    **inputs,
                    tokenizer=self.processor.tokenizer,
                    cfg_scale=cfg_scale,
                    max_new_tokens=None,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                output = self.model.generate(
                    **inputs,
                    tokenizer=self.processor.tokenizer,
                    cfg_scale=cfg_scale,
                    max_new_tokens=None,
                    do_sample=False,
                )
            
            if hasattr(output, 'speech_outputs') and output.speech_outputs:
                speech_tensors = output.speech_outputs
                
                if isinstance(speech_tensors, list) and len(speech_tensors) > 0:
                    audio_tensor = torch.cat(speech_tensors, dim=-1)
                else:
                    audio_tensor = speech_tensors
                
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                elif audio_tensor.dim() == 2:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                return audio_tensor.cpu().float()
            else:
                raise Exception("VibeVoice failed to generate audio")
