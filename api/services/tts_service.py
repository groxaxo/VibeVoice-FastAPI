"""Core TTS generation service wrapping VibeVoice model."""

import torch
import numpy as np
from typing import Iterator, List, Optional, Union
from transformers import set_seed
import logging

logger = logging.getLogger(__name__)

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

from api.config import Settings


class TTSService:
    """Service for TTS generation using VibeVoice model."""

    def __init__(self, settings: Settings):
        """
        Initialize TTS service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        self._model_loaded = False

    def load_model(self):
        """Load VibeVoice model and processor."""
        if self._model_loaded:
            print("Model already loaded")
            return

        print(f"Loading VibeVoice model from {self.settings.vibevoice_model_path}")

        # Get device and dtype
        self.device = self.settings.get_device()
        self.dtype = self.settings.get_dtype()
        attn_implementation = self.settings.get_attn_implementation()

        print(
            f"Using device: {self.device}, dtype: {self.dtype}, attention: {attn_implementation}"
        )
        is_cuda_device = str(self.device).startswith("cuda")
        cuda_device_map = "cuda" if self.device == "cuda" else {"": str(self.device)}

        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(
            self.settings.vibevoice_model_path
        )

        # Determine if we should load to CPU first for quantization
        # This avoids loading full precision model to GPU then quantizing (wastes VRAM)
        # User requested to avoid CPU loading, so force direct GPU loading
        load_to_cpu_first = False

        if load_to_cpu_first:
            print("Loading model to CPU first for quantization (saves GPU memory)...")
            # Use sdpa for CPU loading since flash_attention_2 requires CUDA
            cpu_attn = (
                "sdpa"
                if attn_implementation == "flash_attention_2"
                else attn_implementation
            )
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.settings.vibevoice_model_path,
                torch_dtype=self.dtype,
                device_map="cpu",
                attn_implementation=cpu_attn,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            # Apply quantization on CPU
            self._apply_quantization()

            # Now move to CUDA
            print("Moving quantized model to CUDA...")
            # Use current device (respects CUDA_VISIBLE_DEVICES)
            self.model = self.model.to(self.device)

            # Log final VRAM usage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_final = torch.cuda.memory_allocated() / 1024**3
                logger.info(
                    f"Final VRAM usage after moving to GPU: {vram_final:.2f} GB"
                )
        else:
            # Standard loading path (no quantization or non-CUDA device)
            try:
                if self.device == "mps":
                    self.model = (
                        VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            attn_implementation=attn_implementation,
                            device_map=None,
                        )
                    )
                    self.model.to("mps")
                elif is_cuda_device:
                    self.model = (
                        VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            device_map=cuda_device_map,
                            attn_implementation=attn_implementation,
                        )
                    )
                else:
                    self.model = (
                        VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            device_map="cpu",
                            attn_implementation=attn_implementation,
                        )
                    )
            except Exception as e:
                if attn_implementation == "flash_attention_2":
                    print(f"Flash attention failed: {e}")
                    print("Falling back to SDPA attention")
                    attn_implementation = "sdpa"

                    if self.device == "mps":
                        self.model = (
                            VibeVoiceForConditionalGenerationInference.from_pretrained(
                                self.settings.vibevoice_model_path,
                                torch_dtype=self.dtype,
                                attn_implementation=attn_implementation,
                                device_map=None,
                            )
                        )
                        self.model.to("mps")
                    elif is_cuda_device:
                        self.model = (
                            VibeVoiceForConditionalGenerationInference.from_pretrained(
                                self.settings.vibevoice_model_path,
                                torch_dtype=self.dtype,
                                device_map=cuda_device_map,
                                attn_implementation=attn_implementation,
                            )
                        )
                    else:
                        self.model = (
                            VibeVoiceForConditionalGenerationInference.from_pretrained(
                                self.settings.vibevoice_model_path,
                                torch_dtype=self.dtype,
                                device_map="cpu",
                                attn_implementation=attn_implementation,
                            )
                        )
                else:
                    raise e

            self.model.eval()

            # Apply quantization if requested (even if loaded directly to GPU)
            if self.settings.vibevoice_quantization:
                print(
                    f"Applying {self.settings.vibevoice_quantization} quantization on {self.device}..."
                )
                self._apply_quantization()

        # Apply torch.compile for optimized inference
        if self.settings.torch_compile:
            try:
                compile_mode = self.settings.torch_compile_mode
                self.model = torch.compile(self.model, mode=compile_mode, dynamic=True)
                print(
                    f"Model compiled with torch.compile(mode='{compile_mode}', dynamic=True)"
                )
            except Exception as e:
                print(f"torch.compile() failed: {e}, continuing without compilation")

        # Configure noise scheduler
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )

        # Set inference steps
        self.model.set_ddpm_inference_steps(
            num_steps=self.settings.vibevoice_inference_steps
        )

        self._model_loaded = True
        print("Model loaded successfully")

    def _apply_quantization(self):
        """Apply quantization to the model based on settings."""
        quant_method = self.settings.vibevoice_quantization

        if quant_method == "int8_torchao":
            self._apply_torchao_quant(bits=8)
        elif quant_method == "int4_torchao":
            self._apply_torchao_quant(bits=4)
        elif quant_method == "nf4_bnb":
            self._apply_bnb_nf4_quant()
        else:
            logger.warning(
                f"Unknown quantization method: {quant_method}, skipping quantization"
            )

    def _apply_torchao_quant(self, bits: int = 8):
        """
        Apply torchao weight-only quantization to the language model.

        This selectively quantizes only the LLM (Qwen2) decoder and lm_head,
        keeping audio components (tokenizers, diffusion head, connectors) at full precision.

        Args:
            bits: 8 for INT8 (~40% VRAM reduction) or 4 for INT4 (~60% VRAM reduction, faster)
        """
        try:
            if bits == 4:
                # Use int8 dynamic activation + int4 weight (more compatible than pure int4)
                from torchao.quantization import (
                    quantize_,
                    int8_dynamic_activation_int4_weight,
                )

                quant_fn = int8_dynamic_activation_int4_weight()
                quant_name = "INT8-ACT-INT4-WEIGHT"
            else:
                from torchao.quantization import quantize_, int8_weight_only

                quant_fn = int8_weight_only()
                quant_name = "INT8"
        except ImportError:
            logger.error(
                "torchao not installed. Install with: pip install torchao\n"
                "Falling back to full precision."
            )
            return

        # Check if model is on CUDA (for memory logging)
        model_on_cuda = next(self.model.parameters()).is_cuda

        logger.info(f"Applying torchao {quant_name} quantization...")
        if model_on_cuda:
            logger.info("Model is on CUDA - quantizing in place")
        else:
            logger.info(
                "Model is on CPU - quantizing before moving to GPU (saves VRAM)"
            )

        # Quantize only the language model (Qwen2 decoder) - this is the largest component
        # The audio components (acoustic_tokenizer, semantic_tokenizer, prediction_head, connectors)
        # are kept at full precision to maintain audio quality
        try:
            logger.info(
                f"Quantizing language_model (Qwen2 decoder) with {quant_name}..."
            )
            quantize_(self.model.model.language_model, quant_fn)

            logger.info(f"Quantizing lm_head with {quant_name}...")
            quantize_(self.model.lm_head, quant_fn)

        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
            logger.info("Continuing with full precision model")
            return

        logger.info(f"{quant_name} quantization applied successfully")

        # Force garbage collection
        import gc

        gc.collect()

    def _apply_bnb_nf4_quant(self):
        """
        Apply BitsAndBytes NF4 quantization to the language model.

        NF4 (4-bit NormalFloat) provides ~75% VRAM reduction with minimal quality loss.
        This is the quantization method used in QLoRA and provides excellent results.
        Requires bitsandbytes library.
        """
        try:
            import bitsandbytes as bnb
        except ImportError:
            logger.error(
                "bitsandbytes not installed. Install with: pip install bitsandbytes\n"
                "Falling back to full precision."
            )
            return

        # Check if model is on CUDA (for memory logging)
        model_on_cuda = next(self.model.parameters()).is_cuda

        logger.info("Applying BitsAndBytes NF4 quantization...")
        if model_on_cuda:
            logger.info("Model is on CUDA - quantizing in place")
        else:
            logger.info(
                "Model is on CPU - quantizing before moving to GPU (saves VRAM)"
            )

        # Quantize Linear layers in language model and lm_head
        def replace_with_bnb_linear(module, name=""):
            for attr_name in dir(module):
                target_attr = getattr(module, attr_name)
                if isinstance(target_attr, torch.nn.Linear):
                    # Create NF4 quantized linear layer
                    setattr(
                        module,
                        attr_name,
                        bnb.nn.Linear4bit(
                            target_attr.in_features,
                            target_attr.out_features,
                            bias=target_attr.bias is not None,
                            compute_dtype=torch.bfloat16,
                            compress_statistics=True,
                            quant_type="nf4",
                        ),
                    )
                    # Copy weights
                    new_layer = getattr(module, attr_name)
                    new_layer.weight.data = target_attr.weight.data
                    if target_attr.bias is not None:
                        new_layer.bias.data = target_attr.bias.data

            # Recurse into child modules
            for name, child in module.named_children():
                replace_with_bnb_linear(child, name)

        try:
            logger.info("Quantizing language_model (Qwen2 decoder) with NF4...")
            replace_with_bnb_linear(self.model.model.language_model)

            logger.info("Quantizing lm_head with NF4...")
            replace_with_bnb_linear(self.model.lm_head)

        except Exception as e:
            logger.error(f"Failed to quantize model with NF4: {e}")
            logger.info("Continuing with full precision model")
            return

        logger.info("NF4 quantization applied successfully")

        # Force garbage collection
        import gc

        gc.collect()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def generate_speech(
        self,
        text: str,
        voice_samples: List[np.ndarray],
        cfg_scale: float = 1.3,
        inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        stream: bool = False,
    ) -> Union[np.ndarray, Iterator[np.ndarray]]:
        """
        Generate speech from text.

        Args:
            text: Input text (formatted with Speaker labels)
            voice_samples: List of voice sample arrays
            cfg_scale: Classifier-free guidance scale
            inference_steps: Number of diffusion steps (None = use default)
            seed: Random seed for reproducibility
            stream: Whether to return streaming iterator

        Returns:
            Generated audio array or iterator of audio chunks
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Set seed if provided
        if seed is not None:
            set_seed(seed)

        # Set inference steps if provided
        if inference_steps is not None:
            self.model.set_ddpm_inference_steps(num_steps=inference_steps)

        # Process inputs
        inputs = self.processor(
            text=[text],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        target_device = (
            self.device
            if str(self.device).startswith("cuda") or self.device == "mps"
            else "cpu"
        )
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)

        if stream:
            # Return streaming iterator
            return self._generate_streaming(inputs, cfg_scale)
        else:
            # Generate all at once
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    return_speech=True,
                    verbose=False,
                    refresh_negative=True,
                    show_progress_bar=False,
                )

            # Get audio output
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                audio = outputs.speech_outputs[0]
                if torch.is_tensor(audio):
                    # Convert bfloat16 to float32 before converting to numpy
                    if audio.dtype == torch.bfloat16:
                        audio = audio.float()
                    audio = audio.cpu().numpy()
                return audio
            else:
                raise RuntimeError("No audio generated")

    def _generate_streaming(
        self, inputs: dict, cfg_scale: float
    ) -> Iterator[np.ndarray]:
        """
        Generate speech with streaming.

        Args:
            inputs: Processed model inputs
            cfg_scale: CFG scale

        Yields:
            Audio chunks as numpy arrays
        """
        # Create audio streamer
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)

        # Start generation in background
        import threading

        def generate():
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    audio_streamer=audio_streamer,
                    return_speech=True,
                    verbose=False,
                    refresh_negative=True,
                    show_progress_bar=False,
                )

        generation_thread = threading.Thread(target=generate)
        generation_thread.start()

        # Yield chunks as they arrive
        audio_stream = audio_streamer.get_stream(0)
        for chunk in audio_stream:
            if torch.is_tensor(chunk):
                # Convert bfloat16 to float32 before converting to numpy
                if chunk.dtype == torch.bfloat16:
                    chunk = chunk.float()
                chunk = chunk.cpu().numpy()
            yield chunk

        # Wait for generation to complete
        generation_thread.join(timeout=10.0)

    def format_script_for_single_speaker(self, text: str, speaker_id: int = 0) -> str:
        """
        Format plain text as single-speaker script.

        Args:
            text: Plain text input
            speaker_id: Speaker ID to use

        Returns:
            Formatted script
        """
        # Split into sentences/paragraphs
        lines = text.strip().split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(f"Speaker {speaker_id}: {line}")

        return "\n".join(formatted_lines)
