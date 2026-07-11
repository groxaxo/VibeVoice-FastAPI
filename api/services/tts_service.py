"""Core TTS generation service wrapping the VibeVoice model."""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator, List, Optional, Union

import numpy as np
import torch
from transformers import set_seed

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.modular.streamer import AudioStreamer
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from api.config import Settings
from api.utils.audio_utils import trim_trailing_silence
from api.utils.text_sanitizer import is_speakable, sanitize_text

logger = logging.getLogger(__name__)

AudioResult = Union[np.ndarray, Iterator[np.ndarray], None]


class TTSService:
    """Thread-safe lifecycle and inference wrapper for VibeVoice.

    VibeVoice's scheduler stores mutable step state on the model, so all model
    generation and per-request scheduler/RNG changes must be serialized.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.processor = None
        self.device: Optional[str] = None
        self.dtype = None
        self._model_loaded = False

        # Re-entrant because lazy generation may call load_model while already
        # holding the inference lock. The same lock also protects unloads.
        self._generate_lock = threading.RLock()
        self._load_lock = threading.RLock()

        self._lazy_load = bool(settings.vibevoice_lazy_load)
        self._idle_timeout_seconds = int(settings.vibevoice_idle_timeout_seconds)
        self._idle_timer: Optional[threading.Timer] = None
        self._idle_timer_lock = threading.Lock()

        self._request_count = 0
        self._request_count_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def is_busy(self) -> bool:
        with self._request_count_lock:
            return self._request_count > 0

    def _request_started(self) -> None:
        with self._request_count_lock:
            self._request_count += 1
        self._cancel_idle_timer()

    def _request_finished(self) -> None:
        with self._request_count_lock:
            self._request_count = max(0, self._request_count - 1)
            idle = self._request_count == 0
        if idle:
            self._start_idle_timer()

    @contextmanager
    def request_context(self):
        """Keep the model busy for a whole multi-chunk API request.

        Individual chunk generations still track themselves, but the outer count
        prevents idle-timer creation and unload races between adjacent chunks.
        """
        self._request_started()
        try:
            yield
        finally:
            self._request_finished()

    def load_model(self) -> None:
        """Load the model exactly once, safely under concurrent first requests."""
        with self._generate_lock, self._load_lock:
            if self._model_loaded:
                self._cancel_idle_timer()
                return

            self._cancel_idle_timer()
            self.device = self.settings.get_device()
            self.dtype = self.settings.get_dtype()
            attention = self.settings.get_attn_implementation()

            logger.info(
                "Loading VibeVoice model from %s (device=%s, dtype=%s, attention=%s)",
                self.settings.vibevoice_model_path,
                self.device,
                self.dtype,
                attention,
            )

            try:
                self.processor = VibeVoiceProcessor.from_pretrained(
                    self.settings.vibevoice_model_path
                )
                unified_quantized = self._detect_unified_quantization(
                    self.settings.vibevoice_model_path
                )
                if unified_quantized:
                    logger.info(
                        "Detected pre-quantized %s checkpoint; using embedded quantization config",
                        unified_quantized,
                    )

                load_to_cpu_first = bool(
                    self.settings.vibevoice_quantization
                    and not unified_quantized
                    and self.device.startswith("cuda")
                )

                if load_to_cpu_first:
                    cpu_attention = "sdpa" if attention == "flash_attention_2" else attention
                    self.model = self._from_pretrained(
                        device="cpu",
                        attention=cpu_attention,
                    )
                    self.model.eval()
                    self._apply_quantization()
                    logger.info("Moving runtime-quantized model to %s", self.device)
                    self.model = self.model.to(self.device)
                else:
                    try:
                        self.model = self._from_pretrained(
                            device=self.device,
                            attention=attention,
                        )
                    except Exception:
                        if attention != "flash_attention_2":
                            raise
                        logger.exception(
                            "Flash Attention model load failed; retrying with SDPA"
                        )
                        self._drop_model_references(clear_processor=False)
                        self._clear_cuda_cache()
                        self.model = self._from_pretrained(
                            device=self.device,
                            attention="sdpa",
                        )
                    self.model.eval()

                # Configure scheduler before optional compilation. Per-request step
                # overrides are still applied inside the inference lock.
                self.model.model.noise_scheduler = (
                    self.model.model.noise_scheduler.from_config(
                        self.model.model.noise_scheduler.config,
                        algorithm_type="sde-dpmsolver++",
                        beta_schedule="squaredcos_cap_v2",
                    )
                )
                self.model.set_ddpm_inference_steps(
                    num_steps=self.settings.vibevoice_inference_steps
                )

                if self.settings.torch_compile:
                    try:
                        self.model = torch.compile(
                            self.model,
                            mode=self.settings.torch_compile_mode,
                            dynamic=True,
                        )
                        logger.info(
                            "Enabled torch.compile(mode=%s, dynamic=True)",
                            self.settings.torch_compile_mode,
                        )
                    except Exception:
                        logger.exception("torch.compile failed; continuing eagerly")

                self._model_loaded = True
                logger.info("VibeVoice model loaded successfully")
            except Exception:
                self._drop_model_references(clear_processor=True)
                self._model_loaded = False
                self._clear_cuda_cache()
                raise

    def _from_pretrained(self, device: str, attention: str):
        kwargs = {
            "torch_dtype": self.dtype,
            "attn_implementation": attention,
            "low_cpu_mem_usage": True,
        }
        if device.startswith("cuda"):
            kwargs["device_map"] = {"": device}
            return VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.settings.vibevoice_model_path,
                **kwargs,
            )

        kwargs["device_map"] = "cpu"
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.settings.vibevoice_model_path,
            **kwargs,
        )
        if device == "mps":
            model = model.to("mps")
        return model

    def _drop_model_references(self, *, clear_processor: bool) -> None:
        self.model = None
        if clear_processor:
            self.processor = None
        gc.collect()

    def unload_model(self, *, blocking: bool = True, reason: str = "manual request") -> bool:
        """Release model memory without racing active or queued inference.

        Returns ``False`` only when ``blocking=False`` and inference currently owns
        the lock. Idle timers use the non-blocking mode and retry later.
        """
        if not blocking and self.is_busy:
            return False

        acquired = self._generate_lock.acquire(blocking=blocking)
        if not acquired:
            return False

        try:
            with self._load_lock:
                self._cancel_idle_timer()
                if not self._model_loaded:
                    return True

                logger.info("Unloading VibeVoice model (%s)", reason)
                try:
                    if (
                        self.model is not None
                        and self.device
                        and self.device.startswith("cuda")
                        and torch.cuda.is_available()
                    ):
                        self.model.to("cpu")
                except Exception:
                    logger.exception("Could not move model to CPU before unload")

                self._drop_model_references(clear_processor=True)
                self._model_loaded = False
                self._clear_cuda_cache()
                return True
        finally:
            self._generate_lock.release()

    def _clear_cuda_cache(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            logger.exception("CUDA cache cleanup failed")

    def _ensure_loaded_locked(self) -> None:
        if self._model_loaded:
            return
        if not self._lazy_load:
            raise RuntimeError("Model is not loaded; call /v1/vibevoice/preload first")
        logger.info("Lazy-loading VibeVoice for the first request")
        self.load_model()

    def _start_idle_timer(self) -> None:
        if not self._lazy_load or self._idle_timeout_seconds <= 0 or self.is_busy:
            return
        with self._idle_timer_lock:
            self._cancel_idle_timer_unlocked()
            timer = threading.Timer(self._idle_timeout_seconds, self._on_idle_timeout)
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def _cancel_idle_timer(self) -> None:
        with self._idle_timer_lock:
            self._cancel_idle_timer_unlocked()

    def _cancel_idle_timer_unlocked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _on_idle_timeout(self) -> None:
        with self._idle_timer_lock:
            self._idle_timer = None

        if not self._model_loaded:
            return
        if self.is_busy or not self.unload_model(
            blocking=False, reason=f"idle for {self._idle_timeout_seconds}s"
        ):
            self._start_idle_timer()

    def _prepare_inputs(self, text: str, voice_samples: List[np.ndarray]) -> dict:
        inputs = self.processor(
            text=[text],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        target = self._execution_device()
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(
                    target,
                    non_blocking=target.type == "cuda",
                )
        return inputs

    def _execution_device(self) -> torch.device:
        if self.model is not None:
            try:
                for parameter in self.model.parameters():
                    if parameter.device.type != "meta":
                        return parameter.device
            except (AttributeError, StopIteration):
                pass
        return torch.device(self.device or "cpu")

    def _set_generation_state(self, inference_steps: Optional[int], seed: Optional[int]) -> None:
        # Always apply a resolved step count. This prevents a custom native request
        # from leaking its scheduler setting into later OpenAI requests.
        resolved_steps = (
            inference_steps
            if inference_steps is not None
            else self.settings.vibevoice_inference_steps
        )
        self.model.set_ddpm_inference_steps(num_steps=resolved_steps)
        if seed is not None:
            set_seed(seed)

    def _build_generation_config(
        self,
        do_sample: Optional[bool],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> dict:
        if do_sample is None and (temperature is not None or top_p is not None):
            sampling = True
        else:
            sampling = self.settings.default_do_sample if do_sample is None else do_sample
        config: dict = {"do_sample": sampling}
        if sampling:
            config.update(
                temperature=(
                    self.settings.default_temperature
                    if temperature is None
                    else temperature
                ),
                top_p=self.settings.default_top_p if top_p is None else top_p,
                top_k=self.settings.default_top_k,
            )
        if self.settings.default_repetition_penalty != 1.0:
            config["repetition_penalty"] = self.settings.default_repetition_penalty
        return config

    def generate_speech(
        self,
        text: str,
        voice_samples: List[np.ndarray],
        cfg_scale: float = 1.3,
        inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        stream: bool = False,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> AudioResult:
        """Generate speech, serializing all model state mutations."""
        if not text or not text.strip():
            return iter(()) if stream else None

        if stream:
            return self._generate_streaming(
                text=text,
                voice_samples=voice_samples,
                cfg_scale=cfg_scale,
                inference_steps=inference_steps,
                seed=seed,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                cancel_event=cancel_event,
            )

        self._request_started()
        try:
            with self._generate_lock, torch.inference_mode():
                self._ensure_loaded_locked()
                inputs = self._prepare_inputs(text, voice_samples)
                self._set_generation_state(inference_steps, seed)
                stop_check_fn = (
                    (lambda: cancel_event.is_set())
                    if cancel_event is not None
                    else None
                )

                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.settings.vibevoice_max_new_tokens,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config=self._build_generation_config(
                            do_sample, temperature, top_p
                        ),
                        stop_check_fn=stop_check_fn,
                        return_speech=True,
                        verbose=False,
                        refresh_negative=True,
                        show_progress_bar=False,
                    )
                except Exception as exc:
                    self._handle_generation_exception(exc)
                    raise

                if cancel_event is not None and cancel_event.is_set():
                    return None
                return self._extract_audio(outputs)
        finally:
            self._request_finished()

    def _extract_audio(self, outputs) -> np.ndarray:
        speech_outputs = getattr(outputs, "speech_outputs", None)
        if not speech_outputs or speech_outputs[0] is None:
            raise RuntimeError("No audio generated")

        audio = speech_outputs[0]
        if torch.is_tensor(audio):
            audio = audio.float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if self.settings.vibevoice_trim_silence:
            original_size = audio.size
            audio = trim_trailing_silence(audio, sample_rate=24000)
            removed = original_size - audio.size
            if removed > 12000:
                logger.info(
                    "Trimmed %.2fs of trailing silence",
                    removed / 24000.0,
                )
        return audio

    def _handle_generation_exception(self, exc: Exception) -> None:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if is_oom:
            logger.error("CUDA out of memory during generation")
            self._clear_cuda_cache()

    def _generate_streaming(
        self,
        *,
        text: str,
        voice_samples: List[np.ndarray],
        cfg_scale: float,
        inference_steps: Optional[int],
        seed: Optional[int],
        do_sample: Optional[bool],
        temperature: Optional[float],
        top_p: Optional[float],
        cancel_event: Optional[threading.Event],
    ) -> Iterator[np.ndarray]:
        """Stream model audio while preserving the same locked state semantics."""
        event = cancel_event or threading.Event()
        streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list[BaseException] = []
        self._request_started()

        def worker() -> None:
            try:
                with self._generate_lock, torch.inference_mode():
                    self._ensure_loaded_locked()
                    inputs = self._prepare_inputs(text, voice_samples)
                    self._set_generation_state(inference_steps, seed)
                    self.model.generate(
                        **inputs,
                        max_new_tokens=self.settings.vibevoice_max_new_tokens,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config=self._build_generation_config(
                            do_sample, temperature, top_p
                        ),
                        audio_streamer=streamer,
                        stop_check_fn=lambda: event.is_set(),
                        return_speech=True,
                        verbose=False,
                        refresh_negative=True,
                        show_progress_bar=False,
                    )
            except BaseException as exc:  # propagate worker failures to the consumer
                errors.append(exc)
                if isinstance(exc, Exception):
                    self._handle_generation_exception(exc)
                logger.exception("Streaming generation worker failed")
            finally:
                streamer.end()
                self._request_finished()

        thread = threading.Thread(target=worker, daemon=True, name="vibevoice-stream")
        try:
            thread.start()
        except BaseException:
            self._request_finished()
            raise
        completed = False

        try:
            for chunk in streamer.get_stream(0):
                if event.is_set():
                    break
                if torch.is_tensor(chunk):
                    chunk = chunk.float().cpu().numpy()
                yield np.asarray(chunk, dtype=np.float32).reshape(-1)

            if errors and not event.is_set():
                raise errors[0]
            completed = True
        finally:
            if not completed:
                event.set()
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("Streaming worker did not stop within five seconds")

    def _apply_quantization(self) -> None:
        method = self.settings.vibevoice_quantization
        if method == "int8_torchao":
            self._apply_torchao_quant(bits=8, mode="weight_only")
        elif method == "int8_dynamic_torchao":
            self._apply_torchao_quant(bits=8, mode="dynamic")
        elif method == "int4_torchao":
            self._apply_torchao_quant(bits=4, mode="weight_only")
        elif method:
            logger.warning("Unknown quantization method %r; using full precision", method)

    @staticmethod
    def _detect_unified_quantization(model_path: str) -> Optional[str]:
        try:
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, "config.json")
                if not os.path.isfile(config_path):
                    return None
            else:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(model_path, filename="config.json")

            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            quantization = config.get("quantization_config")
            if isinstance(quantization, dict):
                return quantization.get("quant_method")
        except Exception:
            logger.debug("Could not inspect quantization config for %s", model_path, exc_info=True)
        return None

    def _apply_torchao_quant(self, bits: int = 8, mode: str = "weight_only") -> None:
        try:
            from torchao.quantization import (
                int4_weight_only,
                int8_dynamic_activation_int8_weight,
                int8_weight_only,
                quantize_,
            )
        except ImportError:
            logger.warning("torchao is not installed; skipping runtime quantization")
            return

        if mode == "dynamic" and bits == 8:
            quantizer = int8_dynamic_activation_int8_weight()
            name = "INT8 dynamic activation + weight"
        elif bits == 4:
            quantizer = int4_weight_only()
            name = "INT4 weight-only"
        else:
            quantizer = int8_weight_only()
            name = "INT8 weight-only"

        try:
            logger.info("Applying torchao %s quantization to the language model", name)
            quantize_(self.model.model.language_model, quantizer)
            quantize_(self.model.lm_head, quantizer)
            gc.collect()
        except Exception:
            logger.exception("Runtime quantization failed; continuing with current weights")

    def format_script_for_single_speaker(self, text: str, speaker_id: int = 0) -> str:
        """Sanitize lines and format them for VibeVoice's speaker syntax."""
        formatted: list[str] = []
        for line in text.strip().splitlines():
            cleaned = sanitize_text(line)
            if cleaned and is_speakable(cleaned):
                formatted.append(f"Speaker {speaker_id}: {cleaned}")
        return "\n".join(formatted)
