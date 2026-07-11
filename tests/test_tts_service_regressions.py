"""State-isolation regressions for the TTS service (no model weights required)."""

from __future__ import annotations

import queue

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

import api.services.tts_service as tts_module
from api.config import Settings
from api.services.tts_service import TTSService


class _FakeProcessor:
    tokenizer = object()

    def __call__(self, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}


class _FakeOutput:
    speech_outputs = [torch.tensor([0.1, 0.2], dtype=torch.float32)]


class _FakeModel:
    def __init__(self):
        self.step_calls: list[int] = []
        self.generation_configs: list[dict] = []
        self.parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.parameter

    def set_ddpm_inference_steps(self, num_steps: int):
        self.step_calls.append(num_steps)

    def generate(self, **kwargs):
        self.generation_configs.append(kwargs["generation_config"])
        streamer = kwargs.get("audio_streamer")
        if streamer is not None:
            streamer.put(torch.tensor([0.3, 0.4], dtype=torch.float32))
        return _FakeOutput()


class _FakeAudioStreamer:
    def __init__(self, *args, **kwargs):
        self._queue = queue.Queue()
        self._sentinel = object()

    def put(self, value):
        self._queue.put(value)

    def end(self):
        self._queue.put(self._sentinel)

    def get_stream(self, index):
        while True:
            value = self._queue.get(timeout=2.0)
            if value is self._sentinel:
                return
            yield value


def _service() -> TTSService:
    settings = Settings(
        vibevoice_device="cpu",
        vibevoice_lazy_load=False,
        vibevoice_inference_steps=7,
        vibevoice_trim_silence=False,
    )
    service = TTSService(settings)
    service.model = _FakeModel()
    service.processor = _FakeProcessor()
    service.device = "cpu"
    service._model_loaded = True
    return service


def test_custom_steps_do_not_leak_into_the_next_request():
    service = _service()
    voice = [np.zeros(100, dtype=np.float32)]

    service.generate_speech("Speaker 0: one.", voice, inference_steps=17)
    service.generate_speech("Speaker 0: two.", voice)

    assert service.model.step_calls == [17, 7]
    assert service.is_busy is False


def test_streaming_sampling_values_are_forwarded(monkeypatch):
    monkeypatch.setattr(tts_module, "AudioStreamer", _FakeAudioStreamer)
    service = _service()
    voice = [np.zeros(100, dtype=np.float32)]

    chunks = list(
        service.generate_speech(
            "Speaker 0: stream.",
            voice,
            stream=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )
    )

    assert len(chunks) == 1
    assert np.allclose(chunks[0], [0.3, 0.4])
    assert service.model.generation_configs[-1] == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": service.settings.default_top_k,
    }
    assert service.is_busy is False
