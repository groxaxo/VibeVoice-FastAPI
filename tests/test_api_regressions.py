"""Fast regression tests that do not download or load model weights."""

import numpy as np
import pytest
from pydantic import ValidationError

from api.models import (
    OpenAITTSRequest,
    SpeakerConfig,
    VibeVoiceGenerateRequest,
    VoiceListResponse,
)
from api.services.voice_manager import VoiceManager
from api.utils.audio_utils import concatenate_audio_chunks, convert_to_16_bit_wav
from api.utils.text_chunking import split_text_chunks


def test_unpunctuated_text_is_hard_bounded_without_data_loss():
    text = " ".join(f"token{i}" for i in range(200))
    chunks = split_text_chunks(text, max_chars=80)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 80 for chunk in chunks)
    assert " ".join(chunks) == text


def test_sentence_boundaries_and_ellipsis_are_preserved():
    assert split_text_chunks("Wait... really? Yes!", max_chars=100) == [
        "Wait... really?",
        "Yes!",
    ]


def test_long_sentenced_text_is_packed_between_requested_bounds():
    text = " ".join((f"Sentence {index} " + "word " * 18 + ".") for index in range(40))
    chunks = split_text_chunks(text, min_chars=1000, max_chars=2000)

    assert len(chunks) > 1
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert all(len(chunk) >= 1000 for chunk in chunks[:-1])
    assert " ".join(chunks) == text


def test_openai_and_native_requests_accept_text_beyond_old_limits():
    long_text = "A" * 120_000
    assert OpenAITTSRequest(input=long_text, voice="alloy").input == long_text
    request = VibeVoiceGenerateRequest(
        script=long_text,
        speakers=[SpeakerConfig(speaker_id=0, voice_preset="voice-a")],
    )
    assert request.script == long_text


def test_empty_audio_conversion_is_safe():
    converted = convert_to_16_bit_wav(np.array([], dtype=np.float32))
    assert converted.dtype == np.int16
    assert converted.size == 0
    assert concatenate_audio_chunks([]).dtype == np.float32


def test_speaker_requires_exactly_one_voice_source():
    with pytest.raises(ValidationError):
        SpeakerConfig(speaker_id=0)

    with pytest.raises(ValidationError):
        SpeakerConfig(
            speaker_id=0,
            voice_preset="voice-a",
            voice_sample_base64="Zm9v",
        )


def test_speaker_ids_must_be_unique_and_sequential():
    with pytest.raises(ValidationError):
        VibeVoiceGenerateRequest(
            script="Speaker 0: Hello.",
            speakers=[
                SpeakerConfig(speaker_id=0, voice_preset="voice-a"),
                SpeakerConfig(speaker_id=2, voice_preset="voice-b"),
            ],
        )


def test_voice_list_count_is_part_of_schema():
    response = VoiceListResponse(voices=[{"name": "voice-a"}], count=1)
    assert response.count == 1


def test_voice_audio_is_decoded_once_and_then_served_from_cache(tmp_path, monkeypatch):
    voice_file = tmp_path / "en-test.wav"
    voice_file.write_bytes(b"placeholder")
    manager = VoiceManager(str(tmp_path))
    calls = []

    def fake_decode(path):
        calls.append(path)
        return np.array([0.1, 0.2], dtype=np.float32), 24000

    monkeypatch.setattr(manager, "_decode_voice", fake_decode)
    first = manager.load_voice_audio("en-test")
    second = manager.load_voice_audio("en-test")

    assert len(calls) == 1
    assert np.array_equal(first, second)
    assert first is not second
