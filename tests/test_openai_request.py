"""OpenAI transcription request field selection."""

from __future__ import annotations

from resilient_stt.asr.openai_request import build_transcription_fields, is_openai_transcription_model


def test_whisper_1_requests_verbose_json_with_timestamps() -> None:
    fields = dict(build_transcription_fields("whisper-1"))
    assert fields["response_format"] == "verbose_json"
    assert [v for k, v in build_transcription_fields("whisper-1") if k == "timestamp_granularities[]"] == [
        "segment",
        "word",
    ]


def test_gpt4o_transcribe_requests_json_only() -> None:
    keys = [k for k, _ in build_transcription_fields("gpt-4o-transcribe")]
    values = dict(build_transcription_fields("gpt-4o-transcribe"))
    assert values["response_format"] == "json"
    assert "timestamp_granularities[]" not in keys


def test_is_openai_transcription_model_rejects_huggingface_ids() -> None:
    assert is_openai_transcription_model("whisper-1") is True
    assert is_openai_transcription_model("gpt-4o-transcribe") is True
    assert is_openai_transcription_model("Qwen/Qwen3-ASR-1.7B") is False
