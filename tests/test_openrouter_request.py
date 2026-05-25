"""OpenRouter STT JSON request building."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from resilient_stt.asr.openrouter_request import audio_format_for_path, build_transcription_json
from resilient_stt.orchestrator.asr_discovery import DEFAULT_OPENROUTER_ASR_MODEL, resolve_asr


def test_build_transcription_json_encodes_wav_payload() -> None:
    """Happy path: model, base64 audio, and format are included."""
    audio = b"RIFFfake-wav"
    payload = build_transcription_json("openai/whisper-1", audio, language="en")

    assert payload["model"] == "openai/whisper-1"
    assert payload["language"] == "en"
    input_audio = payload["input_audio"]
    assert isinstance(input_audio, dict)
    assert input_audio["format"] == "wav"
    assert base64.b64decode(str(input_audio["data"])) == audio


def test_audio_format_for_path_maps_known_suffixes() -> None:
    """Unknown extensions fall back to wav (pipeline output is always wav)."""
    assert audio_format_for_path(Path("clip.mp3")) == "mp3"
    assert audio_format_for_path(Path("clip.bin")) == "wav"


def test_resolve_uses_openrouter_when_local_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenRouter auto-detection runs when OPENROUTER_API_KEY is set and locals are down."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    def fake_probe(url: str, timeout_sec: float = 2.0, api_key: str | None = None) -> bool:
        return url == "https://openrouter.ai/api/v1" and api_key == "sk-or-test"

    monkeypatch.setattr("resilient_stt.orchestrator.asr_discovery.probe_asr_endpoint", fake_probe)

    resolved = resolve_asr(allow_fallback=False)

    assert resolved.source == "openrouter"
    assert resolved.model == DEFAULT_OPENROUTER_ASR_MODEL
    assert resolved.provider_label == "openrouter-hosted"
