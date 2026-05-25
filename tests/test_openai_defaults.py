"""OpenAI preset defaults when OPENAI_API_KEY is set."""

from __future__ import annotations

import pytest

from orchestrator.config import JobConfig
from orchestrator.openai_defaults import (
    DEFAULT_OPENAI_ASR_MODEL,
    DEFAULT_OPENAI_REPAIR_MODEL,
    OPENAI_API_BASE_URL,
    apply_openai_presets,
)


def test_apply_openai_presets_enables_repair_and_fills_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENAI_API_KEY alone turns on repair with hosted defaults."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    job = JobConfig(
        audio_path="a.wav",
        output_dir="out",
        asr_base_url="",
        asr_model="",
    )
    assert apply_openai_presets(job, repair_cli=None) is True
    assert job.enable_repair is True
    assert job.repair_base_url == OPENAI_API_BASE_URL
    assert job.repair_model == DEFAULT_OPENAI_REPAIR_MODEL
    assert job.repair_api_key == "sk-test"
    assert job.asr_api_key == "sk-test"


def test_apply_openai_presets_respects_explicit_repair_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """--repair false disables auto repair even when OPENAI_API_KEY is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    job = JobConfig(
        audio_path="a.wav",
        output_dir="out",
        asr_base_url="",
        asr_model="",
    )
    apply_openai_presets(job, repair_cli=False)
    assert job.enable_repair is False


def test_apply_openai_presets_noop_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without OPENAI_API_KEY, presets are not applied."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    job = JobConfig(
        audio_path="a.wav",
        output_dir="out",
        asr_base_url="",
        asr_model="",
    )
    assert apply_openai_presets(job, repair_cli=None) is False
    assert job.enable_repair is False
