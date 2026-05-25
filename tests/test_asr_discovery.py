"""ASR endpoint resolution (configured URL, vLLM probe, local fallback)."""

from __future__ import annotations

import pytest

from orchestrator.asr_discovery import (
    DEFAULT_OPENAI_ASR_MODEL,
    DEFAULT_VLLM_BASE_URL,
    resolve_asr,
)


def test_resolve_uses_configured_endpoint_when_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit --asr-endpoint wins when the probe succeeds."""
    seen: list[str] = []

    def fake_probe(url: str, timeout_sec: float = 2.0, api_key: str | None = None) -> bool:
        seen.append(url)
        return url == "http://asr.example/v1"

    monkeypatch.setattr("orchestrator.asr_discovery.probe_asr_endpoint", fake_probe)

    resolved = resolve_asr(asr_endpoint="http://asr.example/v1", asr_model="custom-model")

    assert resolved.source == "configured"
    assert resolved.base_url == "http://asr.example/v1"
    assert resolved.model == "custom-model"
    assert seen == ["http://asr.example/v1"]


def test_resolve_prefers_vllm_over_openai_when_both_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local vLLM wins over OpenAI when both respond and no --model/--asr-endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_probe(url: str, timeout_sec: float = 2.0, api_key: str | None = None) -> bool:
        return url in (DEFAULT_VLLM_BASE_URL, "https://api.openai.com/v1")

    monkeypatch.setattr("orchestrator.asr_discovery.probe_asr_endpoint", fake_probe)

    resolved = resolve_asr()

    assert resolved.source == "vllm"
    assert resolved.base_url == DEFAULT_VLLM_BASE_URL


def test_resolve_uses_openai_when_local_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI auto-detection runs only when local ASR is down and --model is omitted."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_probe(url: str, timeout_sec: float = 2.0, api_key: str | None = None) -> bool:
        return url == "https://api.openai.com/v1" and api_key == "sk-test"

    monkeypatch.setattr("orchestrator.asr_discovery.probe_asr_endpoint", fake_probe)

    resolved = resolve_asr(allow_fallback=False)

    assert resolved.source == "openai"
    assert resolved.model == DEFAULT_OPENAI_ASR_MODEL


def test_resolve_skips_openai_when_model_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit --model never triggers OpenAI auto-detection."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_probe(url: str, timeout_sec: float = 2.0, api_key: str | None = None) -> bool:
        return url == DEFAULT_VLLM_BASE_URL

    monkeypatch.setattr("orchestrator.asr_discovery.probe_asr_endpoint", fake_probe)

    resolved = resolve_asr(asr_model="Qwen/Qwen3-ASR-1.7B")

    assert resolved.source == "vllm"
    assert resolved.model == "Qwen/Qwen3-ASR-1.7B"
