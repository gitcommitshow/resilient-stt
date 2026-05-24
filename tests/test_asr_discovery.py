"""ASR endpoint resolution (configured URL, vLLM probe, local fallback)."""

from __future__ import annotations

import pytest

from orchestrator.asr_discovery import (
    DEFAULT_VLLM_BASE_URL,
    resolve_asr,
)


def test_resolve_uses_configured_endpoint_when_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit --asr-endpoint wins when the probe succeeds."""
    seen: list[str] = []

    def fake_probe(url: str, timeout_sec: float = 2.0) -> bool:
        seen.append(url)
        return url == "http://asr.example/v1"

    monkeypatch.setattr("orchestrator.asr_discovery.probe_asr_endpoint", fake_probe)

    resolved = resolve_asr(asr_endpoint="http://asr.example/v1", asr_model="custom-model")

    assert resolved.source == "configured"
    assert resolved.base_url == "http://asr.example/v1"
    assert resolved.model == "custom-model"
    assert seen == ["http://asr.example/v1"]


def test_resolve_uses_vllm_when_unconfigured_and_vllm_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without config, a live vLLM server on :8001 is preferred over fallback."""

    def fake_probe(url: str, timeout_sec: float = 2.0) -> bool:
        return url == DEFAULT_VLLM_BASE_URL

    monkeypatch.setattr("orchestrator.asr_discovery.probe_asr_endpoint", fake_probe)

    resolved = resolve_asr()

    assert resolved.source == "vllm"
    assert resolved.base_url == DEFAULT_VLLM_BASE_URL


def test_resolve_errors_when_fallback_disabled_and_nothing_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-asr-fallback fails fast if no endpoint responds."""

    monkeypatch.setattr("orchestrator.asr_discovery.probe_asr_endpoint", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="No ASR endpoint"):
        resolve_asr(allow_fallback=False)
