"""Default opt-out for third-party dependency telemetry."""

from __future__ import annotations

import os

from core.privacy import apply_telemetry_env, disable_dependency_telemetry


def test_disable_dependency_telemetry_sets_opt_out_defaults(monkeypatch) -> None:
    """Unset telemetry env vars get conservative opt-out defaults."""
    for key in ("PYANNOTE_METRICS_ENABLED", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        monkeypatch.delenv(key, raising=False)

    disable_dependency_telemetry()

    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "0"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["DO_NOT_TRACK"] == "1"


def test_disable_dependency_telemetry_respects_existing_env(monkeypatch) -> None:
    """Explicit shell/.env values are not overwritten (opt-in still possible)."""
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "1")
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)

    disable_dependency_telemetry()

    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_apply_telemetry_env_merges_into_subprocess_env() -> None:
    """Child process env copies inherit opt-out without clobbering caller values."""
    env = {"PATH": "/usr/bin", "PYANNOTE_METRICS_ENABLED": "1"}
    apply_telemetry_env(env)
    assert env["PYANNOTE_METRICS_ENABLED"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
