"""Opt out of optional usage telemetry in third-party ML dependencies."""

from __future__ import annotations

import os
from typing import MutableMapping

# Env vars that disable anonymous metrics / usage tracking (see each project's docs).
_TELEMETRY_OPT_OUT: dict[str, str] = {
    # pyannote.audio — https://github.com/pyannote/pyannote-audio#telemetry
    "PYANNOTE_METRICS_ENABLED": "0",
    # Hugging Face Hub (pyannote, transformers, qwen-asr model downloads)
    "HF_HUB_DISABLE_TELEMETRY": "1",
    # Generic convention respected by some tooling
    "DO_NOT_TRACK": "1",
}


def telemetry_env_overrides() -> dict[str, str]:
    """Return default opt-out env vars (does not mutate ``os.environ``)."""
    return dict(_TELEMETRY_OPT_OUT)


def apply_telemetry_env(env: MutableMapping[str, str]) -> None:
    """Merge telemetry opt-out into ``env`` without overriding existing values."""
    for key, value in _TELEMETRY_OPT_OUT.items():
        env.setdefault(key, value)


def disable_dependency_telemetry() -> None:
    """Disable optional dependency telemetry for the current process."""
    apply_telemetry_env(os.environ)


def disable_pyannote_session_telemetry() -> None:
    """Turn off pyannote metrics for the current Python session (after import)."""
    try:
        from pyannote.audio.telemetry import set_telemetry_metrics  # noqa: WPS433
    except ImportError:
        return
    set_telemetry_metrics(False)
