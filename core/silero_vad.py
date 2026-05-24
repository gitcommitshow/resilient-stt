"""Silero VAD helpers (Qwen3-ASR-Toolkit defaults)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Qwen3-ASR-Toolkit audio_tools.py defaults.
SILERO_MIN_SPEECH_MS = 1500
SILERO_MIN_SILENCE_MS = 500

_model: Any = None


@dataclass(frozen=True)
class SileroSpeechSegment:
    """One Silero speech interval in sample indices."""

    start: int
    end: int


SILERO_INSTALL_HINT = "Install Silero VAD: uv sync --extra silero"


def load_silero_model() -> Any:
    """Load and cache the Silero VAD model (requires ``silero-vad`` + torch)."""
    global _model
    if _model is not None:
        return _model
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise ImportError(SILERO_INSTALL_HINT) from exc
    _model = load_silero_vad()
    return _model


def silero_available() -> bool:
    """Return True when ``silero-vad`` can be imported."""
    try:
        import silero_vad  # noqa: F401
    except ImportError:
        return False
    return True


def _to_float_mono(samples: np.ndarray) -> np.ndarray:
    """Convert int16/float PCM to float32 mono in [-1, 1]."""
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    return samples.astype(np.float32)


def detect_silero_segments(
    samples: np.ndarray,
    sample_rate: int,
    *,
    min_speech_duration_ms: int = SILERO_MIN_SPEECH_MS,
    min_silence_duration_ms: int = SILERO_MIN_SILENCE_MS,
) -> list[SileroSpeechSegment]:
    """Run Silero VAD and return speech intervals as sample indices."""
    from silero_vad import get_speech_timestamps

    model = load_silero_model()
    wav = _to_float_mono(samples)
    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sample_rate,
        return_seconds=False,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    if not timestamps:
        return []
    return [SileroSpeechSegment(start=int(ts["start"]), end=int(ts["end"])) for ts in timestamps]
