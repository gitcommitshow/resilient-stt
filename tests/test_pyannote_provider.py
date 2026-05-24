"""Pyannote audio loading without torchcodec."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

from diarization.pyannote_provider import load_pyannote_audio  # noqa: E402


def test_load_pyannote_audio_mono(tmp_path: Path) -> None:
    """Mono WAV becomes (1, frames) float waveform for pyannote."""
    sr = 16000
    samples = np.zeros(sr, dtype=np.int16)
    wav = tmp_path / "mono.wav"
    sf.write(str(wav), samples, sr, subtype="PCM_16")

    payload = load_pyannote_audio(wav)

    assert payload["sample_rate"] == sr
    assert payload["waveform"].shape == (1, sr)
    assert payload["waveform"].dtype == torch.float32


def test_load_pyannote_audio_stereo(tmp_path: Path) -> None:
    """Stereo WAV is transposed to (channels, frames)."""
    sr = 8000
    frames = 400
    stereo = np.stack([np.arange(frames), np.arange(frames, 0, -1)], axis=1).astype(np.int16)
    wav = tmp_path / "stereo.wav"
    sf.write(str(wav), stereo, sr, subtype="PCM_16")

    payload = load_pyannote_audio(wav)

    assert payload["sample_rate"] == sr
    assert payload["waveform"].shape == (2, frames)
