"""Audio normalization with optional ffmpeg enhancement filters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from resilient_stt.core.audio import build_enhance_audio_filter, normalize_audio


def test_build_enhance_audio_filter_includes_denoise_chain() -> None:
    """Enhancement chain targets rumble, steady noise, and quiet speech."""
    filt = build_enhance_audio_filter()
    assert "highpass=f=80" in filt
    assert "afftdn=nf=-25" in filt
    assert "loudnorm=I=-16:TP=-1.5:LRA=11" in filt


def test_normalize_audio_passes_af_when_enhance_enabled(tmp_path: Path) -> None:
    """``--enhance-audio`` adds an ffmpeg audio filter before resampling."""
    src = tmp_path / "in.wav"
    dst = tmp_path / "out.wav"
    src.write_bytes(b"fake")

    proc = MagicMock(returncode=0, stderr="")
    with patch("resilient_stt.core.audio._require_ffmpeg", return_value="/usr/bin/ffmpeg"):
        with patch("resilient_stt.core.audio.subprocess.run", return_value=proc) as run:
            normalize_audio(src, dst, enhance_audio=True)

    cmd = run.call_args.args[0]
    assert "-af" in cmd
    af_idx = cmd.index("-af")
    assert "afftdn" in cmd[af_idx + 1]
