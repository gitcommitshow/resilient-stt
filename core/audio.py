"""Audio normalization via ffmpeg subprocess.

The pipeline operates on 16 kHz mono PCM WAV throughout. This module is the
only place that shells out to ffmpeg; everything downstream reads the produced
WAV directly with `soundfile`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import soundfile as sf


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg is not on PATH."""


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFoundError(
            "ffmpeg is required for audio normalization. Install it (e.g. `brew install ffmpeg`) and retry."
        )
    return path


def normalize_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
) -> Path:
    """Decode `input_path` to mono PCM WAV at `sample_rate` Hz.

    Returns the resolved output path. Overwrites any existing file at the
    destination so repeat runs are deterministic.
    """

    ffmpeg = _require_ffmpeg()
    src = Path(input_path).expanduser().resolve()
    dst = Path(output_path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        "-f",
        "wav",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={result.returncode}): {result.stderr.strip()}")
    return dst


def audio_duration(path: str | Path) -> float:
    """Return the duration in seconds of a normalized WAV file."""

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def slice_wav(
    src_path: str | Path,
    dst_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> Path:
    """Extract `[start_sec, end_sec)` from a WAV file into a new WAV.

    Uses `soundfile` for accuracy. Both endpoints are clamped to the file.
    """

    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(src)) as f:
        sr = f.samplerate
        total = f.frames
        start_frame = max(0, int(round(start_sec * sr)))
        end_frame = min(total, int(round(end_sec * sr)))
        if end_frame <= start_frame:
            raise ValueError(f"Empty slice: start={start_sec}s end={end_sec}s")
        f.seek(start_frame)
        data = f.read(end_frame - start_frame, dtype="int16", always_2d=False)
    sf.write(str(dst), data, sr, subtype="PCM_16")
    return dst
