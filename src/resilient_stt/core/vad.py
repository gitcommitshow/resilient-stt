"""Voice activity detection over a normalized 16 kHz mono WAV.

Backends: ``silero`` (neural, Qwen toolkit default), ``webrtcvad``, ``rms``,
or ``auto`` (silero → webrtcvad → rms). Output is merged/padded ``SpeechRegion``s
for chunking; Silero speech onsets are retained for pause-aligned splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .schema import SpeechRegion
from .silero_vad import (
    SILERO_INSTALL_HINT,
    SILERO_MIN_SILENCE_MS,
    SILERO_MIN_SPEECH_MS,
    detect_silero_segments,
    silero_available,
)


_FRAME_MS = 30  # webrtcvad accepts 10/20/30 ms frames


@dataclass(frozen=True)
class VadResult:
    """VAD output: merged speech regions plus raw speech-onset samples for chunking."""

    regions: list[SpeechRegion]
    speech_onsets_samples: list[int]


def _read_pcm16_mono(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="int16", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    return data, int(sr)


def _frames(samples: np.ndarray, sr: int, frame_ms: int) -> list[tuple[int, int]]:
    frame_len = int(sr * frame_ms / 1000)
    out: list[tuple[int, int]] = []
    for start in range(0, len(samples) - frame_len + 1, frame_len):
        out.append((start, start + frame_len))
    return out


def _webrtcvad_flags(samples: np.ndarray, sr: int, aggressiveness: int) -> list[bool]:
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    flags: list[bool] = []
    for start, end in _frames(samples, sr, _FRAME_MS):
        frame_bytes = samples[start:end].tobytes()
        flags.append(vad.is_speech(frame_bytes, sr))
    return flags


def _rms_flags(samples: np.ndarray, sr: int, threshold_db: float = -40.0) -> list[bool]:
    """Fallback: per-frame RMS energy gate (in dBFS) over int16 samples."""

    flags: list[bool] = []
    full_scale = 32768.0
    for start, end in _frames(samples, sr, _FRAME_MS):
        window = samples[start:end].astype(np.float32) / full_scale
        rms = float(np.sqrt(np.mean(window * window) + 1e-12))
        db = 20.0 * np.log10(rms + 1e-12)
        flags.append(db > threshold_db)
    return flags


def _flags_to_intervals(flags: list[bool], sr: int) -> list[tuple[float, float]]:
    frame_dur = _FRAME_MS / 1000.0
    intervals: list[tuple[float, float]] = []
    in_speech = False
    seg_start = 0.0
    for i, flag in enumerate(flags):
        t = i * frame_dur
        if flag and not in_speech:
            seg_start = t
            in_speech = True
        elif not flag and in_speech:
            intervals.append((seg_start, t))
            in_speech = False
    if in_speech:
        intervals.append((seg_start, len(flags) * frame_dur))
    return intervals


def _merge_and_pad(
    intervals: list[tuple[float, float]],
    duration: float,
    merge_gap_sec: float,
    pad_ms: int,
    min_speech_sec: float,
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    pad = pad_ms / 1000.0
    padded = [(max(0.0, s - pad), min(duration, e + pad)) for s, e in intervals]
    padded.sort()
    merged: list[tuple[float, float]] = [padded[0]]
    for s, e in padded[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_gap_sec:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_speech_sec]


def _resolve_backend(backend: str) -> str:
    """Map ``auto`` to the best available VAD implementation."""
    if backend == "silero" and not silero_available():
        raise ImportError(SILERO_INSTALL_HINT)
    if backend != "auto":
        return backend
    if silero_available():
        return "silero"
    try:
        import webrtcvad  # noqa: F401
    except ImportError:
        return "rms"
    return "webrtcvad"


def _silero_intervals(
    samples: np.ndarray,
    sr: int,
    *,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
) -> tuple[list[tuple[float, float]], list[int]]:
    """Return Silero speech intervals (seconds) and onset sample indices."""
    segments = detect_silero_segments(
        samples,
        sr,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    if not segments:
        return [], []
    onsets = [seg.start for seg in segments]
    intervals = [(seg.start / sr, seg.end / sr) for seg in segments]
    return intervals, onsets


def analyze_vad(
    audio_path: str | Path,
    *,
    aggressiveness: int = 2,
    pad_ms: int = 250,
    merge_gap_sec: float = 0.5,
    min_speech_sec: float = 0.3,
    min_speech_duration_ms: int = SILERO_MIN_SPEECH_MS,
    min_silence_duration_ms: int = SILERO_MIN_SILENCE_MS,
    backend: str = "auto",
) -> VadResult:
    """Run VAD and return speech regions plus speech-onset samples for chunking."""
    samples, sr = _read_pcm16_mono(audio_path)
    duration = len(samples) / float(sr)
    resolved = _resolve_backend(backend)
    speech_onsets: list[int] = []

    if resolved == "silero":
        raw, speech_onsets = _silero_intervals(
            samples,
            sr,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )
        min_len = 0.0
    elif resolved == "rms":
        flags = _rms_flags(samples, sr)
        raw = _flags_to_intervals(flags, sr)
        min_len = min_speech_sec
    elif resolved == "webrtcvad":
        flags = _webrtcvad_flags(samples, sr, aggressiveness)
        raw = _flags_to_intervals(flags, sr)
        min_len = min_speech_sec
    else:
        raise ValueError(f"Unknown VAD backend: {backend!r}")

    merged = _merge_and_pad(raw, duration, merge_gap_sec, pad_ms, min_len)
    regions = [
        SpeechRegion(region_id=f"spk_{i:03d}", start=round(s, 3), end=round(e, 3))
        for i, (s, e) in enumerate(merged)
    ]
    if not speech_onsets and raw:
        speech_onsets = [int(round(s * sr)) for s, _ in raw]
    return VadResult(regions=regions, speech_onsets_samples=speech_onsets)


def detect_speech_regions(
    audio_path: str | Path,
    *,
    aggressiveness: int = 2,
    pad_ms: int = 250,
    merge_gap_sec: float = 0.5,
    min_speech_sec: float = 0.3,
    min_speech_duration_ms: int = SILERO_MIN_SPEECH_MS,
    min_silence_duration_ms: int = SILERO_MIN_SILENCE_MS,
    backend: str = "auto",
) -> list[SpeechRegion]:
    """Return speech regions in seconds (global timeline) for the given WAV.

    ``backend``: ``silero``, ``webrtcvad``, ``rms``, or ``auto`` (best available).
    """
    return analyze_vad(
        audio_path,
        aggressiveness=aggressiveness,
        pad_ms=pad_ms,
        merge_gap_sec=merge_gap_sec,
        min_speech_sec=min_speech_sec,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        backend=backend,
    ).regions


def whole_file_region(audio_path: str | Path) -> SpeechRegion:
    """Construct a single region spanning the full file (used with `--no-vad`)."""

    samples, sr = _read_pcm16_mono(audio_path)
    duration = len(samples) / float(sr)
    return SpeechRegion(region_id="spk_000", start=0.0, end=round(duration, 3))
