"""Voice activity detection over a normalized 16 kHz mono WAV.

Default backend is `webrtcvad`; an RMS-energy fallback is used when the
binding is unavailable so the pipeline still functions in lean environments.
The output is a list of merged, padded `SpeechRegion`s used by the chunker so
that silent stretches are never sent to ASR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .schema import SpeechRegion


_FRAME_MS = 30  # webrtcvad accepts 10/20/30 ms frames


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


def detect_speech_regions(
    audio_path: str | Path,
    *,
    aggressiveness: int = 2,
    pad_ms: int = 250,
    merge_gap_sec: float = 0.5,
    min_speech_sec: float = 0.3,
    backend: str = "auto",
) -> list[SpeechRegion]:
    """Return speech regions in seconds (global timeline) for the given WAV.

    ``backend`` selects the VAD implementation: ``"webrtcvad"`` (requires the
    binding), ``"rms"`` for the energy-only fallback, or ``"auto"`` which
    prefers webrtcvad and falls back to RMS if it is unavailable.
    """

    samples, sr = _read_pcm16_mono(audio_path)
    duration = len(samples) / float(sr)

    if backend == "rms":
        flags = _rms_flags(samples, sr)
    elif backend == "webrtcvad":
        flags = _webrtcvad_flags(samples, sr, aggressiveness)
    else:
        try:
            flags = _webrtcvad_flags(samples, sr, aggressiveness)
        except Exception:
            flags = _rms_flags(samples, sr)

    raw = _flags_to_intervals(flags, sr)
    merged = _merge_and_pad(raw, duration, merge_gap_sec, pad_ms, min_speech_sec)
    return [
        SpeechRegion(region_id=f"spk_{i:03d}", start=round(s, 3), end=round(e, 3))
        for i, (s, e) in enumerate(merged)
    ]


def whole_file_region(audio_path: str | Path) -> SpeechRegion:
    """Construct a single region spanning the full file (used with `--no-vad`)."""

    samples, sr = _read_pcm16_mono(audio_path)
    duration = len(samples) / float(sr)
    return SpeechRegion(region_id="spk_000", start=0.0, end=round(duration, 3))
