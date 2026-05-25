"""Edge: VAD on a mostly-silent fixture produces a single short ASR chunk, not the whole file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from resilient_stt.core.chunking import materialize_chunks, plan_chunks
from resilient_stt.core.vad import detect_speech_regions


def _write_silent_with_speech_burst(path: Path, *, sr: int = 16000) -> float:
    """Create a 30-second mono WAV: silence + 1s tone burst at ~25s."""

    total_sec = 30.0
    burst_start = 25.0
    burst_duration = 1.0
    samples = np.zeros(int(total_sec * sr), dtype=np.int16)
    t = np.arange(int(burst_duration * sr), dtype=np.float32) / sr
    tone = (0.6 * 32767 * np.sin(2 * np.pi * 440.0 * t)).astype(np.int16)
    start_idx = int(burst_start * sr)
    samples[start_idx : start_idx + len(tone)] = tone
    sf.write(str(path), samples, sr, subtype="PCM_16")
    return total_sec


def test_vad_emits_short_region_and_chunker_creates_one_chunk(tmp_path: Path) -> None:
    wav = tmp_path / "burst.wav"
    total = _write_silent_with_speech_burst(wav)

    regions = detect_speech_regions(
        wav,
        pad_ms=100,
        merge_gap_sec=0.3,
        min_speech_sec=0.2,
        backend="rms",
    )
    assert regions, "VAD should detect at least one speech region"
    region = regions[0]
    region_duration = region.end - region.start
    assert region_duration < total / 2, "speech region must be far shorter than the full file"
    assert region.start > 10.0, "speech burst is near the end of the file"

    chunks = plan_chunks(regions, chunk_threshold_sec=600.0, chunk_size_sec=60.0, chunk_overlap_sec=2.0)
    assert len(chunks) == len(regions), "short regions should produce exactly one chunk each"

    materialized = materialize_chunks(wav, chunks, tmp_path / "chunks")
    assert materialized[0].audio_path.endswith(".wav")
    out_path = Path(materialized[0].audio_path)
    assert out_path.exists() and out_path.stat().st_size > 0
