"""Pause-aligned chunking (Qwen3-ASR-Toolkit style)."""

from __future__ import annotations

from core.chunking import (
    compute_pause_aligned_boundaries,
    plan_chunks_pause_aligned,
)
from core.schema import SpeechRegion

SAMPLE_RATE = 16000


def test_compute_pause_aligned_boundaries_snaps_to_onset() -> None:
    """Split targets near 120s snap to the closest speech onset."""
    # 4-minute region; onset at 115s should be chosen near the 120s target.
    onsets = [115 * SAMPLE_RATE]
    segments = compute_pause_aligned_boundaries(
        0,
        240 * SAMPLE_RATE,
        onsets,
        segment_threshold_samples=120 * SAMPLE_RATE,
        max_segment_samples=180 * SAMPLE_RATE,
    )
    assert len(segments) >= 2
    split_samples = {end for _, end in segments[:-1]}
    assert 115 * SAMPLE_RATE in split_samples


def test_plan_chunks_pause_aligned_short_region_is_one_chunk() -> None:
    """Regions under the max segment cap become a single chunk."""
    regions = [SpeechRegion(region_id="spk_000", start=0.0, end=90.0)]
    chunks = plan_chunks_pause_aligned(
        regions,
        speech_onset_samples=[0],
        sample_rate=SAMPLE_RATE,
        max_segment_sec=180.0,
    )
    assert len(chunks) == 1
    assert chunks[0].end_offset - chunks[0].start_offset == 90.0


def test_plan_chunks_pause_aligned_splits_long_region() -> None:
    """Regions longer than max_segment_sec are subdivided."""
    regions = [SpeechRegion(region_id="spk_000", start=0.0, end=400.0)]
    chunks = plan_chunks_pause_aligned(
        regions,
        speech_onset_samples=[0, 200 * SAMPLE_RATE],
        sample_rate=SAMPLE_RATE,
        segment_threshold_sec=120.0,
        max_segment_sec=180.0,
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.end_offset - chunk.start_offset <= 180.0 + 0.01
