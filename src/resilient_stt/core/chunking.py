"""Build ASR chunks from VAD speech regions.

Fixed-size overlapping windows or Qwen3-ASR-Toolkit pause-aligned splits
(snap ~120s targets to speech onsets, cap at 180s per chunk).
"""

from __future__ import annotations

import math
from pathlib import Path

from .audio import slice_wav
from .schema import ChunkMeta, SpeechRegion

# Qwen3-ASR-Toolkit audio_tools.py defaults.
DEFAULT_SEGMENT_THRESHOLD_SEC = 120.0
DEFAULT_MAX_SEGMENT_SEC = 180.0


def compute_pause_aligned_boundaries(
    region_start_sample: int,
    region_end_sample: int,
    speech_onset_samples: list[int],
    *,
    segment_threshold_samples: int,
    max_segment_samples: int,
) -> list[tuple[int, int]]:
    """Split a sample range at speech-onset-aware points (Qwen toolkit algorithm)."""
    if region_end_sample <= region_start_sample:
        return []

    potential_splits = {region_start_sample, region_end_sample}
    for onset in speech_onset_samples:
        if region_start_sample <= onset <= region_end_sample:
            potential_splits.add(onset)
    sorted_splits = sorted(potential_splits)

    final_splits = {region_start_sample, region_end_sample}
    target = region_start_sample + segment_threshold_samples
    while target < region_end_sample:
        closest = min(sorted_splits, key=lambda p: abs(p - target))
        final_splits.add(closest)
        target += segment_threshold_samples
    ordered = sorted(final_splits)

    new_points = [region_start_sample]
    for i in range(1, len(ordered)):
        start = ordered[i - 1]
        end = ordered[i]
        length = end - start
        if length <= max_segment_samples:
            new_points.append(end)
        else:
            parts = int(math.ceil(length / max_segment_samples))
            step = length / parts
            for j in range(1, parts):
                new_points.append(int(start + j * step))
            new_points.append(end)

    segments: list[tuple[int, int]] = []
    for i in range(len(new_points) - 1):
        s, e = int(new_points[i]), int(new_points[i + 1])
        if e > s:
            segments.append((s, e))
    return segments


def _fallback_fixed_segments(
    region_start_sample: int,
    region_end_sample: int,
    max_segment_samples: int,
) -> list[tuple[int, int]]:
    """Fixed-size fallback when no speech onsets are available (Qwen toolkit)."""
    segments: list[tuple[int, int]] = []
    for start in range(region_start_sample, region_end_sample, max_segment_samples):
        end = min(start + max_segment_samples, region_end_sample)
        if end > start:
            segments.append((start, end))
    return segments


def plan_chunks_pause_aligned(
    regions: list[SpeechRegion],
    speech_onset_samples: list[int],
    sample_rate: int,
    *,
    segment_threshold_sec: float = DEFAULT_SEGMENT_THRESHOLD_SEC,
    max_segment_sec: float = DEFAULT_MAX_SEGMENT_SEC,
) -> list[ChunkMeta]:
    """Plan chunks using Qwen-style pause-aligned boundaries per speech region."""
    threshold_samples = int(segment_threshold_sec * sample_rate)
    max_samples = int(max_segment_sec * sample_rate)
    chunks: list[ChunkMeta] = []

    for region in regions:
        region_start = int(round(region.start * sample_rate))
        region_end = int(round(region.end * sample_rate))
        if region_end <= region_start:
            continue

        if speech_onset_samples:
            boundaries = compute_pause_aligned_boundaries(
                region_start,
                region_end,
                speech_onset_samples,
                segment_threshold_samples=threshold_samples,
                max_segment_samples=max_samples,
            )
        else:
            boundaries = _fallback_fixed_segments(region_start, region_end, max_samples)

        if not boundaries:
            boundaries = [(region_start, region_end)]

        for idx, (start_sample, end_sample) in enumerate(boundaries):
            chunks.append(
                ChunkMeta(
                    chunk_id=f"{region.region_id}_{idx:03d}",
                    audio_path="",
                    start_offset=round(start_sample / sample_rate, 3),
                    end_offset=round(end_sample / sample_rate, 3),
                    region_id=region.region_id,
                )
            )
    return chunks


def plan_chunks(
    regions: list[SpeechRegion],
    *,
    chunk_threshold_sec: float = 600.0,
    chunk_size_sec: float = 60.0,
    chunk_overlap_sec: float = 2.0,
) -> list[ChunkMeta]:
    """Return chunk metadata (without writing audio) for the given regions.

    A region shorter than `chunk_threshold_sec` becomes one chunk; longer
    regions are split into overlapping windows of `chunk_size_sec`.
    """

    chunks: list[ChunkMeta] = []
    for region in regions:
        duration = region.duration
        if duration <= 0:
            continue

        if duration <= chunk_threshold_sec:
            chunks.append(
                ChunkMeta(
                    chunk_id=f"{region.region_id}_000",
                    audio_path="",
                    start_offset=region.start,
                    end_offset=region.end,
                    region_id=region.region_id,
                )
            )
            continue

        step = max(chunk_size_sec - chunk_overlap_sec, 1.0)
        cursor = region.start
        idx = 0
        while cursor < region.end:
            start = cursor
            end = min(region.end, start + chunk_size_sec)
            overlap_left = chunk_overlap_sec if idx > 0 else 0.0
            overlap_right = chunk_overlap_sec if end < region.end else 0.0
            chunks.append(
                ChunkMeta(
                    chunk_id=f"{region.region_id}_{idx:03d}",
                    audio_path="",
                    start_offset=round(start, 3),
                    end_offset=round(end, 3),
                    overlap_left=overlap_left,
                    overlap_right=overlap_right,
                    region_id=region.region_id,
                )
            )
            if end >= region.end:
                break
            cursor = end - chunk_overlap_sec
            idx += 1
    return chunks


def materialize_chunks(
    normalized_wav: str | Path,
    chunks: list[ChunkMeta],
    chunks_dir: str | Path,
) -> list[ChunkMeta]:
    """Write per-chunk WAV files and return chunks with `audio_path` filled in."""

    out_dir = Path(chunks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    materialized: list[ChunkMeta] = []
    for chunk in chunks:
        dst = out_dir / f"{chunk.chunk_id}.wav"
        slice_wav(normalized_wav, dst, chunk.start_offset, chunk.end_offset)
        materialized.append(chunk.model_copy(update={"audio_path": str(dst)}))
    return materialized
