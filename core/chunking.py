"""Build ASR chunks from VAD speech regions.

Each speech region either becomes a single chunk (when it's short enough) or
is split into fixed-size windows with overlap. Chunk timestamps are global so
that downstream stages do not need to know about regions at all.
"""

from __future__ import annotations

from pathlib import Path

from .audio import slice_wav
from .schema import ChunkMeta, SpeechRegion


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
