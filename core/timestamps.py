"""Time conversion helpers used by the normalizer, stitcher, and exporters."""

from __future__ import annotations

from .schema import ChunkMeta


def to_global_time(local: float, chunk: ChunkMeta) -> float:
    """Convert a chunk-local timestamp (seconds) into global file time."""

    return float(chunk.start_offset) + float(local)


def format_srt(seconds: float) -> str:
    """Format seconds as `HH:MM:SS,mmm` for SRT cues."""

    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_vtt(seconds: float) -> str:
    """Format seconds as `HH:MM:SS.mmm` for WebVTT cues."""

    return format_srt(seconds).replace(",", ".")
