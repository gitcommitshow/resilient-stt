"""Placeholder for a future Qwen-based forced aligner."""

from __future__ import annotations

from pathlib import Path

from core.schema import TranscriptSegment

from .base import AlignmentProvider


class QwenAligner(AlignmentProvider):
    """Not implemented in v1; wire up when a Qwen alignment service is available."""

    name = "qwen"

    def align(
        self,
        audio_path: str | Path,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        raise NotImplementedError("QwenAligner is not implemented in v1.")
