"""Pluggable alignment hook reserved for WhisperX or similar backends."""

from __future__ import annotations

from pathlib import Path

from resilient_stt.core.schema import TranscriptSegment

from .base import AlignmentProvider


class OptionalAligner(AlignmentProvider):
    """Stub aligner that can be subclassed or swapped without touching the pipeline."""

    name = "optional"

    def align(
        self,
        audio_path: str | Path,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        raise NotImplementedError("OptionalAligner is a stub; wire up a real backend before use.")
