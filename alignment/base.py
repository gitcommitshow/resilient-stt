"""Alignment provider abstraction.

Forced alignment is opt-in. It runs only when ASR returned weak timestamps or
the operator requested it explicitly. The default v1 implementation is a
no-op pass-through so the pipeline never blocks on alignment availability.
"""

from __future__ import annotations

import abc
from pathlib import Path

from core.schema import TranscriptSegment


class AlignmentProvider(abc.ABC):
    """Refines word/segment timestamps for a transcript against the source audio."""

    name: str = "alignment-provider"

    @abc.abstractmethod
    def align(
        self,
        audio_path: str | Path,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        ...


class NoOpAligner(AlignmentProvider):
    """Default aligner: returns segments unchanged."""

    name = "noop"

    def align(
        self,
        audio_path: str | Path,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        return segments
