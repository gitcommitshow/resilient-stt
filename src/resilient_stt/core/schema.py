"""Pydantic v2 schemas shared across the pipeline.

All timestamps stored in these models are global (seconds from the start of the
original audio file) unless explicitly stated otherwise.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class SpeechRegion(_Base):
    """A contiguous speech region detected by VAD on the full normalized audio."""

    region_id: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class ChunkMeta(_Base):
    """Audio slice handed to one ASR call. Offsets are global seconds."""

    chunk_id: str
    audio_path: str
    start_offset: float
    end_offset: float
    overlap_left: float = 0.0
    overlap_right: float = 0.0
    region_id: str | None = None


class ASRWord(_Base):
    """One word from an ASR response with global timestamps."""

    word: str
    start: float
    end: float
    confidence: float | None = None


class ASRSegment(_Base):
    """One segment (sentence/phrase) from an ASR response with global timestamps."""

    text: str
    start: float
    end: float
    confidence: float | None = None


class ASRResult(_Base):
    """Normalized ASR output for a single chunk."""

    provider: str
    model: str
    chunk_id: str
    start_offset: float
    language: str | None = None
    text: str = ""
    segments: list[ASRSegment] = Field(default_factory=list)
    words: list[ASRWord] = Field(default_factory=list)
    weak_timestamps: bool = False
    raw_response: dict[str, Any] = Field(default_factory=dict)


class DiarizationTurn(_Base):
    """A single speaker turn from resilient_stt.diarization."""

    speaker: str
    start: float
    end: float


class TranscriptWord(_Base):
    """A word in the final transcript, annotated with speaker."""

    word: str
    start: float
    end: float
    speaker: str | None = None
    confidence: float | None = None


RepairStatus = Literal["raw", "unchanged", "corrected", "failed"]


class TranscriptSegment(_Base):
    """A speaker-attributed segment in the final transcript.

    `raw_text` is the unmodified ASR text; `clean_text` is the optional repaired
    text. Timestamps and speaker label must not change between raw and repaired.
    """

    speaker: str | None = None
    start: float
    end: float
    raw_text: str
    clean_text: str | None = None
    words: list[TranscriptWord] = Field(default_factory=list)
    asr_model: str | None = None
    asr_provider: str | None = None
    confidence: float | None = None
    repair_status: RepairStatus = "raw"


class TranscriptDocument(_Base):
    """Top-level exported transcript for one audio file."""

    audio_file: str
    duration: float
    language: str | None = None
    asr_provider: str | None = None
    asr_model: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)


SPEAKER_UNKNOWN = "SPEAKER_UNKNOWN"
