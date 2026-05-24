"""Happy path: ASR JSON -> normalize -> stitch -> assign speakers with mocked diarization."""

from __future__ import annotations

from asr.normalizer import normalize_response
from core.schema import ChunkMeta, DiarizationTurn
from core.stitching import stitch_results
from core.timestamps import to_global_time
from diarization.speaker_assignment import assign_speakers


def test_normalize_stitch_and_assign_speakers() -> None:
    raw = {
        "text": "haan let's start karte hain",
        "language": "hi",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "haan let's start", "confidence": 0.91},
            {"start": 1.5, "end": 3.0, "text": "karte hain", "confidence": 0.88},
        ],
        "words": [
            {"start": 0.0, "end": 0.4, "word": "haan", "confidence": 0.95},
            {"start": 0.4, "end": 0.9, "word": "let's", "confidence": 0.90},
            {"start": 0.9, "end": 1.5, "word": "start", "confidence": 0.88},
            {"start": 1.6, "end": 2.2, "word": "karte", "confidence": 0.85},
            {"start": 2.2, "end": 3.0, "word": "hain", "confidence": 0.91},
        ],
    }

    chunk = ChunkMeta(
        chunk_id="spk_000_000",
        audio_path="/tmp/spk_000_000.wav",
        start_offset=120.0,
        end_offset=123.0,
    )

    result = normalize_response(
        raw,
        provider="mock",
        model="mock-model",
        chunk_id=chunk.chunk_id,
        start_offset=chunk.start_offset,
        language="hi",
    )

    assert result.segments[0].start == to_global_time(0.0, chunk)
    assert result.segments[1].end == to_global_time(3.0, chunk)
    assert all(120.0 <= w.start <= 123.0 for w in result.words)
    assert not result.weak_timestamps

    segments, words = stitch_results([result])
    assert len(segments) == 2

    turns = [
        DiarizationTurn(speaker="SPEAKER_00", start=119.0, end=121.5),
        DiarizationTurn(speaker="SPEAKER_01", start=121.5, end=125.0),
    ]
    assigned = assign_speakers(segments, words, turns, asr_provider="mock", asr_model="mock-model")

    assert assigned[0].speaker == "SPEAKER_00"
    assert assigned[1].speaker == "SPEAKER_01"
    assert assigned[0].words[0].speaker == "SPEAKER_00"
    assert assigned[0].repair_status == "raw"
