"""Attach speaker labels to ASR segments and words using diarization turns.

Primary strategy: per-word maximum-overlap with the diarization turns. The
segment speaker is the majority vote of its word speakers. When the ASR did
not return words, fall back to the segment midpoint.
"""

from __future__ import annotations

from collections import Counter

from core.schema import (
    SPEAKER_UNKNOWN,
    ASRSegment,
    ASRWord,
    DiarizationTurn,
    TranscriptSegment,
    TranscriptWord,
)


def _max_overlap_speaker(
    start: float,
    end: float,
    turns: list[DiarizationTurn],
) -> str | None:
    """Pick the speaker whose turn has the largest overlap with `[start, end]`."""

    best_speaker: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    return best_speaker


def _speaker_at(time: float, turns: list[DiarizationTurn]) -> str | None:
    for turn in turns:
        if turn.start <= time <= turn.end:
            return turn.speaker
    return None


def _words_for_segment(seg: ASRSegment, words: list[ASRWord]) -> list[ASRWord]:
    return [w for w in words if w.start >= seg.start - 1e-3 and w.end <= seg.end + 1e-3]


def assign_speakers(
    segments: list[ASRSegment],
    words: list[ASRWord],
    turns: list[DiarizationTurn],
    *,
    asr_provider: str | None = None,
    asr_model: str | None = None,
) -> list[TranscriptSegment]:
    """Build speaker-attributed transcript segments from ASR + diarization."""

    out: list[TranscriptSegment] = []
    for seg in segments:
        seg_words = _words_for_segment(seg, words)
        annotated_words: list[TranscriptWord] = []
        speaker_votes: Counter[str] = Counter()

        for w in seg_words:
            speaker = _max_overlap_speaker(w.start, w.end, turns) if turns else None
            if speaker:
                speaker_votes[speaker] += 1
            annotated_words.append(
                TranscriptWord(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    speaker=speaker,
                    confidence=w.confidence,
                )
            )

        if speaker_votes:
            seg_speaker = speaker_votes.most_common(1)[0][0]
        elif turns:
            midpoint = (seg.start + seg.end) / 2.0
            seg_speaker = (
                _speaker_at(midpoint, turns)
                or _max_overlap_speaker(seg.start, seg.end, turns)
                or SPEAKER_UNKNOWN
            )
        else:
            seg_speaker = None

        out.append(
            TranscriptSegment(
                speaker=seg_speaker,
                start=seg.start,
                end=seg.end,
                raw_text=seg.text,
                words=annotated_words,
                asr_model=asr_model,
                asr_provider=asr_provider,
                confidence=seg.confidence,
                repair_status="raw",
            )
        )
    return out
