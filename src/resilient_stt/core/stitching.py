"""Combine per-chunk ASR results into a single ordered transcript stream.

The main job is to dedupe segments and words that fall inside the overlap
zones between adjacent chunks, since those regions are transcribed twice.
"""

from __future__ import annotations

from .schema import ASRResult, ASRSegment, ASRWord


_WORD_DEDUP_TOLERANCE_SEC = 0.05


def _segment_key(seg: ASRSegment) -> tuple[int, int, str]:
    return (
        int(round(seg.start * 1000)),
        int(round(seg.end * 1000)),
        seg.text.strip().lower(),
    )


def _segment_score(seg: ASRSegment) -> tuple[float, float]:
    """Rank duplicates by confidence first, then by text length as tiebreaker."""

    return (seg.confidence or 0.0, float(len(seg.text)))


def _stitch_segments(results: list[ASRResult]) -> list[ASRSegment]:
    sorted_results = sorted(results, key=lambda r: r.start_offset)
    pool: dict[tuple[int, int, str], ASRSegment] = {}
    ordered: list[ASRSegment] = []
    for res in sorted_results:
        for seg in res.segments:
            key = _segment_key(seg)
            existing = pool.get(key)
            if existing is None:
                pool[key] = seg
                ordered.append(seg)
            elif _segment_score(seg) > _segment_score(existing):
                pool[key] = seg
                ordered[ordered.index(existing)] = seg
    ordered.sort(key=lambda s: (s.start, s.end))
    return ordered


def _stitch_words(results: list[ASRResult]) -> list[ASRWord]:
    """Drop duplicate words occurring within +/- tolerance seconds."""

    sorted_results = sorted(results, key=lambda r: r.start_offset)
    seen: list[tuple[float, float, str]] = []
    ordered: list[ASRWord] = []
    for res in sorted_results:
        for word in res.words:
            norm_word = word.word.strip().lower()
            duplicate = False
            for s, e, w in seen:
                if (
                    w == norm_word
                    and abs(s - word.start) <= _WORD_DEDUP_TOLERANCE_SEC
                    and abs(e - word.end) <= _WORD_DEDUP_TOLERANCE_SEC
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
            seen.append((word.start, word.end, norm_word))
            ordered.append(word)
    ordered.sort(key=lambda w: (w.start, w.end))
    return ordered


def stitch_results(results: list[ASRResult]) -> tuple[list[ASRSegment], list[ASRWord]]:
    """Return globally-ordered, deduplicated segments and words from all chunks."""

    return _stitch_segments(results), _stitch_words(results)
