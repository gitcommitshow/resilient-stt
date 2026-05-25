"""Map heterogeneous ASR responses into the internal `ASRResult` schema.

Local (chunk-relative) timestamps are converted to global timestamps using the
chunk's `start_offset`. If timestamps are missing, the result is flagged with
`weak_timestamps=True` so the alignment stage can fill them in.
"""

from __future__ import annotations

from typing import Any

from resilient_stt.core.schema import ASRResult, ASRSegment, ASRWord


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_word(item: dict[str, Any], offset: float) -> ASRWord | None:
    text = item.get("word") or item.get("text")
    start = _coerce_float(item.get("start"))
    end = _coerce_float(item.get("end"))
    if not text or start is None or end is None:
        return None
    return ASRWord(
        word=str(text),
        start=offset + start,
        end=offset + end,
        confidence=_coerce_float(item.get("confidence") or item.get("probability")),
    )


def _to_segment(item: dict[str, Any], offset: float) -> ASRSegment | None:
    text = item.get("text") or ""
    start = _coerce_float(item.get("start"))
    end = _coerce_float(item.get("end"))
    if start is None or end is None:
        return None
    confidence = _coerce_float(item.get("confidence"))
    if confidence is None:
        avg_logprob = _coerce_float(item.get("avg_logprob"))
        if avg_logprob is not None:
            confidence = max(0.0, min(1.0, 1.0 + avg_logprob))
    return ASRSegment(
        text=str(text).strip(),
        start=offset + start,
        end=offset + end,
        confidence=confidence,
    )


def normalize_response(
    raw: dict[str, Any],
    *,
    provider: str,
    model: str,
    chunk_id: str,
    start_offset: float,
    language: str | None = None,
    fallback_end: float | None = None,
) -> ASRResult:
    """Translate one OpenAI-compatible JSON response into an `ASRResult`."""

    segments_raw = raw.get("segments") or []
    words_raw = raw.get("words") or []

    segments: list[ASRSegment] = []
    for item in segments_raw:
        if isinstance(item, dict):
            seg = _to_segment(item, start_offset)
            if seg:
                segments.append(seg)

    words: list[ASRWord] = []
    for item in words_raw:
        if isinstance(item, dict):
            word = _to_word(item, start_offset)
            if word:
                words.append(word)

    text = str(raw.get("text") or "").strip()
    weak = False
    if not segments and text:
        # No segment-level timestamps available; create a single segment spanning the chunk.
        end_time = fallback_end if fallback_end is not None else start_offset
        segments.append(
            ASRSegment(
                text=text,
                start=start_offset,
                end=max(end_time, start_offset),
                confidence=None,
            )
        )
        weak = True

    return ASRResult(
        provider=provider,
        model=model,
        chunk_id=chunk_id,
        start_offset=start_offset,
        language=language or raw.get("language"),
        text=text or " ".join(s.text for s in segments).strip(),
        segments=segments,
        words=words,
        weak_timestamps=weak,
        raw_response=raw,
    )
