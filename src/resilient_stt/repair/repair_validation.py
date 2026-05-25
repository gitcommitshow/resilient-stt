"""Validate one repaired segment against its original to detect drift."""

from __future__ import annotations

from typing import Any

from resilient_stt.core.schema import TranscriptSegment

_LENGTH_RATIO_MAX = 1.8
_TIMESTAMP_TOLERANCE = 1e-3


def validate_repair(original: TranscriptSegment, repaired: dict[str, Any]) -> bool:
    """Return True if the repaired segment may safely replace the original's text."""

    if not isinstance(repaired, dict):
        return False
    if "text" not in repaired:
        return False

    text = repaired.get("text")
    if not isinstance(text, str):
        return False

    if repaired.get("speaker", original.speaker) != original.speaker:
        return False

    try:
        if abs(float(repaired.get("start", original.start)) - float(original.start)) > _TIMESTAMP_TOLERANCE:
            return False
        if abs(float(repaired.get("end", original.end)) - float(original.end)) > _TIMESTAMP_TOLERANCE:
            return False
    except (TypeError, ValueError):
        return False

    raw_len = max(1, len(original.raw_text or ""))
    if len(text) > raw_len * _LENGTH_RATIO_MAX:
        return False

    return True
