"""Error path: repair validator rejects timestamp drift and speaker changes."""

from __future__ import annotations

from core.schema import TranscriptSegment
from repair.repair_validation import validate_repair


def _segment() -> TranscriptSegment:
    return TranscriptSegment(
        speaker="SPEAKER_00",
        start=10.0,
        end=12.5,
        raw_text="muje lagta he ye api kal tak ho jayga",
    )


def test_validator_rejects_timestamp_drift_and_speaker_change() -> None:
    original = _segment()

    drifted = {
        "speaker": "SPEAKER_00",
        "start": 10.5,  # drifted
        "end": 12.5,
        "text": "Mujhe lagta hai ye API kal tak ho jayega.",
    }
    assert validate_repair(original, drifted) is False

    speaker_changed = {
        "speaker": "SPEAKER_01",
        "start": 10.0,
        "end": 12.5,
        "text": "Mujhe lagta hai ye API kal tak ho jayega.",
    }
    assert validate_repair(original, speaker_changed) is False

    overlong = {
        "speaker": "SPEAKER_00",
        "start": 10.0,
        "end": 12.5,
        "text": "x" * (len(original.raw_text) * 2),
    }
    assert validate_repair(original, overlong) is False

    valid = {
        "speaker": "SPEAKER_00",
        "start": 10.0,
        "end": 12.5,
        "text": "Mujhe lagta hai ye API kal tak ho jayega.",
    }
    assert validate_repair(original, valid) is True
