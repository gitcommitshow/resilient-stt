"""Transcript export to JSON, SRT, and WebVTT formats."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import TranscriptDocument, TranscriptSegment
from .timestamps import format_srt, format_vtt


def _segment_text(seg: TranscriptSegment) -> str:
    """Prefer repaired text, fall back to raw ASR text."""

    return (seg.clean_text or seg.raw_text or "").strip()


def export_json(doc: TranscriptDocument, path: str | Path) -> Path:
    """Write the canonical transcript JSON document."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return out


def export_srt(doc: TranscriptDocument, path: str | Path) -> Path:
    """Write SRT subtitles using `clean_text` if present, otherwise `raw_text`."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for idx, seg in enumerate(doc.segments, start=1):
        text = _segment_text(seg)
        if not text:
            continue
        speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
        lines.append(str(idx))
        lines.append(f"{format_srt(seg.start)} --> {format_srt(seg.end)}")
        lines.append(f"{speaker_prefix}{text}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_vtt(doc: TranscriptDocument, path: str | Path) -> Path:
    """Write WebVTT subtitles using `clean_text` if present, otherwise `raw_text`."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["WEBVTT", ""]
    for seg in doc.segments:
        text = _segment_text(seg)
        if not text:
            continue
        speaker_prefix = f"<v {seg.speaker}>" if seg.speaker else ""
        lines.append(f"{format_vtt(seg.start)} --> {format_vtt(seg.end)}")
        lines.append(f"{speaker_prefix}{text}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_all(doc: TranscriptDocument, output_dir: str | Path, stem: str = "transcript") -> dict[str, Path]:
    """Write JSON, SRT, and VTT side by side under `output_dir`."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "json": export_json(doc, out_dir / f"{stem}.json"),
        "srt": export_srt(doc, out_dir / f"{stem}.srt"),
        "vtt": export_vtt(doc, out_dir / f"{stem}.vtt"),
    }


def write_json(obj: object, path: str | Path) -> Path:
    """Generic JSON writer used to persist intermediate artifacts."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "model_dump"):
        payload = obj.model_dump()
    else:
        payload = obj
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out
