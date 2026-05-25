"""Build OpenRouter `/audio/transcriptions` JSON bodies (base64 audio, not multipart)."""

from __future__ import annotations

import base64
from pathlib import Path

# OpenRouter model slugs use provider prefixes, e.g. `openai/whisper-1`.
DEFAULT_OPENROUTER_AUDIO_FORMAT = "wav"


def audio_format_for_path(path: Path) -> str:
    """Map a file suffix to OpenRouter ``input_audio.format`` (defaults to wav)."""
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"}:
        return suffix
    return DEFAULT_OPENROUTER_AUDIO_FORMAT


def build_transcription_json(
    model: str,
    audio_bytes: bytes,
    *,
    audio_format: str = DEFAULT_OPENROUTER_AUDIO_FORMAT,
    language: str | None = None,
    temperature: float | None = None,
) -> dict[str, object]:
    """Return the JSON body for OpenRouter STT (see openrouter.ai STT docs)."""
    payload: dict[str, object] = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": audio_format,
        },
    }
    if language:
        payload["language"] = language
    if temperature is not None:
        payload["temperature"] = temperature
    return payload
