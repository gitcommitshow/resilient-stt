"""Build OpenAI `/audio/transcriptions` form fields per model capabilities."""

from __future__ import annotations

WHISPER_MODELS = frozenset({"whisper-1"})
GPT4O_TRANSCRIBE_MODELS = frozenset({"gpt-4o-transcribe", "gpt-4o-mini-transcribe"})
GPT4O_DIARIZE_MODEL = "gpt-4o-transcribe-diarize"
OPENAI_TRANSCRIPTION_MODELS = WHISPER_MODELS | GPT4O_TRANSCRIBE_MODELS | frozenset({GPT4O_DIARIZE_MODEL})


def is_openai_transcription_model(model: str) -> bool:
    """Return True when ``model`` is a hosted OpenAI ``/audio/transcriptions`` model id."""
    if model in OPENAI_TRANSCRIPTION_MODELS:
        return True
    if model.startswith("whisper-"):
        return True
    if model.startswith("gpt-4o") and "transcribe" in model:
        return True
    return False


def build_transcription_fields(
    model: str,
    *,
    language: str | None = None,
    prompt: str | None = None,
    duration_sec: float | None = None,
) -> list[tuple[str, str]]:
    """Return multipart form fields appropriate for the given transcription model."""
    fields: list[tuple[str, str]] = [("model", model)]

    if model in WHISPER_MODELS or model.startswith("whisper-"):
        fields.extend(
            [
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"),
                ("timestamp_granularities[]", "word"),
            ]
        )
    elif model in GPT4O_TRANSCRIBE_MODELS:
        # gpt-4o transcribe models only accept json/text — not verbose_json or timestamps.
        fields.append(("response_format", "json"))
    elif model == GPT4O_DIARIZE_MODEL:
        fields.append(("response_format", "diarized_json"))
        if duration_sec is not None and duration_sec > 30.0:
            fields.append(("chunking_strategy", "auto"))
    else:
        # vLLM, qwen-asr workers, and other OpenAI-shaped servers expect verbose_json.
        fields.extend(
            [
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"),
                ("timestamp_granularities[]", "word"),
            ]
        )

    if language:
        fields.append(("language", language))
    if prompt and model != GPT4O_DIARIZE_MODEL:
        fields.append(("prompt", prompt))
    return fields
