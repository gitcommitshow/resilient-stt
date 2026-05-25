"""Defaults when ``OPENAI_API_KEY`` is set — hosted ASR + LLM repair with no extra flags."""

from __future__ import annotations

import os

from .config import JobConfig

OPENAI_API_BASE_URL = "https://api.openai.com/v1"
# whisper-1 supports word/segment timestamps required by diarization + stitching.
DEFAULT_OPENAI_ASR_MODEL = "whisper-1"
DEFAULT_OPENAI_REPAIR_MODEL = "gpt-4o-mini"


def openai_api_key() -> str | None:
    """Return ``OPENAI_API_KEY`` when set (trimmed)."""
    value = (os.getenv("OPENAI_API_KEY") or "").strip()
    return value or None


def apply_openai_presets(job: JobConfig, *, repair_cli: bool | None = None) -> bool:
    """Fill OpenAI ASR/repair defaults on ``job`` when ``OPENAI_API_KEY`` is present."""
    key = openai_api_key()
    if not key:
        return False

    if job.asr_api_key is None:
        job.asr_api_key = key
    if job.repair_api_key is None:
        job.repair_api_key = key
    if not job.repair_base_url:
        job.repair_base_url = OPENAI_API_BASE_URL
    if not job.repair_model:
        job.repair_model = DEFAULT_OPENAI_REPAIR_MODEL
    if repair_cli is None:
        job.enable_repair = True
    else:
        job.enable_repair = repair_cli
    return True
