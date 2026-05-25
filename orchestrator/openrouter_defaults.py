"""Defaults when ``OPENROUTER_API_KEY`` is set — hosted ASR + LLM repair with no extra flags."""

from __future__ import annotations

import os

from .config import JobConfig

OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
# whisper-1 on OpenRouter; response is text-only (weak timestamps → optional align).
DEFAULT_OPENROUTER_ASR_MODEL = "openai/whisper-1"
DEFAULT_OPENROUTER_REPAIR_MODEL = "openai/gpt-4o-mini"


def openrouter_api_key() -> str | None:
    """Return ``OPENROUTER_API_KEY`` when set (trimmed)."""
    value = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    return value or None


def is_openrouter_endpoint(base_url: str) -> bool:
    """Return True when ``base_url`` points at the OpenRouter API."""
    return "openrouter.ai" in base_url


def apply_openrouter_presets(job: JobConfig, *, repair_cli: bool | None = None) -> bool:
    """Fill OpenRouter ASR/repair defaults on ``job`` when ``OPENROUTER_API_KEY`` is present."""
    key = openrouter_api_key()
    if not key:
        return False

    if job.asr_api_key is None:
        job.asr_api_key = key
    if job.repair_api_key is None:
        job.repair_api_key = key
    if not job.repair_base_url:
        job.repair_base_url = OPENROUTER_API_BASE_URL
    if not job.repair_model:
        job.repair_model = DEFAULT_OPENROUTER_REPAIR_MODEL
    if repair_cli is None:
        job.enable_repair = True
    else:
        job.enable_repair = repair_cli
    return True
