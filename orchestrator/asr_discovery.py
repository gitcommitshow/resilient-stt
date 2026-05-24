"""Resolve an ASR base URL: configured endpoint, vLLM probe, or local qwen-asr fallback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from asr.fallback_worker import (
    DEFAULT_BASE_URL as FALLBACK_BASE_URL,
    DEFAULT_MODEL as FALLBACK_MODEL,
    FallbackServerHandle,
    install_worker_deps,
    start_fallback_server,
    stop_fallback_server,
    worker_deps_installed,
)
from asr.probe import probe_asr_endpoint

logger = logging.getLogger("resilient_stt.asr_discovery")

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-ASR-1.7B"


@dataclass
class ResolvedASR:
    """Chosen ASR endpoint and optional locally managed worker process."""

    base_url: str
    model: str
    provider_label: str
    source: str
    _fallback: FallbackServerHandle | None = field(default=None, repr=False)

    def stop_managed(self) -> None:
        """Stop a fallback worker started for this run."""
        stop_fallback_server(self._fallback)


def _env_asr_endpoint() -> str | None:
    """Read an explicit ASR base URL from the environment."""
    for key in ("ASR_BASE_URL", "ASR_ENDPOINT"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value.rstrip("/")
    return None


def resolve_asr(
    *,
    asr_endpoint: str | None = None,
    asr_model: str | None = None,
    allow_fallback: bool = True,
) -> ResolvedASR:
    """Pick an ASR endpoint or start the local qwen-asr fallback when needed."""
    explicit = (asr_endpoint or _env_asr_endpoint() or "").strip().rstrip("/") or None

    if explicit:
        if not probe_asr_endpoint(explicit):
            raise RuntimeError(
                f"ASR endpoint configured but unreachable: {explicit}. "
                "Start your ASR service or omit --asr-endpoint to auto-detect."
            )
        model = asr_model or DEFAULT_VLLM_MODEL
        logger.info("Using configured ASR endpoint %s (model=%s)", explicit, model)
        return ResolvedASR(
            base_url=explicit,
            model=model,
            provider_label="external-openai-compatible",
            source="configured",
        )

    if probe_asr_endpoint(DEFAULT_VLLM_BASE_URL):
        model = asr_model or DEFAULT_VLLM_MODEL
        logger.info("Using vLLM ASR at %s (model=%s)", DEFAULT_VLLM_BASE_URL, model)
        return ResolvedASR(
            base_url=DEFAULT_VLLM_BASE_URL,
            model=model,
            provider_label="vllm-openai-compatible",
            source="vllm",
        )

    if probe_asr_endpoint(FALLBACK_BASE_URL):
        model = asr_model or FALLBACK_MODEL
        logger.info("Using existing local qwen-asr worker at %s", FALLBACK_BASE_URL)
        return ResolvedASR(
            base_url=FALLBACK_BASE_URL,
            model=model,
            provider_label="qwen-transformers-local",
            source="fallback-existing",
        )

    if not allow_fallback:
        raise RuntimeError(
            "No ASR endpoint configured and none detected on "
            f"{DEFAULT_VLLM_BASE_URL} or {FALLBACK_BASE_URL}. "
            "Pass --asr-endpoint or allow fallback (default)."
        )

    model = asr_model or FALLBACK_MODEL
    logger.info(
        "No external ASR detected; starting local qwen-asr worker (model=%s, slow CPU/MPS OK) …",
        model,
    )
    if not worker_deps_installed():
        logger.info("Installing qwen-asr worker dependencies (one-time) …")
        install_worker_deps()
    handle = start_fallback_server(model=model)
    logger.info("Local qwen-asr worker ready at %s", handle.base_url)
    return ResolvedASR(
        base_url=handle.base_url,
        model=handle.model,
        provider_label="qwen-transformers-local",
        source="fallback-started",
        _fallback=handle,
    )
