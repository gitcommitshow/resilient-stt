"""Resolve an ASR base URL: configured endpoint, vLLM probe, or local qwen-asr fallback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from resilient_stt.asr.fallback_worker import (
    DEFAULT_BASE_URL as FALLBACK_BASE_URL,
    DEFAULT_HOST as FALLBACK_HOST,
    DEFAULT_MODEL as FALLBACK_MODEL,
    DEFAULT_PORT as FALLBACK_PORT,
    FallbackServerHandle,
    install_worker_deps,
    is_tcp_port_open,
    start_fallback_server,
    stop_fallback_server,
    wait_for_existing_worker,
    worker_deps_installed,
)
from resilient_stt.asr.probe import probe_asr_endpoint

from .openai_defaults import (
    DEFAULT_OPENAI_ASR_MODEL,
    OPENAI_API_BASE_URL,
    openai_api_key,
)
from .openrouter_defaults import (
    DEFAULT_OPENROUTER_ASR_MODEL,
    OPENROUTER_API_BASE_URL,
    is_openrouter_endpoint,
    openrouter_api_key,
)

logger = logging.getLogger("resilient_stt.asr_discovery")

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-ASR-1.7B"
# Local CPU/MPS inference can exceed 10 min for long clips; avoid premature HTTP retries.
LOCAL_ASR_TIMEOUT_SEC = 7200.0


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


def _env_asr_api_key() -> str | None:
    """Read an ASR bearer token from the environment."""
    for key in ("ASR_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return None


def _default_model_for_endpoint(endpoint: str, asr_model: str | None) -> str:
    """Pick a sensible default model id when the caller did not pass ``--model``."""
    if asr_model:
        return asr_model
    if is_openrouter_endpoint(endpoint):
        return DEFAULT_OPENROUTER_ASR_MODEL
    if "api.openai.com" in endpoint:
        return DEFAULT_OPENAI_ASR_MODEL
    return DEFAULT_VLLM_MODEL


def _resolve_openrouter_asr() -> ResolvedASR | None:
    """Use hosted OpenRouter STT when reachable (caller checks prerequisites)."""
    key = openrouter_api_key()
    if not key:
        return None
    if not probe_asr_endpoint(OPENROUTER_API_BASE_URL, api_key=key):
        return None
    logger.info(
        "Using OpenRouter ASR at %s (model=%s)",
        OPENROUTER_API_BASE_URL,
        DEFAULT_OPENROUTER_ASR_MODEL,
    )
    return ResolvedASR(
        base_url=OPENROUTER_API_BASE_URL,
        model=DEFAULT_OPENROUTER_ASR_MODEL,
        provider_label="openrouter-hosted",
        source="openrouter",
    )


def _try_openrouter_asr(*, asr_model: str | None, asr_endpoint: str | None) -> ResolvedASR | None:
    """OpenRouter auto-detection: no explicit model/URL, key present, nothing local."""
    if asr_model is not None:
        return None
    if asr_endpoint or _env_asr_endpoint():
        return None
    return _resolve_openrouter_asr()


def _resolve_openai_asr() -> ResolvedASR | None:
    """Use hosted OpenAI transcription when reachable (caller checks prerequisites)."""
    key = openai_api_key()
    if not key:
        return None
    if not probe_asr_endpoint(OPENAI_API_BASE_URL, api_key=key):
        return None
    logger.info(
        "Using OpenAI ASR at %s (model=%s)",
        OPENAI_API_BASE_URL,
        DEFAULT_OPENAI_ASR_MODEL,
    )
    return ResolvedASR(
        base_url=OPENAI_API_BASE_URL,
        model=DEFAULT_OPENAI_ASR_MODEL,
        provider_label="openai-hosted",
        source="openai",
    )


def _try_openai_asr(*, asr_model: str | None, asr_endpoint: str | None) -> ResolvedASR | None:
    """OpenAI auto-detection: no explicit model/URL, key present, nothing local."""
    if asr_model is not None:
        return None
    if asr_endpoint or _env_asr_endpoint():
        return None
    return _resolve_openai_asr()


def resolve_asr(
    *,
    asr_endpoint: str | None = None,
    asr_model: str | None = None,
    allow_fallback: bool = True,
) -> ResolvedASR:
    """Pick an ASR endpoint or start the local qwen-asr fallback when needed."""
    explicit = (asr_endpoint or _env_asr_endpoint() or "").strip().rstrip("/") or None

    if explicit:
        api_key = _env_asr_api_key()
        if not probe_asr_endpoint(explicit, api_key=api_key):
            raise RuntimeError(
                f"ASR endpoint configured but unreachable: {explicit}. "
                "Start your ASR service or omit --asr-endpoint to auto-detect."
            )
        model = _default_model_for_endpoint(explicit, asr_model)
        label = "openrouter-hosted" if is_openrouter_endpoint(explicit) else "external-openai-compatible"
        logger.info("Using configured ASR endpoint %s (model=%s)", explicit, model)
        return ResolvedASR(
            base_url=explicit,
            model=model,
            provider_label=label,
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

    openrouter = _try_openrouter_asr(asr_model=asr_model, asr_endpoint=asr_endpoint)
    if openrouter is not None:
        return openrouter

    openai = _try_openai_asr(asr_model=asr_model, asr_endpoint=asr_endpoint)
    if openai is not None:
        return openai

    if not allow_fallback:
        raise RuntimeError(
            "No ASR endpoint configured and none detected on "
            f"{DEFAULT_VLLM_BASE_URL} or {FALLBACK_BASE_URL}. "
            "Pass --asr-endpoint or allow fallback (default)."
        )

    if is_tcp_port_open(FALLBACK_HOST, FALLBACK_PORT):
        logger.info(
            "Port %d is in use; waiting for existing qwen-asr worker at %s …",
            FALLBACK_PORT,
            FALLBACK_BASE_URL,
        )
        if wait_for_existing_worker(FALLBACK_BASE_URL, host=FALLBACK_HOST, port=FALLBACK_PORT):
            model = asr_model or FALLBACK_MODEL
            logger.info("Using existing local qwen-asr worker at %s", FALLBACK_BASE_URL)
            return ResolvedASR(
                base_url=FALLBACK_BASE_URL,
                model=model,
                provider_label="qwen-transformers-local",
                source="fallback-existing",
            )
        raise RuntimeError(
            f"Port {FALLBACK_HOST}:{FALLBACK_PORT} is in use but {FALLBACK_BASE_URL} "
            "did not respond. Stop the stale worker on that port and retry."
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
