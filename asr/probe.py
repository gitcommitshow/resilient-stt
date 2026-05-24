"""Health checks for OpenAI-compatible ASR HTTPException endpoints."""

from __future__ import annotations

import urllib.error
import urllib.request


def probe_asr_endpoint(base_url: str, timeout_sec: float = 2.0) -> bool:
    """Return True if ``GET {base_url}/models`` responds like an OpenAI ASR server."""
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
