"""Health checks for OpenAI-compatible ASR HTTPException endpoints."""

from __future__ import annotations

import urllib.error
import urllib.request


def probe_asr_endpoint(
    base_url: str,
    timeout_sec: float = 2.0,
    api_key: str | None = None,
) -> bool:
    """Return True if ``GET {base_url}/models`` responds like an OpenAI ASR server."""
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
