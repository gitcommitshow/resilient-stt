"""Trivial single-provider router; placeholder for future multi-endpoint routing."""

from __future__ import annotations

from .endpoint_client import ASRProvider


class ASRRouter:
    """Picks an `ASRProvider` for a given chunk. v1 always returns the same one."""

    def __init__(self, provider: ASRProvider) -> None:
        self.provider = provider

    def for_chunk(self, chunk_id: str) -> ASRProvider:  # noqa: ARG002 - chunk_id reserved for future routing
        return self.provider
