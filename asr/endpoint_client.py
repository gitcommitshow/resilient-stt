"""ASR providers. The orchestrator only talks to ASR through this abstraction."""

from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import Any

import httpx

from core.schema import ASRResult

from .normalizer import normalize_response
from .openai_request import build_transcription_fields


class ASRProvider(abc.ABC):
    """Minimal contract every ASR backend must satisfy."""

    name: str = "asr-provider"

    @abc.abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        chunk_id: str | None = None,
        start_offset: float = 0.0,
    ) -> ASRResult:
        ...


class OpenAICompatibleASRProvider(ASRProvider):
    """Talks to any service exposing `POST /v1/audio/transcriptions`."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        provider_label: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/audio/transcriptions"):
            self.base_url = self.base_url[: -len("/audio/transcriptions")]
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.provider_label = provider_label or self.name

    def _endpoint(self) -> str:
        return f"{self.base_url}/audio/transcriptions"

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _build_multipart(self, audio_path: Path, fields: list[tuple[str, str]]) -> list[tuple[str, Any]]:
        """Build a single httpx ``files`` list for multipart/form-data (OpenAI ASR shape)."""
        handle = audio_path.open("rb")
        parts: list[tuple[str, Any]] = [
            ("file", (audio_path.name, handle, "audio/wav")),
        ]
        for key, value in fields:
            parts.append((key, (None, value)))
        return parts

    def _post(self, audio_path: Path, fields: list[tuple[str, str]]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            handle = None
            try:
                multipart = self._build_multipart(audio_path, fields)
                handle = multipart[0][1][1]
                response = httpx.post(
                    self._endpoint(),
                    headers=self._headers(),
                    files=multipart,
                    timeout=self.timeout,
                )
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"ASR server error {response.status_code}: {response.text[:500]}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"ASR client error {response.status_code}: {response.text[:500]}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_sec * attempt)
            finally:
                if handle is not None:
                    handle.close()
        raise RuntimeError(f"ASR request failed after {self.max_retries} attempts: {last_exc}")

    def transcribe(
        self,
        audio_path: str | Path,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        chunk_id: str | None = None,
        start_offset: float = 0.0,
    ) -> ASRResult:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Form fields depend on model capabilities (whisper-1 vs gpt-4o-transcribe, etc.).
        fields = build_transcription_fields(
            model,
            language=language,
            prompt=prompt,
        )

        raw = self._post(path, fields)
        return normalize_response(
            raw,
            provider=self.provider_label,
            model=model,
            chunk_id=chunk_id or path.stem,
            start_offset=start_offset,
            language=language,
        )
