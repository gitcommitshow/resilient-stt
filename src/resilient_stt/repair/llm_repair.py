"""Two-pass LLM transcript repair over overlapping windows of segments.

Calls an OpenAI-compatible `POST /chat/completions` endpoint. The LLM is only
allowed to modify the `text` field; everything else is enforced by
`validate_repair`. Output segments retain `raw_text` and gain a `clean_text`
when repair succeeded.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import httpx

from resilient_stt.core.schema import TranscriptSegment

from .repair_prompts import (
    PASS1_INSTRUCTION,
    PASS2_INSTRUCTION,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from .repair_validation import validate_repair


DEFAULT_PASS1_WINDOW = 20
DEFAULT_PASS1_STRIDE = 18
DEFAULT_PASS2_CONFIDENCE = 0.6


class OpenAICompatibleRepairClient:
    """Minimal client for chat-completions style endpoints (OpenAI, vLLM, Ollama)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        self.base_url = base
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(self, system: str, user: str) -> str:
        """Send a single chat completion request and return the assistant content."""

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"LLM repair request failed: {exc}") from exc
        raise RuntimeError(f"LLM repair request failed: {last_exc}")


def _segment_to_payload(seg: TranscriptSegment) -> dict[str, Any]:
    return {
        "speaker": seg.speaker,
        "start": seg.start,
        "end": seg.end,
        "text": seg.raw_text,
    }


def _parse_segments(content: str) -> list[dict[str, Any]] | None:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        items = data.get("segments")
    elif isinstance(data, list):
        items = data
    else:
        return None
    if not isinstance(items, list):
        return None
    return items


def _windows(total: int, size: int, stride: int) -> Iterable[tuple[int, int]]:
    if total <= 0:
        return
    start = 0
    while start < total:
        end = min(total, start + size)
        yield start, end
        if end >= total:
            return
        start += stride


def _accept_window_indices(total: int, window_start: int, window_end: int, stride: int) -> range:
    """Center-only acceptance: skip the trailing overlap so adjacent windows don't fight."""

    if window_end >= total:
        return range(window_start, window_end)
    return range(window_start, window_start + stride)


def repair_pass(
    client: OpenAICompatibleRepairClient,
    segments: list[TranscriptSegment],
    *,
    instruction: str,
    window: int,
    stride: int,
    indices: list[int] | None = None,
) -> list[TranscriptSegment]:
    """Run one repair pass and return updated segments (out-of-place)."""

    updated = [seg.model_copy(deep=True) for seg in segments]
    target_indices = set(indices) if indices is not None else None

    for win_start, win_end in _windows(len(updated), window, stride):
        window_segments = updated[win_start:win_end]
        payload = [_segment_to_payload(s) for s in window_segments]
        try:
            content = client.complete(SYSTEM_PROMPT, build_user_prompt(payload, instruction))
        except Exception:
            continue
        parsed = _parse_segments(content)
        if not parsed or len(parsed) != len(window_segments):
            continue

        accept = _accept_window_indices(len(updated), win_start, win_end, stride)
        for offset, repaired in enumerate(parsed):
            absolute_index = win_start + offset
            if absolute_index not in accept:
                continue
            if target_indices is not None and absolute_index not in target_indices:
                continue

            original = updated[absolute_index]
            if not validate_repair(original, repaired):
                updated[absolute_index] = original.model_copy(
                    update={"repair_status": "failed" if original.repair_status == "raw" else original.repair_status}
                )
                continue

            new_text = (repaired["text"] or "").strip()
            if not new_text or new_text == (original.raw_text or "").strip():
                updated[absolute_index] = original.model_copy(
                    update={"repair_status": "unchanged"}
                )
                continue

            updated[absolute_index] = original.model_copy(
                update={"clean_text": new_text, "repair_status": "corrected"}
            )

    return updated


def repair_transcript(
    client: OpenAICompatibleRepairClient,
    segments: list[TranscriptSegment],
    *,
    pass1_window: int = DEFAULT_PASS1_WINDOW,
    pass1_stride: int = DEFAULT_PASS1_STRIDE,
    pass2_confidence: float = DEFAULT_PASS2_CONFIDENCE,
) -> list[TranscriptSegment]:
    """Run pass 1 across all segments, then pass 2 on low-confidence segments."""

    if not segments:
        return segments

    after_pass1 = repair_pass(
        client,
        segments,
        instruction=PASS1_INSTRUCTION,
        window=pass1_window,
        stride=pass1_stride,
    )

    low_conf_indices = [
        i for i, seg in enumerate(after_pass1)
        if (seg.confidence is not None and seg.confidence < pass2_confidence)
    ]
    if not low_conf_indices:
        return after_pass1

    return repair_pass(
        client,
        after_pass1,
        instruction=PASS2_INSTRUCTION,
        window=pass1_window,
        stride=pass1_stride,
        indices=low_conf_indices,
    )
