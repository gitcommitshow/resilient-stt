"""OpenAI-compatible HTTP server wrapping qwen-asr (transformers backend)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.privacy import disable_dependency_telemetry  # noqa: E402

disable_dependency_telemetry()

import soundfile as sf
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger("qwen_transformers_service")

# BCP-47 hints from the orchestrator -> qwen-asr language names.
LANGUAGE_HINTS: dict[str, str] = {
    "hi": "Hindi",
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "vi": "Vietnamese",
    "th": "Thai",
    "tr": "Turkish",
    "id": "Indonesian",
    "it": "Italian",
    "nl": "Dutch",
}

FORCED_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
# Upper cap at model init; per-request budget is scaled from audio duration.
DEFAULT_MAX_NEW_TOKENS = 4096
TOKENS_PER_AUDIO_SEC = 12
_model = None
_model_lock = threading.Lock()
_runtime: dict[str, Any] = {}


def pick_device(name: str) -> str:
    """Resolve ``auto`` to MPS on Apple Silicon when available, else CPU."""
    import torch

    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    """Choose a safe dtype for the selected device."""
    import torch

    if device == "cpu":
        return torch.float32
    if device == "mps" and hasattr(torch, "bfloat16"):
        return torch.bfloat16
    return torch.float16


def normalize_language(hint: str | None) -> str | None:
    """Map orchestrator language hints to qwen-asr language names."""
    if not hint:
        return None
    return LANGUAGE_HINTS.get(hint.lower(), hint)


def max_new_tokens_for_duration(duration_sec: float, *, cap: int = DEFAULT_MAX_NEW_TOKENS) -> int:
    """Estimate decoder budget from audio length (~12 tok/s Hindi, with headroom)."""
    if duration_sec <= 0:
        return min(cap, 512)
    return min(cap, max(512, int(duration_sec * TOKENS_PER_AUDIO_SEC) + 128))


def qwen_result_to_openai(result: Any, *, duration_sec: float) -> dict[str, Any]:
    """Convert one qwen-asr transcription result into OpenAI verbose_json shape."""
    text = (result.text or "").strip()
    payload: dict[str, Any] = {
        "text": text,
        "language": getattr(result, "language", None),
        "segments": [],
        "words": [],
    }
    stamps = getattr(result, "time_stamps", None) or []
    words: list[dict[str, Any]] = []
    for item in stamps:
        token = (getattr(item, "text", None) or "").strip()
        start = float(getattr(item, "start_time", 0.0))
        end = float(getattr(item, "end_time", start))
        if not token:
            continue
        words.append({"start": start, "end": end, "word": token})
    if words:
        payload["words"] = words
        payload["segments"] = [
            {
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": text,
            }
        ]
    elif text:
        end = max(duration_sec, 0.0)
        payload["segments"] = [{"start": 0.0, "end": end, "text": text}]
    return payload


def load_model() -> Any:
    """Lazy-load the shared qwen-asr model (thread-safe)."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        import torch
        from qwen_asr import Qwen3ASRModel

        device = _runtime["device"]
        dtype = pick_dtype(device)
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device,
            "max_inference_batch_size": 1,
            "max_new_tokens": _runtime.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
        }
        if _runtime["with_aligner"]:
            kwargs["forced_aligner"] = FORCED_ALIGNER
            kwargs["forced_aligner_kwargs"] = {"dtype": dtype, "device_map": device}
        logger.info("Loading qwen-asr model %s on %s …", _runtime["model"], device)
        _model = Qwen3ASRModel.from_pretrained(_runtime["model"], **kwargs)
        logger.info("Model ready.")
        return _model


async def list_models(_: Request) -> JSONResponse:
    """OpenAI-compatible model list used by ASR health probes."""
    return JSONResponse(
        {
            "object": "list",
            "data": [{"id": _runtime["model"], "object": "model"}],
        }
    )


def _form_text(value: Any) -> str | None:
    """Coerce a Starlette form value to plain text."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text or None


def _transcribe_file(
    audio_path: str,
    *,
    language: str | None,
    align: bool,
    duration_sec: float,
) -> Any:
    """Run blocking qwen-asr inference (off the asyncio event loop)."""
    asr = load_model()
    token_cap = _runtime.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    token_budget = max_new_tokens_for_duration(duration_sec, cap=token_cap)
    asr.max_new_tokens = token_budget
    logger.info(
        "Transcribing %.1fs audio (language=%s, max_new_tokens=%d, aligner=%s) …",
        duration_sec,
        language or "auto",
        token_budget,
        align,
    )
    t0 = time.monotonic()
    with _model_lock:
        results = asr.transcribe(
            audio=audio_path,
            language=language,
            return_time_stamps=align,
        )
    elapsed = time.monotonic() - t0
    text_len = len((results[0].text or "").strip()) if results else 0
    logger.info("Transcription finished in %.1fs (%d chars).", elapsed, text_len)
    return results


async def transcribe(request: Request) -> JSONResponse:
    """Handle ``POST /v1/audio/transcriptions`` (multipart form, OpenAI shape)."""
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "missing required field: file"}, status_code=400)

    model_name = _form_text(form.get("model")) or _runtime["model"]
    language = normalize_language(_form_text(form.get("language")))
    response_format = (_form_text(form.get("response_format")) or "json").lower()
    want_timestamps = response_format == "verbose_json" or bool(form.getlist("timestamp_granularities[]"))

    raw = await upload.read()
    suffix = Path(getattr(upload, "filename", "audio.wav")).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(raw)
        tmp.flush()
        tmp.close()
        try:
            info = sf.info(tmp.name)
            duration_sec = float(info.duration)
        except Exception:
            duration_sec = 0.0

        align = want_timestamps and _runtime["with_aligner"]
        results = await asyncio.to_thread(
            _transcribe_file,
            tmp.name,
            language=language,
            align=align,
            duration_sec=duration_sec,
        )
        if not results:
            return JSONResponse({"text": "", "segments": [], "words": []})
        body = qwen_result_to_openai(results[0], duration_sec=duration_sec)
        body["model"] = model_name
        return JSONResponse(body)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def build_app() -> Starlette:
    """Construct the Starlette application."""
    return Starlette(
        routes=[
            Route("/v1/models", list_models, methods=["GET"]),
            Route("/v1/audio/transcriptions", transcribe, methods=["POST"]),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    """CLI for the standalone qwen-asr HTTP worker."""
    p = argparse.ArgumentParser(description="Local qwen-asr OpenAI-compatible ASR server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--no-aligner", action="store_true", help="Disable forced aligner (no word timestamps).")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Decoder token cap; raise for long audio (default: %(default)s).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _runtime["model"] = args.model
    _runtime["device"] = pick_device(args.device)
    _runtime["with_aligner"] = not args.no_aligner
    _runtime["max_new_tokens"] = args.max_new_tokens
    app = build_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
