"""Command-line entry point: `python -m orchestrator.main ...`."""

from __future__ import annotations

import argparse
import logging
import sys

from core.privacy import disable_dependency_telemetry

from asr.endpoint_client import OpenAICompatibleASRProvider

from .asr_discovery import LOCAL_ASR_TIMEOUT_SEC, resolve_asr
from .config import JobConfig
from .openai_defaults import apply_openai_presets
from .pipeline import run


def _bool_flag(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser. Exposed for tests."""

    p = argparse.ArgumentParser(
        prog="resilient-stt",
        description="Local STT orchestrator with external OpenAI-compatible ASR.",
    )
    p.add_argument("--audio", required=True, help="Path to input audio file.")
    p.add_argument("--output", required=True, help="Output directory for transcript JSON/SRT/VTT.")
    p.add_argument(
        "--asr-endpoint",
        default=None,
        help="OpenAI-compatible ASR base URL (default: auto-detect vLLM :8001 or start local qwen-asr :8002).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="ASR model id (default: Qwen3-ASR-1.7B for vLLM, Qwen3-ASR-0.6B for local fallback).",
    )
    p.add_argument(
        "--no-asr-fallback",
        action="store_true",
        help="Do not auto-start the local qwen-asr (CPU/MPS) worker when no ASR is reachable.",
    )
    p.add_argument("--language", default=None, help="Optional BCP-47 language hint (e.g. `hi`, `en`).")
    p.add_argument("--prompt", default=None, help="Optional ASR prompt for biasing decoding.")
    p.add_argument("--asr-provider-label", default="external-openai-compatible")

    p.add_argument("--no-vad", action="store_true", help="Disable VAD; transcribe entire file.")
    p.add_argument(
        "--vad-backend",
        default="auto",
        choices=["auto", "silero", "webrtcvad", "rms"],
        help="VAD backend (auto: silero if installed, else webrtcvad, else rms).",
    )
    p.add_argument("--vad-pad-ms", type=int, default=250)
    p.add_argument("--vad-merge-gap-sec", type=float, default=0.5)
    p.add_argument("--vad-min-speech-sec", type=float, default=0.3)
    p.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=1500,
        help="Silero: minimum speech segment length (Qwen toolkit default).",
    )
    p.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="Silero: minimum silence between speech segments.",
    )
    p.add_argument(
        "--chunk-mode",
        default="fixed",
        choices=["fixed", "pause-aligned"],
        help="fixed: overlapping windows; pause-aligned: Qwen toolkit speech-onset splits.",
    )
    p.add_argument("--chunk-threshold-sec", type=float, default=600.0)
    p.add_argument("--chunk-size-sec", type=float, default=60.0)
    p.add_argument("--chunk-overlap-sec", type=float, default=2.0)
    p.add_argument(
        "--chunk-segment-threshold-sec",
        type=float,
        default=120.0,
        help="pause-aligned: target split interval (Qwen toolkit default).",
    )
    p.add_argument(
        "--chunk-max-segment-sec",
        type=float,
        default=180.0,
        help="pause-aligned: hard cap per ASR chunk (Qwen toolkit default).",
    )
    p.add_argument(
        "--enhance-audio",
        action="store_true",
        help="High-pass + FFT denoise + loudness norm during normalize (fan/cooler noise, quiet speech).",
    )

    p.add_argument("--skip-diarization", action="store_true")
    p.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1")
    p.add_argument("--diarization-model-path", default=None, help="Local clone path for offline pyannote use.")
    p.add_argument("--diarization-device", default=None, help='e.g. "cpu" or "cuda".')
    p.add_argument("--num-speakers", type=int, default=None)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)

    p.add_argument(
        "--repair",
        type=_bool_flag,
        nargs="?",
        const=True,
        default=None,
        help="Run LLM transcript repair (default: on when OPENAI_API_KEY is set).",
    )
    p.add_argument("--align", action="store_true", help="Force forced-alignment stage.")
    p.add_argument("--resume", action="store_true", help="Reuse existing artifacts when present.")
    p.add_argument("--work-root", default="data/work")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _job_from_args(args: argparse.Namespace) -> JobConfig:
    kwargs: dict = dict(
        audio_path=args.audio,
        output_dir=args.output,
        asr_base_url=args.asr_endpoint,
        asr_model=args.model,
        asr_provider_label=args.asr_provider_label,
        language=args.language,
        prompt=args.prompt,
        enable_vad=not args.no_vad,
        vad_backend=args.vad_backend,
        vad_pad_ms=args.vad_pad_ms,
        vad_merge_gap_sec=args.vad_merge_gap_sec,
        vad_min_speech_sec=args.vad_min_speech_sec,
        vad_min_speech_duration_ms=args.vad_min_speech_ms,
        vad_min_silence_duration_ms=args.vad_min_silence_ms,
        chunk_mode=args.chunk_mode,
        chunk_threshold_sec=args.chunk_threshold_sec,
        chunk_size_sec=args.chunk_size_sec,
        chunk_overlap_sec=args.chunk_overlap_sec,
        chunk_segment_threshold_sec=args.chunk_segment_threshold_sec,
        chunk_max_segment_sec=args.chunk_max_segment_sec,
        enhance_audio=args.enhance_audio,
        skip_diarization=args.skip_diarization,
        diarization_model=args.diarization_model,
        diarization_model_path=args.diarization_model_path,
        diarization_device=args.diarization_device,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        enable_alignment=args.align,
        resume=args.resume,
        work_root=args.work_root,
    )
    if args.repair is not None:
        kwargs["enable_repair"] = args.repair
    job = JobConfig.from_env_and_args(**kwargs)
    apply_openai_presets(job, repair_cli=args.repair)
    return job


def _load_dotenv() -> None:
    """Load `.env` from the repo root when present (see `.env.example`)."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Search upward from cwd; does not override variables already set in the shell.
    load_dotenv(override=False)


def cli(argv: list[str] | None = None) -> int:
    """Entry point usable as a console script or via `python -m orchestrator.main`."""

    _load_dotenv()
    disable_dependency_telemetry()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    job = _job_from_args(args)
    if job.enable_repair and job.repair_model:
        logging.getLogger("resilient_stt.main").info(
            "LLM repair enabled (model=%s, base=%s)",
            job.repair_model,
            job.repair_base_url,
        )
    resolved = resolve_asr(
        asr_endpoint=args.asr_endpoint,
        asr_model=args.model,
        allow_fallback=not args.no_asr_fallback,
    )
    job.asr_base_url = resolved.base_url
    job.asr_model = resolved.model
    job.asr_provider_label = resolved.provider_label
    asr_timeout = (
        LOCAL_ASR_TIMEOUT_SEC
        if resolved.provider_label == "qwen-transformers-local"
        else 600.0
    )
    asr_provider = OpenAICompatibleASRProvider(
        base_url=resolved.base_url,
        api_key=job.asr_api_key,
        provider_label=resolved.provider_label,
        timeout=asr_timeout,
        max_retries=1 if resolved.provider_label == "qwen-transformers-local" else 3,
    )
    if resolved.provider_label == "qwen-transformers-local":
        logging.getLogger("resilient_stt.main").info(
            "Local qwen-asr: HTTP timeout %.0fs, no retries (long clips are slow on CPU/MPS).",
            asr_timeout,
        )
    try:
        run(job, asr_provider=asr_provider)
    except Exception as exc:  # noqa: BLE001 - top-level CLI handler
        logging.error("Pipeline failed: %s", exc, exc_info=args.verbose)
        return 1
    finally:
        resolved.stop_managed()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
