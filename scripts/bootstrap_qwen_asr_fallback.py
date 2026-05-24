#!/usr/bin/env python3
"""Install and run the local qwen-asr (transformers) OpenAI-compatible worker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.privacy import disable_dependency_telemetry  # noqa: E402

disable_dependency_telemetry()

from asr.fallback_worker import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    install_worker_deps,
    start_fallback_server,
    stop_fallback_server,
)
from asr.probe import probe_asr_endpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """CLI for manual qwen-asr worker bootstrap."""
    p = argparse.ArgumentParser(description="Bootstrap local qwen-asr ASR (CPU/MPS, OpenAI-compatible).")
    p.add_argument("--install-only", action="store_true", help="Create venv and install deps, then exit.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-aligner", action="store_true")
    p.add_argument(
        "--check-only",
        action="store_true",
        help=f"Exit 0 if {DEFAULT_BASE_URL} responds.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_only:
        return 0 if probe_asr_endpoint(DEFAULT_BASE_URL) else 1

    install_worker_deps()
    if args.install_only:
        print(f"Installed. Run: workers/qwen_transformers_service/.venv/bin/python workers/qwen_transformers_service/server.py")
        return 0

    if probe_asr_endpoint(f"http://{args.host}:{args.port}/v1"):
        print(f"Worker already listening on http://{args.host}:{args.port}/v1")
        return 0

    handle = start_fallback_server(
        host=args.host,
        port=args.port,
        model=args.model,
        with_aligner=not args.no_aligner,
    )
    print(f"qwen-asr ready at {handle.base_url} (model={handle.model})")
    print("Press Ctrl+C to stop.")
    try:
        return handle.process.wait()
    except KeyboardInterrupt:
        stop_fallback_server(handle)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
