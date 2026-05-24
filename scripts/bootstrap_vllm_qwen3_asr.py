#!/usr/bin/env python3
"""Optional bootstrap for a local Qwen3-ASR vLLM server (OpenAI-compatible).

Use when no ASR microservice is running. Creates an isolated venv under
``workers/qwen_vllm_service/.venv``, installs vLLM with audio extras, and
starts ``vllm serve``. See the vLLM recipe:
https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Defaults aligned with README examples (--asr-endpoint http://localhost:8001/v1).
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = REPO_ROOT / "workers" / "qwen_vllm_service" / ".venv"
# Qwen3-ASR landed in vLLM well after 0.11.x; recipe assumes recent nightly on CUDA.
MIN_VLLM_VERSION = "0.21.0"


def _base_url(host: str, port: int) -> str:
    """OpenAI-compatible API base URL (no trailing slash on path beyond /v1)."""
    return f"http://{host}:{port}/v1"


def probe_asr_endpoint(base_url: str, timeout_sec: float = 2.0) -> bool:
    """Return True if something responds like an OpenAI-compatible ASR server."""
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _venv_python(venv_dir: Path) -> Path:
    """Resolve the Python executable inside a virtualenv."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_vllm(venv_dir: Path) -> Path:
    """Resolve the vllm CLI inside a virtualenv."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "vllm.exe"
    return venv_dir / "bin" / "vllm"


def has_nvidia_gpu() -> bool:
    """Best-effort check for an NVIDIA GPU suitable for CUDA wheels."""
    return shutil.which("nvidia-smi") is not None


def is_apple_silicon() -> bool:
    """True on macOS arm64 (M-series)."""
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def platform_summary() -> str:
    """Short description of the current machine for error messages."""
    return f"{sys.platform}/{platform.machine()}"


def check_platform_allowed(*, allow_unsupported: bool) -> str | None:
    """Return an error message if this host is unlikely to run Qwen3-ASR via vLLM."""
    if has_nvidia_gpu():
        return None
    if is_apple_silicon() or sys.platform == "darwin":
        return (
            f"macOS ({platform_summary()}) detected without NVIDIA CUDA. "
            "vLLM Qwen3-ASR is intended for Linux + NVIDIA GPU (see the vLLM recipe). "
            "Apple Silicon may use vLLM-Metal for other models, but Qwen3-ASR is not "
            "supported on the CPU backend that pip installs here."
        )
    return (
        f"No NVIDIA GPU detected ({platform_summary()}). "
        "Qwen3-ASR via vLLM expects CUDA nightly wheels on Linux."
    )


def installed_vllm_version(venv_dir: Path) -> str | None:
    """Read the vllm package version from the ASR venv, if present."""
    py = _venv_python(venv_dir)
    if not py.is_file():
        return None
    try:
        out = subprocess.run(
            [str(py), "-c", "import vllm; print(vllm.__version__)"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    """Parse a leading numeric semver prefix into a comparison tuple."""
    parts: list[int] = []
    for piece in version.split(".")[:4]:
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def version_too_old(version: str | None) -> bool:
    """True when the installed vLLM is below MIN_VLLM_VERSION."""
    if not version:
        return True
    return _parse_version_tuple(version) < _parse_version_tuple(MIN_VLLM_VERSION)


def ensure_venv(venv_dir: Path) -> Path:
    """Create a virtualenv at ``venv_dir`` if it does not exist."""
    venv_dir = venv_dir.resolve()
    if venv_dir.exists():
        return _venv_python(venv_dir)
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "venv", str(venv_dir)], check=True, cwd=REPO_ROOT)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    return _venv_python(venv_dir)


def install_vllm(venv_dir: Path, *, use_cuda_nightly: bool) -> None:
    """Install vLLM and audio extras into the dedicated ASR venv."""
    py = _venv_python(venv_dir)
    uv = shutil.which("uv")

    def run_pip(*pip_args: str) -> None:
        if uv:
            subprocess.run(
                [uv, "pip", "install", "-p", str(py), *pip_args],
                check=True,
                cwd=REPO_ROOT,
            )
        else:
            subprocess.run([str(py), "-m", "pip", "install", *pip_args], check=True, cwd=REPO_ROOT)

    if use_cuda_nightly:
        # Nightly CUDA 12.9 wheels per vLLM Qwen3-ASR recipe.
        cuda_args = [
            "-U",
            "vllm",
            "--pre",
            "--extra-index-url",
            "https://wheels.vllm.ai/nightly/cu129",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu129",
        ]
        if uv:
            cuda_args.extend(["--index-strategy", "unsafe-best-match"])
        run_pip(*cuda_args)
        run_pip("vllm[audio]")
    else:
        # PyPI on macOS/CPU often resolves ancient vllm (e.g. 0.11.x) without Qwen3-ASR.
        run_pip("-U", f"vllm[audio]>={MIN_VLLM_VERSION}")


def start_vllm_server(
    venv_dir: Path,
    *,
    model: str,
    host: str,
    port: int,
) -> subprocess.Popen[str]:
    """Launch ``vllm serve`` in the background and return the process handle."""
    vllm = _venv_vllm(venv_dir)
    if not vllm.is_file():
        raise FileNotFoundError(
            f"vllm CLI not found at {vllm}. Run with --install (or --install-only) first."
        )
    cmd = [str(vllm), "serve", model, "--host", host, "--port", str(port)]
    env = os.environ.copy()
    return subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _stream_process_output(proc: subprocess.Popen[str]) -> None:
    """Forward child stdout/stderr to this process (runs in a daemon thread)."""
    if proc.stdout is None:
        return
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()


def wait_for_server(
    base_url: str,
    proc: subprocess.Popen[str],
    *,
    timeout_sec: float,
    poll_sec: float = 2.0,
) -> bool:
    """Poll until the server answers, the child exits, or timeout elapses."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            if proc.stdout:
                proc.stdout.close()
            print(f"\nvLLM exited with code {code} before {base_url} became ready.", file=sys.stderr)
            return False
        if probe_asr_endpoint(base_url):
            return True
        time.sleep(poll_sec)
    return False


def build_parser() -> argparse.ArgumentParser:
    """CLI for optional Qwen3-ASR vLLM bootstrap."""
    p = argparse.ArgumentParser(
        description="Bootstrap a local vLLM Qwen3-ASR server (optional; GPU Linux recommended).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default: {DEFAULT_MODEL}).")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host (default: {DEFAULT_HOST}).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Listen port (default: {DEFAULT_PORT}).")
    p.add_argument(
        "--venv",
        type=Path,
        default=DEFAULT_VENV,
        help=f"Isolated venv path (default: {DEFAULT_VENV.relative_to(REPO_ROOT)}).",
    )
    p.add_argument(
        "--install",
        action="store_true",
        help="Create venv and install vLLM before serving (skipped if vllm already present).",
    )
    p.add_argument("--install-only", action="store_true", help="Install vLLM into the venv and exit.")
    p.add_argument("--no-serve", action="store_true", help="Install only; do not start the server.")
    p.add_argument(
        "--cuda-nightly",
        action="store_true",
        help="Use vLLM nightly CUDA 12.9 wheels (auto-selected when nvidia-smi is found).",
    )
    p.add_argument(
        "--no-cuda-nightly",
        action="store_true",
        help="Force generic pip install even if nvidia-smi is present.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Probe --asr-base-url / derived URL; exit 0 if reachable, 1 otherwise.",
    )
    p.add_argument(
        "--asr-base-url",
        default=None,
        help="Override probe URL (default: http://<host>:<port>/v1).",
    )
    p.add_argument(
        "--wait-sec",
        type=float,
        default=600.0,
        help="Seconds to wait for the server after start (default: 600).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Start install/serve even if the endpoint already responds.",
    )
    p.add_argument(
        "--allow-unsupported-platform",
        action="store_true",
        help="Do not refuse macOS/CPU-only hosts (likely to fail loading Qwen3-ASR).",
    )
    p.add_argument(
        "--reinstall",
        action="store_true",
        help="Re-run pip install even when vllm is already present (e.g. after a bad 0.11.x install).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_url = args.asr_base_url or _base_url(args.host, args.port)
    venv_dir: Path = args.venv

    if args.check_only:
        return 0 if probe_asr_endpoint(base_url) else 1

    platform_err = check_platform_allowed(allow_unsupported=args.allow_unsupported_platform)
    if platform_err:
        print(platform_err, file=sys.stderr)
        print(
            "\nAlternatives: run this script on Linux with an NVIDIA GPU; use a hosted "
            "OpenAI-compatible ASR API; or see workers/README.md for other worker placeholders.",
            file=sys.stderr,
        )
        print("To attempt anyway: pass --allow-unsupported-platform", file=sys.stderr)
        return 2

    if probe_asr_endpoint(base_url) and not args.force:
        print(f"ASR endpoint already reachable at {base_url}")
        print("Orchestrator example:")
        print(
            f"  uv run python -m orchestrator.main --audio <file> --output <dir> "
            f"--asr-endpoint {base_url} --model {args.model}"
        )
        return 0

    need_install = args.install or args.install_only or args.reinstall
    vllm_bin = _venv_vllm(venv_dir)
    if not vllm_bin.is_file():
        need_install = True
    elif not need_install:
        ver = installed_vllm_version(venv_dir)
        if version_too_old(ver):
            print(
                f"Installed vLLM {ver!r} is older than {MIN_VLLM_VERSION} "
                "(no Qwen3-ASR support). Re-run with --reinstall.",
                file=sys.stderr,
            )
            return 2

    if need_install:
        print(f"Creating/using venv at {venv_dir}")
        ensure_venv(venv_dir)
        use_cuda = args.cuda_nightly or (has_nvidia_gpu() and not args.no_cuda_nightly)
        if use_cuda:
            print("Installing vLLM (CUDA nightly wheels) + vllm[audio] …")
        else:
            print("Installing vLLM (vllm[audio]) …")
        install_vllm(venv_dir, use_cuda_nightly=use_cuda)
        ver = installed_vllm_version(venv_dir)
        print(f"Installed vLLM {ver}")
        if version_too_old(ver):
            print(
                f"Could not install vLLM>={MIN_VLLM_VERSION} on {platform_summary()}. "
                "Qwen3-ASR requires Linux + NVIDIA per the vLLM recipe.",
                file=sys.stderr,
            )
            return 2

    if args.install_only or args.no_serve:
        print(f"Install complete. Start manually:\n  {_venv_vllm(venv_dir)} serve {args.model} --host {args.host} --port {args.port}")
        return 0

    ver = installed_vllm_version(venv_dir)
    print(f"Starting vLLM {ver}: {args.model} on {args.host}:{args.port}")
    proc = start_vllm_server(venv_dir, model=args.model, host=args.host, port=args.port)
    threading.Thread(target=_stream_process_output, args=(proc,), daemon=True).start()
    try:
        if not wait_for_server(base_url, proc, timeout_sec=args.wait_sec):
            if proc.poll() is None:
                print(f"Timed out waiting for {base_url}", file=sys.stderr)
            return 1
        print(f"Qwen3-ASR ready at {base_url}")
        print("Orchestrator example:")
        print(
            f"  uv run python -m orchestrator.main --audio <file> --output <dir> "
            f"--asr-endpoint {base_url} --model {args.model}"
        )
        print("Press Ctrl+C to stop the server.")
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
