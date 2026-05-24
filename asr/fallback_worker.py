"""Install and launch the local qwen-asr (transformers) OpenAI-compatible worker."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from asr.probe import probe_asr_endpoint
from core.privacy import apply_telemetry_env

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = REPO_ROOT / "workers" / "qwen_transformers_service"
DEFAULT_VENV = WORKER_DIR / ".venv"
SERVER_SCRIPT = WORKER_DIR / "server.py"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8002
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1"
DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
STARTUP_TIMEOUT_SEC = 120.0


@dataclass
class FallbackServerHandle:
    """A locally spawned qwen-asr HTTP worker."""

    base_url: str
    model: str
    process: subprocess.Popen[str]


def _venv_python(venv_dir: Path) -> Path:
    """Resolve the Python executable inside the worker virtualenv."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv(venv_dir: Path = DEFAULT_VENV) -> Path:
    """Create the worker virtualenv when missing."""
    venv_dir = venv_dir.resolve()
    py = _venv_python(venv_dir)
    if py.is_file():
        return py
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "venv", str(venv_dir)], check=True, cwd=REPO_ROOT)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    return _venv_python(venv_dir)


def install_worker_deps(venv_dir: Path = DEFAULT_VENV) -> None:
    """Install qwen-asr and the minimal HTTP stack into the worker venv."""
    py = ensure_venv(venv_dir)
    uv = shutil.which("uv")
    packages = ["qwen-asr", "uvicorn[standard]", "starlette", "python-multipart"]
    if uv:
        subprocess.run(
            [uv, "pip", "install", "-p", str(py), *packages],
            check=True,
            cwd=REPO_ROOT,
        )
    else:
        subprocess.run([str(py), "-m", "pip", "install", *packages], check=True, cwd=REPO_ROOT)


def worker_deps_installed(venv_dir: Path = DEFAULT_VENV) -> bool:
    """Return True when the worker venv can import qwen-asr and the HTTP stack."""
    py = _venv_python(venv_dir)
    if not py.is_file():
        return False
    try:
        subprocess.run(
            [str(py), "-c", "import qwen_asr, starlette, uvicorn"],
            check=True,
            capture_output=True,
            cwd=REPO_ROOT,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def start_fallback_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model: str = DEFAULT_MODEL,
    venv_dir: Path = DEFAULT_VENV,
    with_aligner: bool = True,
    device: str = "auto",
) -> FallbackServerHandle:
    """Spawn the qwen-asr HTTP worker and wait until ``/v1/models`` responds."""
    py = ensure_venv(venv_dir)
    if not SERVER_SCRIPT.is_file():
        raise FileNotFoundError(f"Worker entrypoint missing: {SERVER_SCRIPT}")

    cmd = [
        str(py),
        str(SERVER_SCRIPT),
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--device",
        device,
    ]
    if with_aligner:
        pass  # aligner enabled by default; no flag needed
    else:
        cmd.append("--no-aligner")

    env = os.environ.copy()
    apply_telemetry_env(env)
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    base_url = f"http://{host}:{port}/v1"
    deadline = time.monotonic() + STARTUP_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = ""
            if proc.stdout:
                tail = proc.stdout.read() or ""
            raise RuntimeError(
                f"Local qwen-asr worker exited before becoming ready (code {proc.returncode}).\n{tail[-4000:]}"
            )
        if probe_asr_endpoint(base_url):
            return FallbackServerHandle(base_url=base_url, model=model, process=proc)
        time.sleep(1.0)
    proc.terminate()
    raise TimeoutError(f"Timed out waiting for local qwen-asr worker at {base_url}")


def stop_fallback_server(handle: FallbackServerHandle | None) -> None:
    """Terminate a worker started by ``start_fallback_server``."""
    if handle is None or handle.process.poll() is not None:
        return
    handle.process.terminate()
    try:
        handle.process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        handle.process.kill()
