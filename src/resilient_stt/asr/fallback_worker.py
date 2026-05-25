"""Install and launch the local qwen-asr (transformers) OpenAI-compatible worker."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from resilient_stt.asr.probe import probe_asr_endpoint
from resilient_stt.core.privacy import apply_telemetry_env

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = _PACKAGE_ROOT / "workers" / "qwen_transformers_service"
SERVER_SCRIPT = WORKER_DIR / "server.py"
WORKER_MODULE = "resilient_stt.workers.qwen_transformers_service.server"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8002
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1"
DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
STARTUP_TIMEOUT_SEC = 120.0
BUSY_WORKER_WAIT_SEC = 120.0


def _worker_venv_dir() -> Path:
    """Return a user-writable path for the isolated qwen-asr worker virtualenv."""
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "resilient-stt" / "qwen-transformers-worker" / ".venv"


DEFAULT_VENV = _worker_venv_dir()


def _editable_repo_root() -> Path | None:
    """Repo root when running from a source checkout (``src/`` layout)."""
    candidate = Path(__file__).resolve().parents[3]
    return candidate if (candidate / "pyproject.toml").is_file() else None


def is_tcp_port_open(host: str, port: int, *, timeout_sec: float = 0.5) -> bool:
    """Return True when something is listening on ``host:port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_sec)
        return sock.connect_ex((host, port)) == 0


def wait_for_existing_worker(
    base_url: str = DEFAULT_BASE_URL,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    wait_sec: float = BUSY_WORKER_WAIT_SEC,
    probe_timeout_sec: float = 5.0,
) -> bool:
    """Poll until the fallback worker answers probes or the port goes idle."""
    deadline = time.monotonic() + wait_sec
    while time.monotonic() < deadline:
        if probe_asr_endpoint(base_url, timeout_sec=probe_timeout_sec):
            return True
        if not is_tcp_port_open(host, port):
            return False
        time.sleep(2.0)
    return probe_asr_endpoint(base_url, timeout_sec=probe_timeout_sec)


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
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    py = _venv_python(venv_dir)
    if py.is_file():
        return py
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "venv", str(venv_dir)], check=True, cwd=WORKER_DIR)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, cwd=WORKER_DIR)
    return _venv_python(venv_dir)


def _pip_install_into_worker_venv(py: Path, *args: str) -> None:
    """Install into the worker venv; use ``uv pip`` when uv created the venv (no bundled pip)."""
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "pip", "install", "-p", str(py), *args], check=True, cwd=WORKER_DIR)
    else:
        subprocess.run([str(py), "-m", "pip", "install", *args], check=True, cwd=WORKER_DIR)


def _install_orchestrator_into_worker_venv(py: Path) -> None:
    """Install resilient-stt into the worker venv so ``server`` can import the package."""
    repo_root = _editable_repo_root()
    if repo_root is not None:
        _pip_install_into_worker_venv(py, "-e", str(repo_root))
        return
    try:
        version = importlib.metadata.version("resilient-stt")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "resilient-stt is not installed; cannot provision the qwen-asr worker venv."
        ) from exc
    _pip_install_into_worker_venv(py, f"resilient-stt=={version}")


def install_worker_deps(venv_dir: Path = DEFAULT_VENV) -> None:
    """Install qwen-asr, HTTP stack, and resilient-stt into the worker venv."""
    py = ensure_venv(venv_dir)
    _pip_install_into_worker_venv(
        py,
        "qwen-asr",
        "uvicorn[standard]",
        "starlette",
        "python-multipart",
    )
    _install_orchestrator_into_worker_venv(py)


def worker_deps_installed(venv_dir: Path = DEFAULT_VENV) -> bool:
    """Return True when the worker venv can import qwen-asr and resilient_stt."""
    py = _venv_python(venv_dir)
    if not py.is_file():
        return False
    try:
        subprocess.run(
            [str(py), "-c", "import qwen_asr, starlette, uvicorn, resilient_stt"],
            check=True,
            capture_output=True,
            cwd=WORKER_DIR,
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

    base_url = f"http://{host}:{port}/v1"
    if is_tcp_port_open(host, port):
        raise RuntimeError(
            f"Cannot bind qwen-asr worker to {host}:{port}: address already in use. "
            f"Stop the process on that port or wait for the current job to finish."
        )

    cmd = [
        str(py),
        "-m",
        WORKER_MODULE,
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--device",
        device,
    ]
    if not with_aligner:
        cmd.append("--no-aligner")

    env = os.environ.copy()
    apply_telemetry_env(env)
    proc = subprocess.Popen(
        cmd,
        cwd=WORKER_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
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
