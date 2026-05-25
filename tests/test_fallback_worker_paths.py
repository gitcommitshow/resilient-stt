"""Fallback worker resolves bundled server script inside the installed package layout."""

from pathlib import Path

from resilient_stt.asr.fallback_worker import SERVER_SCRIPT, WORKER_DIR, _worker_venv_dir


def test_worker_script_exists_in_package() -> None:
    """Bundled qwen-asr server ships next to the asr package under resilient_stt."""
    assert SERVER_SCRIPT == WORKER_DIR / "server.py"
    assert SERVER_SCRIPT.is_file()


def test_worker_venv_uses_user_cache() -> None:
    """Worker venv is not created under site-packages."""
    venv = _worker_venv_dir()
    assert "resilient-stt" in str(venv)
    assert "site-packages" not in str(venv).lower()
    assert venv.name == ".venv"
