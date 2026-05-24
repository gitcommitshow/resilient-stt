# qwen_vllm_service

Optional local ASR via [vLLM Qwen3-ASR](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html).
The orchestrator calls `POST /v1/audio/transcriptions` on whatever you start here.

## Quick start (bootstrap script)

From the repo root, with **Python 3.11+** and preferably [uv](https://docs.astral.sh/uv/) on PATH:

```text
python scripts/bootstrap_vllm_qwen3_asr.py --install
```

This creates `workers/qwen_vllm_service/.venv`, installs `vllm[audio]` (CUDA nightly wheels when `nvidia-smi` is present), and runs:

```text
vllm serve Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port 8001
```

Defaults match the main README (`--asr-endpoint http://localhost:8001/v1`).

### Useful flags

| Flag | Effect |
|------|--------|
| `--check-only` | Exit 0 if `http://127.0.0.1:8001/v1` already responds |
| `--install-only` | Install into the isolated venv and exit |
| `--force` | Install/serve even if the endpoint is already up |
| `--cuda-nightly` / `--no-cuda-nightly` | Override CUDA wheel selection |
| `--port` / `--host` / `--model` | Override serve defaults |

Install only, then serve manually:

```text
python scripts/bootstrap_vllm_qwen3_asr.py --install-only
workers/qwen_vllm_service/.venv/bin/vllm serve Qwen/Qwen3-ASR-1.7B --port 8001
```

## Platform notes

- **Linux + NVIDIA GPU** is the supported path for vLLM Qwen3-ASR. The bootstrap script selects [nightly CUDA 12.9 wheels](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html) when `nvidia-smi` is available.
- **macOS (including Apple Silicon)** installs generic `vllm` from PyPI, which often resolves to **0.11.x** and logs `Automatically detected platform cpu`. That build has **no `qwen3_asr` model** — `vllm serve Qwen/Qwen3-ASR-1.7B` will crash during startup. The bootstrap script now **refuses to start** on Mac unless you pass `--allow-unsupported-platform`.
- On Mac, use a **hosted** OpenAI-compatible ASR API, run vLLM on a **Linux GPU box**, or implement another worker (e.g. faster-whisper) under `workers/`.
- vLLM is installed in a **separate venv** so it does not conflict with the orchestrator’s `pyannote` / `torch` stack.

If you previously installed on Mac, remove the bad venv and do not reuse it on Linux without reinstalling:

```text
rm -rf workers/qwen_vllm_service/.venv
```

## Contract

See [workers/README.md](../README.md). The orchestrator sends 16 kHz mono WAV chunks and expects OpenAI-style `verbose_json` timestamps when possible.
