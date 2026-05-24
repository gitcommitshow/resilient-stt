# qwen_transformers_service

Local **qwen-asr** (transformers backend) on **CPU or Apple MPS**, exposed as OpenAI-compatible
`POST /v1/audio/transcriptions`. Used automatically when no `--asr-endpoint` is set, vLLM on
`:8001` is down, and fallback is allowed (default).

## Manual bootstrap

```text
python scripts/bootstrap_qwen_asr_fallback.py --install-only
python scripts/bootstrap_qwen_asr_fallback.py
```

Creates `workers/qwen_transformers_service/.venv`, installs `qwen-asr` (Apache-2.0), and listens on
`http://127.0.0.1:8002/v1` with **`Qwen/Qwen3-ASR-0.6B`** by default (lighter for CPU/MPS).

## Auto-start from orchestrator

```text
uv run python -m orchestrator.main \
  --audio data/input/meeting.mp3 \
  --output data/output/meeting \
  --language hi
```

Resolution order:

1. `--asr-endpoint` or `ASR_BASE_URL` / `ASR_ENDPOINT` (must be reachable)
2. vLLM probe at `http://127.0.0.1:8001/v1`
3. Existing worker at `http://127.0.0.1:8002/v1`
4. Start this worker (install deps on first run)

Pass `--no-asr-fallback` to disable step 4.

## Notes

- First transcription loads model weights from Hugging Face (set `HF_TOKEN` if rate-limited).
- Forced aligner timestamps are on by default; pass `--no-aligner` to the server for text-only.
- Expect **slow** inference on CPU; MPS is faster on Apple Silicon but still much slower than CUDA vLLM.

See [workers/README.md](../README.md) for the HTTP contract.
