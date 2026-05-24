# resilient-stt

A local speech transcription pipeline for English, Hindi, and Hinglish audio.
The orchestrator owns chunking, diarization, repair, and exports; ASR
inference is **externalized** behind an OpenAI-compatible
`/v1/audio/transcriptions` endpoint that you choose (vLLM, hosted API,
faster-whisper wrapper, Parakeet wrapper, etc.).

**Architecture & design decisions:** [docs/design.md](docs/design.md)

## Architecture

```
Audio input
  -> ffmpeg normalize (mono 16 kHz WAV)
  -> VAD (skip silent regions)
  -> chunk speech regions (60s / 2s overlap when long)
  -> ASR microservice calls (OpenAI-compatible)
  -> normalize + stitch global timestamps
  -> pyannote diarization on the full normalized audio
  -> speaker assignment (word IoU + segment fallback)
  -> optional forced alignment
  -> optional LLM transcript repair (two passes, validated)
  -> export JSON / SRT / VTT
```

The orchestrator never embeds ASR model weights. Point `--asr-endpoint` at any
service that implements `POST /v1/audio/transcriptions` per the OpenAI
contract.

## Install (uv)

Prerequisites: [uv](https://docs.astral.sh/uv/), **ffmpeg** on PATH. ASR is optional to
configure manually — see [ASR auto-detection](#asr-auto-detection) below.

```bash
cd resilient-stt

# 1) Create a project virtualenv (recommended; uv also creates one on sync if missing)
uv venv

# 2) Install dependencies into .venv
uv sync --extra diarization --extra dev

# 3) Configure secrets (optional)
cp .env.example .env
# edit .env — loaded automatically on each run
```

ASR-only install (no torch / pyannote):

```bash
uv venv
uv sync --extra dev
```

Activate the venv if you prefer plain `python` without `uv run`:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Diarization needs `torch` and `pyannote.audio` 4.x (the `diarization` extra).

### Platform notes (torch / diarization)

`pyannote.audio` 4.x requires **torch ≥ 2.8**. On **Apple Silicon** (M1–M4), PyTorch uses the Metal **MPS** backend ([Accelerated PyTorch on Mac](https://developer.apple.com/metal/pytorch/)). Use a **native arm64** Python/venv — not Rosetta x86_64.

| Platform | Install |
|----------|---------|
| **Apple Silicon (M1–M4)** | Native arm64 venv, then `uv sync --extra diarization --extra dev`. Optional GPU: `--diarization-device mps` |
| **Linux** or **Windows** | `uv sync --extra diarization --extra dev` |
| **Intel Mac (x86_64)** | Diarization extra is **not supported** (no compatible torch wheels). Use `uv sync --extra dev` and `--skip-diarization`. |

If `uv` errors mention `x86_64` on an M-series Mac, your Python/venv is Rosetta. Recreate it:

```bash
rm -rf .venv uv.lock
uv venv --python 3.12
python -c "import platform; print(platform.machine())"   # must print arm64
uv sync --extra diarization --extra dev
```

Verify MPS after install:

```bash
uv run python -c "import torch; print('mps', torch.backends.mps.is_available())"
```

## Usage

Minimal run (no ASR setup — auto-starts local **qwen-asr** on CPU/MPS when needed):

```bash
uv run python -m orchestrator.main \
  --audio data/input/meeting.mp3 \
  --output data/output/meeting \
  --language hi
```

With an external ASR service (vLLM, hosted API, etc.):

```bash
uv run python -m orchestrator.main \
  --audio data/input/meeting.mp3 \
  --asr-endpoint http://localhost:8001/v1 \
  --model Qwen/Qwen3-ASR-1.7B \
  --language hi \
  --repair true \
  --output data/output/meeting
```

### ASR auto-detection

When `--asr-endpoint` is omitted (and `ASR_BASE_URL` / `ASR_ENDPOINT` are unset):

1. Probe **vLLM** at `http://127.0.0.1:8001/v1` ([optional bootstrap](workers/qwen_vllm_service/README.md))
2. Probe an existing **qwen-asr** worker at `http://127.0.0.1:8002/v1`
3. Otherwise **start** the local qwen-asr fallback ([workers/qwen_transformers_service/README.md](workers/qwen_transformers_service/README.md)) — slow on CPU/MPS but needs no NVIDIA GPU

Use `--no-asr-fallback` to require an explicit or already-running ASR service.

Useful flags:

| Flag | Effect |
|------|--------|
| `--no-asr-fallback` | Require explicit/running ASR; do not auto-start qwen-asr on :8002 |
| `--no-vad` | Disable VAD; transcribe the entire timeline |
| `--skip-diarization` | Skip pyannote; export without speaker labels |
| `--diarization-model-path PATH` | Load a local `git clone` of the pyannote model (offline) |
| `--align` | Force the optional alignment stage even when ASR returned timestamps |
| `--repair true` | Run the LLM repair stage (needs `REPAIR_BASE_URL`/`REPAIR_MODEL`) |
| `--resume` | Reuse existing intermediates under `data/work/<job_id>/` |

## Environment variables

Copy [`.env.example`](.env.example) to `.env` and fill in what you need. The CLI
loads `.env` on startup; shell exports take precedence.

| Variable | Purpose |
|----------|---------|
| `ASR_BASE_URL` / `ASR_ENDPOINT` | Optional fixed ASR base URL (same as `--asr-endpoint`) |
| `ASR_API_KEY` | Optional Bearer token for the ASR endpoint |
| `REPAIR_BASE_URL` | OpenAI-compatible chat endpoint (e.g. `https://api.openai.com/v1`) |
| `REPAIR_MODEL` | Repair model id (e.g. `gpt-4o-mini`) |
| `REPAIR_API_KEY` | Bearer token for the repair endpoint |
| `HF_TOKEN` | Used only to **download** gated pyannote weights. Skip with `--skip-diarization` or use `--diarization-model-path` after a local clone. |

The default diarization model is
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
(CC-BY-4.0). Accept the model card terms once on Hugging Face, then either
provide `HF_TOKEN` for the first download or follow the model card's "Offline
use" instructions to clone the repo and point `--diarization-model-path` at
the local copy.

### Privacy / telemetry

On startup the orchestrator disables optional usage metrics from dependencies
(pyannote [`PYANNOTE_METRICS_ENABLED=0`](https://github.com/pyannote/pyannote-audio#telemetry),
Hugging Face Hub `HF_HUB_DISABLE_TELEMETRY=1`). Shell exports and `.env` values
take precedence — set `PYANNOTE_METRICS_ENABLED=1` to opt back in.

## Intermediate artifacts

Each run materializes everything under `data/work/<job_id>/`:

```
normalized.wav
speech_regions.json
chunks/                # per-chunk WAV slices
chunks.json
asr_raw/<chunk_id>.json
asr_normalized.json
diarization.json
speaker_segments_raw.json
speaker_segments_repaired.json   # only when --repair is on
```

Final exports land under `--output`: `transcript.json`, `transcript.srt`,
`transcript.vtt`.

## Constraints (what not to do)

- Do not embed ASR models inside the orchestrator process.
- Do not chunk inside the ASR microservice; the orchestrator owns chunking.
- Do not diarize per chunk; pyannote runs on the full normalized audio.
- Do not discard raw ASR responses — they live under `asr_raw/`.
- Do not let LLM repair change timestamps, speaker labels, or segment count.

## Tests

```bash
uv run pytest
```

The included tests use synthetic fixtures and mocked HTTP responses; they do
not invoke ffmpeg, pyannote, or any LLM.
