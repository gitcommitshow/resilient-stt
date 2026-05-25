# Resilient STT (Speech-To-Text)

A local speech transcription pipeline for English, Hindi, and Hinglish audio.  
The orchestrator owns chunking, diarization, repair, and exports; ASR inference is **externalized** behind an OpenAI-compatible `/v1/audio/transcriptions` endpoint that you choose (vLLM, hosted APIs including OpenAI and Open Router API, faster-whisper wrapper, Parakeet wrapper, etc.).

> **Experimental:** This project is under active development and is **not production-ready**. Expect breaking changes, incomplete features, and behavior that may shift between releases. Use for evaluation and prototyping only.

**Architecture & design decisions:** [docs/design.md](docs/design.md)

## Architecture

```
Audio input
  -> ffmpeg normalize (mono 16 kHz WAV; optional --enhance-audio)
  -> VAD (Silero if installed, else webrtcvad / RMS; skip silent regions)
  -> chunk speech regions (fixed: 60s/2s overlap; or pause-aligned: ~120s at onsets)
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

## Supported models

The orchestrator is **provider-agnostic**: any service that implements
`POST /v1/audio/transcriptions` (OpenAI multipart shape) or OpenRouter STT
(JSON + base64) works when you set the right **endpoint**, **model id**, and
**API key**. The lists below are **examples**, not an exhaustive allowlist —
use whatever model id your provider documents.

Configure via `--asr-endpoint`, `--model`, and env vars (see
[docs/cli.md](docs/cli.md) and [ASR auto-detection](#asr-auto-detection)).

### ASR — OpenAI API

| Setting | Value |
| -------- | ----- |
| Endpoint | `https://api.openai.com/v1` or omit + `OPENAI_API_KEY` for auto-detect |
| API key | `OPENAI_API_KEY` or `ASR_API_KEY` |
| Default `--model` | `whisper-1` |

Example transcription models on OpenAI ([Audio API](https://platform.openai.com/docs/guides/speech-to-text)):

| Model | Timestamps |
| ----- | ---------- |
| `whisper-1` | Word + segment (`verbose_json`) |
| `gpt-4o-transcribe` | Text only |
| `gpt-4o-mini-transcribe` | Text only |
| `gpt-4o-transcribe-diarize` | Speaker labels in response |

New OpenAI transcription models generally work with `--model` as long as they
use the same endpoint.

### ASR — OpenRouter

| Setting | Value |
| -------- | ----- |
| Endpoint | `https://openrouter.ai/api/v1` or omit + `OPENROUTER_API_KEY` for auto-detect |
| API key | `OPENROUTER_API_KEY` or `ASR_API_KEY` |
| Default `--model` | `openai/whisper-1` |

Example STT models on OpenRouter ([STT docs](https://openrouter.ai/docs/guides/overview/multimodal/stt); browse [speech-to-text models](https://openrouter.ai/collections/speech-to-text-models)):

| Model slug | Notes |
| ---------- | ----- |
| `openai/whisper-1` | Default auto-detect |
| `openai/whisper-large-v3` | Whisper via OpenRouter |
| `google/chirp-3` | Google Chirp |
| `mistralai/voxtral-mini-transcribe` | Mistral Voxtral |

Any OpenRouter model with **transcription** output modality works — pass its slug
to `--model`. Responses are **text-only** (no segment timestamps); the pipeline
sets `weak_timestamps` and may run optional alignment.

### ASR — Local OpenAI-compatible workers

Point `--asr-endpoint` at any local or remote server that speaks the
[worker contract](workers/README.md) (`multipart` upload, `verbose_json` preferred).
The orchestrator sends 16 kHz mono WAV chunks.

| Worker | Endpoint (default) | Example `--model` | Status |
| ------ | ------------------ | ----------------- | ------ |
| [qwen_transformers_service](workers/qwen_transformers_service/README.md) | `http://127.0.0.1:8002/v1` | `Qwen/Qwen3-ASR-0.6B` | **Bundled** — auto-started when no other ASR is reachable |
| [qwen_vllm_service](workers/qwen_vllm_service/README.md) | `http://127.0.0.1:8001/v1` | `Qwen/Qwen3-ASR-1.7B` | **Bundled bootstrap** — Linux + NVIDIA GPU |
| Custom (faster-whisper, vLLM, hosted proxy, etc.) | Your URL | Your model id | **Supported** — implement or deploy separately |
| [whisper_openai_service](workers/whisper_openai_service/README.md) | — | — | **Planned Roadmap** |
| [parakeet_openai_service](workers/parakeet_openai_service/README.md) | — | — | **Planned Roadmap** |

For a third-party server, only the **API shape** must match; the model name is
whatever that server expects.

### Diarization (orchestrator-local)

Not routed through OpenAI-compatible ASR — runs inside the pipeline on the full
normalized file.

| Model | Configure |
| ----- | --------- |
| `pyannote/speaker-diarization-community-1` (default) | `HF_TOKEN` or `--diarization-model-path`; skip with `--skip-diarization` |

### LLM transcript repair

Any OpenAI-compatible **`/chat/completions`** endpoint. Set `REPAIR_BASE_URL`,
`REPAIR_MODEL`, and `REPAIR_API_KEY` (auto-filled from `OPENAI_API_KEY` or
`OPENROUTER_API_KEY`).

| Provider | Default `REPAIR_MODEL` | Example alternatives |
| -------- | ---------------------- | -------------------- |
| OpenAI API | `gpt-4o-mini` | `gpt-4o`, `gpt-4.1-mini`, … |
| OpenRouter | `openai/gpt-4o-mini` | Any chat model slug your key can access |
| Other | — | Any id your endpoint accepts |

### Forced alignment

| Component | Status |
| --------- | ------ |
| Qwen forced aligner (`alignment/qwen_aligner.py`) | **Planned Roadmap** |
| WhisperX | **Planned Roadmap** |

Today, alignment runs only when `--align` is set or ASR returned weak
timestamps; the default aligner is a no-op pass-through.

## Install (uv)

Prerequisites: [uv](https://docs.astral.sh/uv/), **ffmpeg** on PATH. ASR is optional to
configure manually — see [ASR auto-detection](#asr-auto-detection) below.

```bash
cd resilient-stt

# 1) Create a project virtualenv (recommended; uv also creates one on sync if missing)
uv venv

# 2) Install dependencies into .venv
#    --extra full = Silero VAD + diarization + torch (recommended on Apple Silicon / Linux)
uv sync --extra full --extra dev

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

Extras: `silero` (Qwen-aligned VAD), `diarization` (pyannote), `full` (both + torch).
Without `silero`, VAD falls back to webrtcvad.

### Platform notes (torch / diarization)

`pyannote.audio` 4.x requires **torch ≥ 2.8**. On **Apple Silicon** (M1–M4), PyTorch uses the Metal **MPS** backend ([Accelerated PyTorch on Mac](https://developer.apple.com/metal/pytorch/)). Use a **native arm64** Python/venv — not Rosetta x86_64.


| Platform                  | Install                                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Apple Silicon (M1–M4)** | Native arm64 venv, then `uv sync --extra full --extra dev`. Optional GPU: `--diarization-device mps`                     |
| **Linux** or **Windows**  | `uv sync --extra full --extra dev`                                                                                       |
| **Intel Mac (x86_64)**    | Diarization extra is **not supported** (no compatible torch wheels). Use `uv sync --extra dev` and `--skip-diarization`. |


If `uv` errors mention `x86_64` on an M-series Mac, your Python/venv is Rosetta. Recreate it:

```bash
rm -rf .venv uv.lock
uv venv --python 3.12
python -c "import platform; print(platform.machine())"   # must print arm64
uv sync --extra full --extra dev
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

**OpenAI API (`whisper-1`)** — set `OPENAI_API_KEY` in `.env`, then either
auto-detect (no local ASR on `:8001`/`:8002`) or point at the API explicitly:

```bash
# Auto-detect when no local ASR is running (--no-asr-fallback avoids starting qwen-asr)
uv run python -m orchestrator.main \
  --audio data/input/speech.wav \
  --output data/output/openai \
  --no-asr-fallback \
  --skip-diarization

# Explicit endpoint
uv run python -m orchestrator.main \
  --audio data/input/speech.wav \
  --output data/output/openai \
  --asr-endpoint https://api.openai.com/v1 \
  --model whisper-1 \
  --skip-diarization
```

**OpenRouter (`google/chirp-3`)** — set `OPENROUTER_API_KEY` in `.env` or pass
the key via `ASR_API_KEY`:

```bash
uv run python -m orchestrator.main \
  --audio data/input/speech.wav \
  --output data/output/openrouter/chirp \
  --asr-endpoint https://openrouter.ai/api/v1 \
  --model google/chirp-3 \
  --skip-diarization \
  --repair false
```

For music or other non-speech audio, add `--no-vad` so VAD does not skip the
file. Full flag reference: [docs/cli.md](docs/cli.md).

### ASR auto-detection

When `--asr-endpoint` is omitted (and `ASR_BASE_URL` / `ASR_ENDPOINT` are unset):

1. Probe **vLLM** at `http://127.0.0.1:8001/v1` ([optional bootstrap](workers/qwen_vllm_service/README.md))
2. Probe an existing **qwen-asr** worker at `http://127.0.0.1:8002/v1`
3. **OpenRouter** when `OPENROUTER_API_KEY` is set (no `--model`)
4. **OpenAI** when `OPENAI_API_KEY` is set (no `--model`)
5. Otherwise **start** the local qwen-asr fallback ([workers/qwen_transformers_service/README.md](workers/qwen_transformers_service/README.md)) — slow on CPU/MPS but needs no NVIDIA GPU

Use `--no-asr-fallback` to require an explicit or already-running ASR service.

Useful flags:


| Flag                                                   | Effect                                                                  |
| ------------------------------------------------------ | ----------------------------------------------------------------------- |
| `--no-asr-fallback`                                    | Require explicit/running ASR; do not auto-start qwen-asr on :8002       |
| `--no-vad`                                             | Disable VAD; transcribe the entire timeline                             |
| `--vad-backend`                                        | `auto`, `silero`, `webrtcvad`, or `rms` (default `auto`)                |
| `--chunk-mode`                                         | `fixed` (60s/2s overlap) or `pause-aligned` (~120s at onsets, max 180s) |
| `--enhance-audio`                                      | High-pass + denoise + loudnorm during normalize                         |
| `--skip-diarization`                                   | Skip pyannote; export without speaker labels                            |
| `--diarization-model-path PATH`                        | Load a local `git clone` of the pyannote model (offline)                |
| `--num-speakers` / `--min-speakers` / `--max-speakers` | Hint pyannote speaker count                                             |
| `--align`                                              | Force the optional alignment stage even when ASR returned timestamps    |
| `--repair true`                                        | Run the LLM repair stage (needs `REPAIR_BASE_URL`/`REPAIR_MODEL`)       |
| `--resume`                                             | Reuse existing intermediates under `data/work/<job_id>/`                |


## Environment variables

Copy `[.env.example](.env.example)` to `.env` and fill in what you need. The CLI
loads `.env` on startup; shell exports take precedence.


| Variable                        | Purpose                                                                                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `ASR_BASE_URL` / `ASR_ENDPOINT` | Optional fixed ASR base URL (same as `--asr-endpoint`)                                                                                  |
| `ASR_API_KEY`                   | Optional Bearer token for the ASR endpoint                                                                                              |
| `OPENROUTER_API_KEY`            | OpenRouter key; enables hosted ASR/repair presets                                                                                       |
| `OPENAI_API_KEY`                | OpenAI key; enables hosted ASR/repair presets                                                                                           |
| `REPAIR_BASE_URL`               | OpenAI-compatible chat endpoint (e.g. `https://api.openai.com/v1`)                                                                      |
| `REPAIR_MODEL`                  | Repair model id (e.g. `gpt-4o-mini`)                                                                                                    |
| `REPAIR_API_KEY`                | Bearer token for the repair endpoint                                                                                                    |
| `HF_TOKEN`                      | Used only to **download** gated pyannote weights. Skip with `--skip-diarization` or use `--diarization-model-path` after a local clone. |


The default diarization model is
`[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)`
(CC-BY-4.0). Accept the model card terms once on Hugging Face, then either
provide `HF_TOKEN` for the first download or follow the model card's "Offline
use" instructions to clone the repo and point `--diarization-model-path` at
the local copy.

### Privacy / telemetry

On startup the orchestrator disables optional usage metrics from dependencies
(pyannote `[PYANNOTE_METRICS_ENABLED=0](https://github.com/pyannote/pyannote-audio#telemetry)`,
Hugging Face Hub `HF_HUB_DISABLE_TELEMETRY=1`). Shell exports and `.env` values
take precedence — set `PYANNOTE_METRICS_ENABLED=1` to opt back in.

## Intermediate artifacts

Each run materializes everything under `data/work/<job_id>/`:

```
normalized.wav
speech_regions.json    # {regions, speech_onsets_samples}
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