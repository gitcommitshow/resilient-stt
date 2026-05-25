# Resilient STT

**The only Speech-To-Text pipeline you need.**

A provider-agnostic speech-to-text pipeline that plugs into **any OpenAI-compatible ASR endpoint** — OpenAI, OpenRouter, vLLM, faster-whisper, or the bundled Qwen worker — and returns rich, timestamped transcripts with speaker labels and validated metadata. Resilient STT handles everything around inference: audio preprocessing, voice-activity detection, intelligent chunking, pyannote diarization, and LLM-powered transcript repair. Tested on English, Hindi, and Hinglish but it is designed to support all the languages the ASR model of your choice supports.

## Features

- **Universal ASR** — Connect to any OpenAI-compatible endpoint, switching models easily by flag.
- **Automatic Discovery** — Finds local or remote ASR, retries if needed, starts Qwen fallback automatically.
- **Seamless Audio Prep** — Effortless ffmpeg normalization, optional enhancement.
- **Smart Silence Skipping** — Uses Silero, webrtcvad, or RMS to focus only on speech.
- **Chunking & Stitching** — Handles long audio with intelligent segmentation, accurate timestamps.
- **Speaker Diarization** — Word-level speakers with pyannote.
- **LLM Repair** — Optional transcript refinement via your preferred chat endpoint.
- **Versatile Export** — Output to JSON, SRT, and VTT including rich metadata.
- **Lightweight** — Orchestrator only; no model weights required.
- **CLI & Python API** — Use via terminal or integrate in code.

**Architecture & design decisions:** [docs/design.md](docs/design.md)

**CLI reference:** [docs/cli.md](docs/cli.md)

## Quickstart

Prerequisites: **Python 3.11 or 3.12**, **ffmpeg** on PATH.

```bash
pip install "resilient-stt[full]"   # minimal: pip install resilient-stt (no pyannote/Silero)

resilient-stt \
  --audio /path/to/audio.wav \
  --output /path/to/output-dir \
  --language hi

# Output (under --output):
#   transcript.json   — segments, words, speakers, repair metadata
#   transcript.srt    — subtitles
#   transcript.vtt    — WebVTT
#
# Example transcript.json (truncated):
# {
#   "audio_file": "/path/to/audio.wav",
#   "duration": 142.5,
#   "language": "hi",
#   "asr_provider": "qwen-asr-fallback",
#   "asr_model": "Qwen/Qwen3-ASR-0.6B",
#   "segments": [{
#     "speaker": "SPEAKER_00",
#     "start": 0.12,
#     "end": 4.85,
#     "raw_text": "Namaste, aaj hum meeting shuru karte hain.",
#     "clean_text": "Namaste, aaj hum meeting shuru karte hain.",
#     "repair_status": "unchanged",
#     "words": [{ "word": "Namaste,", "start": 0.12, "end": 0.58, "speaker": "SPEAKER_00" }]
#   }]
# }
```

No ASR setup required — the CLI auto-detects local workers and hosted APIs, or starts a bundled Qwen worker. Quick smoke test without pyannote or repair: add `--skip-diarization --repair false`. For OpenAI or OpenRouter, set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in `.env` and pass `--no-asr-fallback`. Full flags: [docs/cli.md](docs/cli.md). Repository layout: [docs/design.md](docs/design.md#4-repository-layout).


> **Note**
> 
> 🚧 Active development – not production-ready.  
> Expect changes, incomplete features, and ongoing improvements.

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
| [qwen_transformers_service](src/resilient_stt/workers/qwen_transformers_service/README.md) | `http://127.0.0.1:8002/v1` | `Qwen/Qwen3-ASR-0.6B` | **Bundled** — auto-started when no other ASR is reachable |
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
| Qwen forced aligner (`src/resilient_stt/alignment/qwen_aligner.py`) | **Planned Roadmap** |
| WhisperX | **Planned Roadmap** |

Today, alignment runs only when `--align` is set or ASR returned weak
timestamps; the default aligner is a no-op pass-through.

## Install (PyPI)

```bash
# Minimal orchestrator (webrtcvad VAD; ASR via API or bundled qwen worker)
pip install resilient-stt

# Recommended on Apple Silicon / Linux (Silero VAD + pyannote diarization + torch)
pip install "resilient-stt[full]"

# Contributors
pip install "resilient-stt[full,dev]"
```

Verify the CLI:

```bash
resilient-stt --help
```

### Usage after `pip install`

Run from any directory (creates `data/work/` under the current working dir unless you pass `--work-root`):

```bash
resilient-stt \
  --audio /path/to/audio.wav \
  --output /path/to/output-dir \
  --language hi
```

With diarization and Silero VAD you need the `[full]` extra and usually `HF_TOKEN` in `.env` (see [Environment variables](#environment-variables)). Quick smoke test without pyannote:

```bash
resilient-stt \
  --audio /path/to/audio.wav \
  --output /path/to/output-dir \
  --language hi \
  --skip-diarization \
  --repair false
```

Hosted ASR (no local qwen worker) — set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in `.env`:

```bash
resilient-stt \
  --audio /path/to/audio.wav \
  --output /path/to/output-dir \
  --no-asr-fallback \
  --skip-diarization
```

**Bundled qwen-asr worker:** On first auto-start, the CLI creates an isolated venv at
`~/.cache/resilient-stt/qwen-transformers-worker/.venv` (requires network to download
`qwen-asr` and model weights). If inference fails on Apple Silicon, start the worker
manually with `python scripts/bootstrap_qwen_asr_fallback.py --no-aligner` (from a git
checkout) or see the [bundled worker README](src/resilient_stt/workers/qwen_transformers_service/README.md).

Platform notes for `[full]` match [Platform notes (torch / diarization)](#platform-notes-torch--diarization) below (Intel Mac: base install + `--skip-diarization`).

## Install from source (uv)

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

From a **git checkout**, prefix commands with `uv run` (or activate `.venv` and use `resilient-stt` directly). After **`pip install`**, use `resilient-stt` only.

Minimal run (no ASR setup — auto-starts local **qwen-asr** on CPU/MPS when needed):

```bash
uv run resilient-stt \
  --audio data/input/meeting.mp3 \
  --output data/output/meeting \
  --language hi
```

With an external ASR service (vLLM, hosted API, etc.) — the endpoint must already be running and respond to `GET {base}/v1/models`:

```bash
uv run resilient-stt \
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
uv run resilient-stt \
  --audio data/input/speech.wav \
  --output data/output/openai \
  --no-asr-fallback \
  --skip-diarization

# Explicit endpoint
uv run resilient-stt \
  --audio data/input/speech.wav \
  --output data/output/openai \
  --asr-endpoint https://api.openai.com/v1 \
  --model whisper-1 \
  --skip-diarization
```

**OpenRouter (`google/chirp-3`)** — set `OPENROUTER_API_KEY` in `.env` or pass
the key via `ASR_API_KEY`:

```bash
uv run resilient-stt \
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
5. Otherwise **start** the local qwen-asr fallback ([bundled worker](src/resilient_stt/workers/qwen_transformers_service/README.md)) — slow on CPU/MPS but needs no NVIDIA GPU

Use `--no-asr-fallback` to require an explicit or already-running ASR service.

If you pass `--asr-endpoint` explicitly, that URL must respond to `GET …/v1/models` before the pipeline runs (otherwise the CLI exits with “endpoint configured but unreachable”).

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

From source:

```bash
uv sync --extra dev
uv run pytest
```

After `pip install "resilient-stt[dev]"` from a checkout (with `src/` on `PYTHONPATH`) or when developing in the repo, the same `pytest` command applies if `src` is configured in `pyproject.toml` (`pythonpath = ["src"]`).

The included tests use synthetic fixtures and mocked HTTP responses; they do
not invoke ffmpeg, pyannote, or any LLM.

## Publishing (maintainers)

1. **Commits on `main`** — use [Conventional Commits](https://www.conventionalcommits.org/) in PR titles or squash messages (`feat:`, `fix:`, `chore:`, etc.).
2. **release-please** (`.github/workflows/release-please.yml`) — opens/updates a **Release PR** that bumps `pyproject.toml`, `CHANGELOG.md`, and `.release-please-manifest.json`.
3. **Ship** — merge the Release PR; release-please creates GitHub Release + tag `vX.Y.Z`.
4. **PyPI** (`.github/workflows/publish.yml`) — runs on `release: created` when the tag matches `version` in `pyproject.toml`.

**Retry a missed publish** — re-run the failed **Publish to PyPI** workflow from Actions (the release event is preserved).

One-time: configure a [PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/) for workflow `publish.yml` on repo `gitcommitshow/resilient-stt`. In the org/repo settings, allow GitHub Actions to create and approve pull requests if Release PRs do not appear.

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

This project is licensed under the [GNU General Public License v3.0 or
later](LICENSE) (GPL-3.0-or-later). See [LICENSE](LICENSE) for the full text.