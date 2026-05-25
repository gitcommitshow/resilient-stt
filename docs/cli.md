# CLI reference

Complete reference for `resilient-stt`, the local speech transcription orchestrator.
For architecture and pipeline stages, see [design.md](design.md). For install and
platform notes, see [README.md](../README.md).

---

## Invocation

```bash
uv run python -m orchestrator.main [OPTIONS]
```

The CLI loads `.env` from the repo root on startup (via `python-dotenv`). Shell
exports take precedence over `.env` values. See [`.env.example`](../.env.example)
for a template.

Exit codes: `0` on success, `1` on pipeline failure (stack trace with `-v`).

---

## Required arguments

| Flag | Description |
|------|-------------|
| `--audio PATH` | Input audio file (any format **ffmpeg** can decode). |
| `--output DIR` | Directory for final exports: `transcript.json`, `transcript.srt`, `transcript.vtt`. |

---

## General

| Flag | Default | Description |
|------|---------|-------------|
| `-v`, `--verbose` | off | Enable DEBUG logging (HTTP traces, full exception stacks). |
| `--work-root DIR` | `data/work` | Root for per-run intermediate artifacts (`<work-root>/<job_id>/`). |
| `--resume` | off | Reuse existing artifacts under the job work dir when present. |

---

## ASR

| Flag | Default | Description |
|------|---------|-------------|
| `--asr-endpoint URL` | auto | OpenAI-compatible ASR base URL (e.g. `http://127.0.0.1:8001/v1`). Same as env `ASR_BASE_URL` / `ASR_ENDPOINT`. |
| `--model ID` | see below | ASR model id passed to the endpoint. |
| `--no-asr-fallback` | off | Do not auto-start the local qwen-asr worker on `:8002` when nothing else is reachable. |
| `--language CODE` | — | Optional BCP-47 language hint (e.g. `en`, `hi`). |
| `--prompt TEXT` | — | Optional ASR prompt for biasing decoding (ignored by OpenRouter STT). |
| `--asr-provider-label NAME` | `external-openai-compatible` | Label stored in transcript metadata; overridden by auto-detection. |

### Default `--model` by endpoint

| Endpoint | Default model |
|----------|---------------|
| vLLM (`:8001`) | `Qwen/Qwen3-ASR-1.7B` |
| Local qwen-asr (`:8002`) | `Qwen/Qwen3-ASR-0.6B` |
| OpenAI (`api.openai.com`) | `whisper-1` |
| OpenRouter (`openrouter.ai`) | `openai/whisper-1` |
| Other OpenAI-compatible | `Qwen/Qwen3-ASR-1.7B` |

### ASR auto-detection order

When `--asr-endpoint` is omitted and `ASR_BASE_URL` / `ASR_ENDPOINT` are unset:

1. **`--asr-endpoint` or `ASR_BASE_URL` / `ASR_ENDPOINT`** — explicit URL (always wins).
2. **vLLM** at `http://127.0.0.1:8001/v1`.
3. **Existing qwen-asr worker** at `http://127.0.0.1:8002/v1`.
4. **OpenRouter** when `OPENROUTER_API_KEY` is set, reachable, and `--model` is omitted.
5. **OpenAI** when `OPENAI_API_KEY` is set, reachable, and `--model` is omitted.
6. **Start local qwen-asr** on `:8002` (unless `--no-asr-fallback`).

Hosted auto-detection (steps 4–5) requires no explicit `--model`, no
`ASR_BASE_URL`, and no reachable local ASR. Local ASR is always preferred over
cloud when it responds.

### Hosted providers

**OpenAI** — multipart `POST /v1/audio/transcriptions`. Set `OPENAI_API_KEY` in
`.env` or pass via `ASR_API_KEY`. Auto-fills repair defaults (`gpt-4o-mini`).

**OpenRouter** — JSON + base64 audio at `POST /api/v1/audio/transcriptions`.
Set `OPENROUTER_API_KEY` or use explicit endpoint:

```bash
uv run python -m orchestrator.main \
  --audio data/input/speech.wav \
  --output data/output/run \
  --asr-endpoint https://openrouter.ai/api/v1 \
  --model openai/whisper-1 \
  --skip-diarization
```

OpenRouter responses are text-only (no segment timestamps). The pipeline sets
`weak_timestamps=True`, which enables the alignment stage automatically. Use
`--align` to force alignment even when timestamps exist.

When both `OPENROUTER_API_KEY` and `OPENAI_API_KEY` are set, OpenRouter presets
and auto-detection take precedence.

---

## Voice activity detection (VAD)

VAD runs on the normalized 16 kHz mono WAV and produces speech regions for
chunking. Music and non-speech audio often yield zero regions — use `--no-vad`
to transcribe the full file.

| Flag | Default | Description |
|------|---------|-------------|
| `--no-vad` | off | Disable VAD; treat the entire normalized file as one speech region. |
| `--vad-backend` | `auto` | `auto`, `silero`, `webrtcvad`, or `rms`. `auto`: silero → webrtcvad → rms. |
| `--vad-pad-ms` | `250` | Pad each detected region by this many milliseconds. |
| `--vad-merge-gap-sec` | `0.5` | Merge regions separated by less than this gap. |
| `--vad-min-speech-sec` | `0.3` | Minimum region length for webrtcvad / rms backends. |
| `--vad-min-speech-ms` | `1500` | Silero: minimum speech segment length. |
| `--vad-min-silence-ms` | `500` | Silero: minimum silence between speech segments. |

When VAD finds no speech, the pipeline logs a warning and exports an empty
transcript (zero ASR chunks).

---

## Chunking

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-mode` | `fixed` | `fixed` or `pause-aligned`. |
| `--chunk-threshold-sec` | `600.0` | Speech longer than this triggers chunking. |
| `--chunk-size-sec` | `60.0` | **fixed:** window size per ASR request. |
| `--chunk-overlap-sec` | `2.0` | **fixed:** overlap between consecutive windows. |
| `--chunk-segment-threshold-sec` | `120.0` | **pause-aligned:** target split interval at speech onsets. |
| `--chunk-max-segment-sec` | `180.0` | **pause-aligned:** hard cap per ASR chunk. |

---

## Audio normalization

| Flag | Default | Description |
|------|---------|-------------|
| `--enhance-audio` | off | Apply high-pass + FFT denoise + loudness normalization during ffmpeg normalize. Useful for fan noise or quiet speech. |

---

## Diarization

Requires the `full` or `diarization` install extra and `HF_TOKEN` (or a local
model clone). Skip for ASR-only smoke tests.

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-diarization` | off | Skip pyannote; export without speaker labels. |
| `--diarization-model` | `pyannote/speaker-diarization-community-1` | Hugging Face model id. |
| `--diarization-model-path PATH` | — | Local clone path for offline use (see model card). |
| `--diarization-device` | auto | Device hint: `cpu`, `cuda`, or `mps`. |
| `--num-speakers N` | — | Exact speaker count hint for pyannote. |
| `--min-speakers N` | — | Lower bound on speaker count. |
| `--max-speakers N` | — | Upper bound on speaker count. |

---

## Alignment & repair

| Flag | Default | Description |
|------|---------|-------------|
| `--align` | off | Force the optional forced-alignment stage. Also runs automatically when any chunk has `weak_timestamps`. |
| `--repair [BOOL]` | see below | Run two-pass LLM transcript repair. |

### `--repair` tri-state

| Invocation | Behavior |
|------------|----------|
| omitted | On when `OPENROUTER_API_KEY` or `OPENAI_API_KEY` presets apply; off otherwise. |
| `--repair` | Enable repair. |
| `--repair true` | Enable repair. |
| `--repair false` | Disable repair even when API keys are set. |

Accepted boolean strings: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`.

Repair requires `REPAIR_BASE_URL` and `REPAIR_MODEL` (auto-filled from OpenRouter
or OpenAI presets). Repair only modifies segment text; timestamps and speaker
labels are validated and preserved.

---

## Environment variables

Loaded from `.env` and the shell. CLI flags override config fields where noted.

| Variable | Purpose |
|----------|---------|
| `ASR_BASE_URL` / `ASR_ENDPOINT` | Fixed ASR base URL (same as `--asr-endpoint`). |
| `ASR_API_KEY` | Bearer token for ASR requests. |
| `OPENROUTER_API_KEY` | OpenRouter key; enables OpenRouter presets and auto-detection. |
| `OPENAI_API_KEY` | OpenAI key; enables OpenAI presets and auto-detection. |
| `REPAIR_BASE_URL` | OpenAI-compatible chat base URL (e.g. `https://api.openai.com/v1`). |
| `REPAIR_MODEL` | Repair model id (e.g. `gpt-4o-mini`, `openai/gpt-4o-mini`). |
| `REPAIR_API_KEY` | Bearer token for repair requests. |
| `HF_TOKEN` | Hugging Face token for downloading gated pyannote weights. |

Key resolution order for ASR/repair tokens: `ASR_API_KEY` / `REPAIR_API_KEY` →
`OPENROUTER_API_KEY` → `OPENAI_API_KEY`.

### Privacy / telemetry

On startup the orchestrator disables optional dependency telemetry by default.
Opt back in via shell or `.env`:

| Variable | Default | Effect |
|----------|---------|--------|
| `PYANNOTE_METRICS_ENABLED` | `0` | Set `1` to re-enable pyannote usage metrics. |
| `HF_HUB_DISABLE_TELEMETRY` | `1` | Set `0` to re-enable Hugging Face Hub telemetry. |

---

## Example commands

**Minimal (local qwen-asr auto-start):**

```bash
uv run python -m orchestrator.main \
  --audio data/input/meeting.mp3 \
  --output data/output/meeting \
  --language hi
```

**vLLM ASR:**

```bash
uv run python -m orchestrator.main \
  --audio data/input/meeting.mp3 \
  --output data/output/meeting \
  --asr-endpoint http://127.0.0.1:8001/v1 \
  --model Qwen/Qwen3-ASR-1.7B \
  --language hi \
  --repair true
```

**OpenRouter ASR (explicit):**

```bash
uv run python -m orchestrator.main \
  --audio data/input/speech.wav \
  --output data/output/openrouter \
  --asr-endpoint https://openrouter.ai/api/v1 \
  --model mistralai/voxtral-mini-transcribe \
  --skip-diarization \
  --repair false
```

**Music / non-speech (skip VAD):**

```bash
uv run python -m orchestrator.main \
  --audio data/input/music.wav \
  --output data/output/music \
  --asr-endpoint https://openrouter.ai/api/v1 \
  --model google/chirp-3 \
  --no-vad \
  --enhance-audio \
  --skip-diarization
```

**Resume after interruption:**

```bash
uv run python -m orchestrator.main \
  --audio data/input/meeting.mp3 \
  --output data/output/meeting \
  --resume
```

---

## Intermediate artifacts

Each run writes under `data/work/<job_id>/` (or `--work-root`):

```
normalized.wav
speech_regions.json       # {regions, speech_onsets_samples}
chunks/                   # per-chunk WAV slices
chunks.json
asr_raw/<chunk_id>.json
asr_normalized.json
diarization.json
speaker_segments_raw.json
speaker_segments_repaired.json   # when --repair is on
```

Final exports land under `--output`.

---

## Programmatic use

The CLI is a thin wrapper around `orchestrator.pipeline.run(JobConfig)`. For
embedding in other tools or services, construct a `JobConfig` and pass an
`ASRProvider` instance directly. See [design.md](design.md) §1.
