# ASR worker microservices

The orchestrator only talks to ASR through an OpenAI-compatible
`POST /v1/audio/transcriptions` endpoint. One worker is bundled; others are
documented contracts you can implement separately.

## Contract

Request (multipart/form-data):

| Field | Required | Notes |
|-------|----------|-------|
| `file` | yes | Audio chunk (16 kHz mono WAV is what the orchestrator sends) |
| `model` | yes | Model identifier |
| `language` | no | BCP-47 hint, e.g. `hi`, `en` |
| `prompt` | no | Optional decoding bias |
| `response_format` | no | `verbose_json` is preferred; `json` is acceptable |
| `timestamp_granularities[]` | no | Repeated values; the orchestrator sends `segment` and `word` |

Response (JSON):

```json
{
  "text": "haan let's start karte hain",
  "language": "hi",
  "segments": [
    {"start": 0.0, "end": 1.5, "text": "haan let's start", "confidence": 0.91}
  ],
  "words": [
    {"start": 0.0, "end": 0.4, "word": "haan", "confidence": 0.95}
  ]
}
```

Timestamps in the response must be **local to the audio chunk**; the
orchestrator adds the chunk's global `start_offset`.

## Sanity check with curl

```bash
curl -X POST http://localhost:8001/v1/audio/transcriptions \
  -F file=@chunk.wav \
  -F model=Qwen3-ASR-1.7B \
  -F language=hi \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment' \
  -F 'timestamp_granularities[]=word'
```

## Workers

- **Bundled (in PyPI wheel):** [`src/resilient_stt/workers/qwen_transformers_service/`](../src/resilient_stt/workers/qwen_transformers_service/README.md) — qwen-asr on CPU/MPS; auto-started by the orchestrator or via `python scripts/bootstrap_qwen_asr_fallback.py` from a git checkout.
- **Documented only (under repo `workers/`):**
  - [`qwen_vllm_service/`](qwen_vllm_service/README.md) — Qwen3-ASR via vLLM (Linux + NVIDIA); bootstrap: `python scripts/bootstrap_vllm_qwen3_asr.py --install`
  - [`parakeet_openai_service/`](parakeet_openai_service/README.md) — Placeholder: wrap NeMo/Parakeet behind the OpenAI shape
  - [`whisper_openai_service/`](whisper_openai_service/README.md) — Placeholder: wrap faster-whisper / whisper.cpp behind the OpenAI shape
