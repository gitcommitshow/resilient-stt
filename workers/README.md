# ASR worker microservices

The orchestrator only talks to ASR through an OpenAI-compatible
`POST /v1/audio/transcriptions` endpoint. Workers live here as a docs-only
folder in v1; bring your own.

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

## Placeholders

The directories below are intentional placeholders. Implement each as a
separate service that satisfies the contract above.

- `qwen_vllm_service/` — Qwen3-ASR via vLLM (Linux + NVIDIA); optional bootstrap:
  `python scripts/bootstrap_vllm_qwen3_asr.py --install` (see
  [qwen_vllm_service/README.md](qwen_vllm_service/README.md)).
- `qwen_transformers_service/` — qwen-asr on CPU/MPS; auto-started by the orchestrator
  or manual `python scripts/bootstrap_qwen_asr_fallback.py` (see
  [qwen_transformers_service/README.md](qwen_transformers_service/README.md)).
- `parakeet_openai_service/` — Wrap NeMo/Parakeet behind the OpenAI shape.
- `whisper_openai_service/` — Wrap faster-whisper / whisper.cpp behind the OpenAI shape.
