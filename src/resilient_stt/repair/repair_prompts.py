"""Prompts used by the LLM transcript repair stage."""

SYSTEM_PROMPT = """You are a transcript correction engine for English, Hindi, and Hinglish speech.

Fix only clear ASR errors.
Preserve the original meaning.
Do not add new facts.
Do not remove speaker intent.
Preserve Hinglish naturally.
Keep Hindi either in Latin script or Devanagari based on the input style.
Preserve timestamps and speaker labels exactly.
Return valid JSON only.

Each segment has:
speaker, start, end, text

Only modify the text field.
Respond with a JSON object: {"segments": [{"speaker": ..., "start": ..., "end": ..., "text": ...}, ...]}.
The number of segments must match the input and items must stay in the same order."""


PASS1_INSTRUCTION = (
    "Pass 1: focus on punctuation, casing, and obvious spelling fixes. "
    "Leave anything ambiguous unchanged."
)

PASS2_INSTRUCTION = (
    "Pass 2: re-examine only the highlighted low-confidence segments. "
    "Improve clearly mis-recognized words; do not rewrite confident text."
)


def build_user_prompt(segments_payload: list[dict], instruction: str) -> str:
    """Render a user prompt around a JSON segment list and a per-pass instruction."""

    import json

    return (
        f"{instruction}\n\n"
        f"Input segments (JSON):\n{json.dumps(segments_payload, ensure_ascii=False)}\n\n"
        'Return JSON in the shape {"segments": [...]}.'
    )
