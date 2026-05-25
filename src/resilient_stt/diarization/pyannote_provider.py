"""Run pyannote.audio diarization on the full normalized audio.

Lazy-imports torch and pyannote so the orchestrator can run end-to-end (with
`--skip-diarization`) on lean environments without the GPU stack installed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import soundfile as sf

from resilient_stt.core.privacy import disable_dependency_telemetry, disable_pyannote_session_telemetry
from resilient_stt.core.schema import DiarizationTurn

disable_dependency_telemetry()

DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


def load_pyannote_audio(path: str | Path) -> dict[str, Any]:
    """Load WAV as a pyannote pipeline input dict (avoids torchcodec/AudioDecoder)."""

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    import torch  # noqa: WPS433 - lazy import; diarization extra only

    # soundfile: (frames, channels) -> pyannote: (channels, frames)
    waveform = torch.from_numpy(data.T.copy())
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


class PyannoteDiarizationProvider:
    """Thin wrapper around `pyannote.audio.Pipeline` for full-file diarization."""

    def __init__(
        self,
        model: str = DEFAULT_PYANNOTE_MODEL,
        hf_token: str | None = None,
        local_path: str | Path | None = None,
        use_exclusive: bool = True,
        device: str | None = None,
    ) -> None:
        self.model = str(local_path) if local_path else model
        self.hf_token = hf_token
        self.use_exclusive = use_exclusive
        self.device = device
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        # pyannote imports torchcodec at load time; we pass preloaded waveforms instead.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*torchcodec is not installed correctly.*",
                category=UserWarning,
            )
            from pyannote.audio import Pipeline  # noqa: WPS433 - intentional lazy import

        disable_pyannote_session_telemetry()

        if Path(self.model).exists():
            pipeline = Pipeline.from_pretrained(self.model)
        else:
            pipeline = Pipeline.from_pretrained(self.model, token=self.hf_token)

        if self.device:
            import torch  # noqa: WPS433 - lazy import

            pipeline.to(torch.device(self.device))
        self._pipeline = pipeline
        return pipeline

    def diarize(
        self,
        audio_path: str | Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[DiarizationTurn]:
        """Run diarization and return non-overlapping speaker turns when possible."""

        pipeline = self._load()
        kwargs: dict[str, int] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        output = pipeline(load_pyannote_audio(audio_path), **kwargs)

        annotation = None
        if self.use_exclusive:
            annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", None)
        if annotation is None:
            annotation = output  # Community-1 may return Annotation directly

        turns: list[DiarizationTurn] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(
                    speaker=str(speaker),
                    start=float(turn.start),
                    end=float(turn.end),
                )
            )
        turns.sort(key=lambda t: (t.start, t.end))
        return turns
