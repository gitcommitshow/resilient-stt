"""Runtime configuration for a single transcription job.

`JobConfig` is built from CLI args + environment variables in `main.py` and
then passed to `pipeline.run`. Everything in here is plain data.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


def _job_id_from(input_path: str) -> str:
    digest = hashlib.sha256(input_path.encode("utf-8")).hexdigest()[:10]
    return f"{int(time.time())}_{digest}"


@dataclass
class JobConfig:
    """Inputs, outputs, and tunables for one pipeline run."""

    audio_path: str
    output_dir: str
    asr_base_url: str
    asr_model: str

    job_id: str = ""
    work_dir: str = ""

    asr_api_key: str | None = None
    asr_provider_label: str = "external-openai-compatible"

    language: str | None = None
    prompt: str | None = None

    enable_vad: bool = True
    vad_backend: str = "auto"
    vad_pad_ms: int = 250
    vad_merge_gap_sec: float = 0.5
    vad_min_speech_sec: float = 0.3
    vad_min_speech_duration_ms: int = 1500
    vad_min_silence_duration_ms: int = 500

    chunk_mode: str = "fixed"
    chunk_threshold_sec: float = 600.0
    chunk_size_sec: float = 60.0
    chunk_overlap_sec: float = 2.0
    chunk_segment_threshold_sec: float = 120.0
    chunk_max_segment_sec: float = 180.0
    sample_rate: int = 16000
    enhance_audio: bool = False

    skip_diarization: bool = False
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_model_path: str | None = None
    use_exclusive_diarization: bool = True
    hf_token: str | None = None
    diarization_device: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    enable_repair: bool = False
    repair_base_url: str | None = None
    repair_model: str | None = None
    repair_api_key: str | None = None

    enable_alignment: bool = False

    resume: bool = False

    work_root: str = "data/work"

    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_id:
            self.job_id = _job_id_from(self.audio_path)
        if not self.work_dir:
            self.work_dir = str(Path(self.work_root) / self.job_id)

    @classmethod
    def from_env_and_args(cls, **kwargs) -> "JobConfig":
        """Convenience builder that fills secrets/defaults from environment variables."""

        kwargs.setdefault("asr_api_key", os.getenv("ASR_API_KEY"))
        kwargs.setdefault("repair_base_url", os.getenv("REPAIR_BASE_URL"))
        kwargs.setdefault("repair_model", os.getenv("REPAIR_MODEL"))
        kwargs.setdefault("repair_api_key", os.getenv("REPAIR_API_KEY"))
        kwargs.setdefault("hf_token", os.getenv("HF_TOKEN"))
        return cls(**kwargs)

    def artifact_path(self, name: str) -> Path:
        """Path under `work_dir` for a named intermediate artifact."""

        return Path(self.work_dir) / name
