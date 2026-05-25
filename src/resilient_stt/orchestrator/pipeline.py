"""End-to-end pipeline wiring: normalize -> VAD -> chunk -> ASR -> stitch -> diarize -> assign -> align -> repair -> export."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from resilient_stt.alignment.base import AlignmentProvider, NoOpAligner
from resilient_stt.asr.endpoint_client import ASRProvider, OpenAICompatibleASRProvider
from resilient_stt.asr.router import ASRRouter
from resilient_stt.core.audio import audio_duration, normalize_audio
from resilient_stt.core.chunking import materialize_chunks, plan_chunks, plan_chunks_pause_aligned
from resilient_stt.core.exporters import export_all, write_json
from resilient_stt.core.schema import (
    ASRResult,
    ASRSegment,
    ASRWord,
    ChunkMeta,
    DiarizationTurn,
    SpeechRegion,
    TranscriptDocument,
    TranscriptSegment,
)
from resilient_stt.core.stitching import stitch_results
from resilient_stt.core.vad import analyze_vad, whole_file_region
from resilient_stt.diarization.speaker_assignment import assign_speakers
from resilient_stt.repair.llm_repair import OpenAICompatibleRepairClient, repair_transcript

from .config import JobConfig

logger = logging.getLogger("resilient_stt.pipeline")


def _ensure_work_dir(job: JobConfig) -> Path:
    root = Path(job.work_dir)
    (root / "chunks").mkdir(parents=True, exist_ok=True)
    (root / "asr_raw").mkdir(parents=True, exist_ok=True)
    return root


def _normalize_stage(job: JobConfig) -> Path:
    out = Path(job.work_dir) / "normalized.wav"
    if job.resume and out.exists():
        logger.info("[normalize] reuse %s", out)
        return out
    if job.enhance_audio:
        logger.info("[normalize] %s -> %s (enhance: highpass + afftdn + loudnorm)", job.audio_path, out)
    else:
        logger.info("[normalize] %s -> %s", job.audio_path, out)
    return normalize_audio(
        job.audio_path,
        out,
        sample_rate=job.sample_rate,
        enhance_audio=job.enhance_audio,
    )


def _load_vad_artifact(path: Path) -> tuple[list[SpeechRegion], list[int]]:
    """Load VAD artifact (supports legacy list-only format)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [SpeechRegion(**item) for item in data], []
    regions = [SpeechRegion(**item) for item in data.get("regions", [])]
    onsets = [int(x) for x in data.get("speech_onsets_samples", [])]
    return regions, onsets


def _vad_stage(job: JobConfig, normalized: Path) -> tuple[list[SpeechRegion], list[int]]:
    artifact = Path(job.work_dir) / "speech_regions.json"
    if job.resume and artifact.exists():
        logger.info("[vad] reuse %s", artifact)
        return _load_vad_artifact(artifact)

    speech_onsets: list[int] = []
    if not job.enable_vad:
        regions = [whole_file_region(normalized)]
    else:
        try:
            vad = analyze_vad(
                normalized,
                pad_ms=job.vad_pad_ms,
                merge_gap_sec=job.vad_merge_gap_sec,
                min_speech_sec=job.vad_min_speech_sec,
                min_speech_duration_ms=job.vad_min_speech_duration_ms,
                min_silence_duration_ms=job.vad_min_silence_duration_ms,
                backend=job.vad_backend,
            )
        except ImportError as exc:
            raise RuntimeError(str(exc)) from exc
        regions = vad.regions
        speech_onsets = vad.speech_onsets_samples
        if not regions:
            logger.warning("[vad] no speech detected; pipeline will export an empty transcript")

    write_json(
        {"regions": [r.model_dump() for r in regions], "speech_onsets_samples": speech_onsets},
        artifact,
    )
    logger.info("[vad] %d speech region(s), %d onset(s)", len(regions), len(speech_onsets))
    return regions, speech_onsets


def _chunk_stage(
    job: JobConfig,
    normalized: Path,
    regions: list[SpeechRegion],
    speech_onsets: list[int],
) -> list[ChunkMeta]:
    artifact = Path(job.work_dir) / "chunks.json"
    chunks_dir = Path(job.work_dir) / "chunks"

    if job.resume and artifact.exists():
        logger.info("[chunk] reuse %s", artifact)
        data = json.loads(artifact.read_text(encoding="utf-8"))
        return [ChunkMeta(**item) for item in data]

    if job.chunk_mode == "pause-aligned":
        planned = plan_chunks_pause_aligned(
            regions,
            speech_onsets,
            job.sample_rate,
            segment_threshold_sec=job.chunk_segment_threshold_sec,
            max_segment_sec=job.chunk_max_segment_sec,
        )
        logger.info(
            "[chunk] pause-aligned mode (target=%.0fs, max=%.0fs)",
            job.chunk_segment_threshold_sec,
            job.chunk_max_segment_sec,
        )
    else:
        planned = plan_chunks(
            regions,
            chunk_threshold_sec=job.chunk_threshold_sec,
            chunk_size_sec=job.chunk_size_sec,
            chunk_overlap_sec=job.chunk_overlap_sec,
        )
    materialized = materialize_chunks(normalized, planned, chunks_dir)
    write_json([c.model_dump() for c in materialized], artifact)
    logger.info("[chunk] %d ASR chunk(s)", len(materialized))
    if (
        job.asr_provider_label == "qwen-transformers-local"
        and len(materialized) == 1
        and materialized[0].end_offset - materialized[0].start_offset > 120.0
    ):
        logger.warning(
            "[chunk] Single ASR chunk is %.0fs; local qwen-asr is much faster with "
            "--chunk-threshold-sec 120 (multiple ~60s windows).",
            materialized[0].end_offset - materialized[0].start_offset,
        )
    return materialized


def _asr_stage(
    job: JobConfig,
    chunks: list[ChunkMeta],
    provider: ASRProvider,
) -> list[ASRResult]:
    router = ASRRouter(provider)
    raw_dir = Path(job.work_dir) / "asr_raw"
    results: list[ASRResult] = []
    for chunk in chunks:
        raw_path = raw_dir / f"{chunk.chunk_id}.json"
        if job.resume and raw_path.exists():
            data = json.loads(raw_path.read_text(encoding="utf-8"))
            results.append(ASRResult(**data))
            continue
        logger.info("[asr] %s -> %s", chunk.audio_path, chunk.chunk_id)
        result = router.for_chunk(chunk.chunk_id).transcribe(
            audio_path=chunk.audio_path,
            model=job.asr_model,
            language=job.language,
            prompt=job.prompt,
            chunk_id=chunk.chunk_id,
            start_offset=chunk.start_offset,
        )
        raw_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        results.append(result)
    return results


def _diarize_stage(job: JobConfig, normalized: Path) -> list[DiarizationTurn]:
    artifact = Path(job.work_dir) / "diarization.json"
    if job.resume and artifact.exists():
        data = json.loads(artifact.read_text(encoding="utf-8"))
        return [DiarizationTurn(**item) for item in data]

    if job.skip_diarization:
        logger.info("[diarize] skipped via --skip-diarization")
        write_json([], artifact)
        return []

    try:
        from resilient_stt.diarization.pyannote_provider import PyannoteDiarizationProvider
    except ImportError as exc:
        raise RuntimeError(
            "Diarization requires pyannote: uv sync --extra diarization "
            "(or uv sync --extra full). To skip speaker labels, pass --skip-diarization."
        ) from exc

    provider = PyannoteDiarizationProvider(
        model=job.diarization_model,
        hf_token=job.hf_token,
        local_path=job.diarization_model_path,
        use_exclusive=job.use_exclusive_diarization,
        device=job.diarization_device,
    )
    logger.info("[diarize] running pyannote on %s", normalized)
    turns = provider.diarize(
        normalized,
        num_speakers=job.num_speakers,
        min_speakers=job.min_speakers,
        max_speakers=job.max_speakers,
    )
    write_json([t.model_dump() for t in turns], artifact)
    logger.info("[diarize] %d turns", len(turns))
    return turns


def _assign_stage(
    job: JobConfig,
    segments: list[ASRSegment],
    words: list[ASRWord],
    turns: list[DiarizationTurn],
) -> list[TranscriptSegment]:
    assigned = assign_speakers(
        segments,
        words,
        turns,
        asr_provider=job.asr_provider_label,
        asr_model=job.asr_model,
    )
    write_json(
        [s.model_dump() for s in assigned],
        Path(job.work_dir) / "speaker_segments_raw.json",
    )
    return assigned


def _align_stage(
    job: JobConfig,
    normalized: Path,
    segments: list[TranscriptSegment],
    weak: bool,
    aligner: AlignmentProvider | None,
) -> list[TranscriptSegment]:
    if not segments:
        return segments
    if not (job.enable_alignment or weak):
        return segments
    aligner = aligner or NoOpAligner()
    logger.info("[align] running %s aligner (weak_timestamps=%s)", aligner.name, weak)
    try:
        return aligner.align(normalized, segments)
    except NotImplementedError:
        logger.warning("[align] aligner %s not implemented; skipping", aligner.name)
        return segments


def _repair_stage(job: JobConfig, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    if not job.enable_repair or not segments:
        return segments
    if not (job.repair_base_url and job.repair_model):
        logger.warning("[repair] --repair requested but REPAIR_BASE_URL/REPAIR_MODEL missing; skipping")
        return segments

    client = OpenAICompatibleRepairClient(
        base_url=job.repair_base_url,
        model=job.repair_model,
        api_key=job.repair_api_key,
    )
    logger.info("[repair] repairing %d segments via %s", len(segments), job.repair_model)
    repaired = repair_transcript(client, segments)
    write_json(
        [s.model_dump() for s in repaired],
        Path(job.work_dir) / "speaker_segments_repaired.json",
    )
    return repaired


def run(
    job: JobConfig,
    asr_provider: ASRProvider | None = None,
    aligner: AlignmentProvider | None = None,
) -> TranscriptDocument:
    """Execute the full pipeline and return the final transcript document."""

    _ensure_work_dir(job)
    normalized = _normalize_stage(job)
    duration = audio_duration(normalized)

    regions, speech_onsets = _vad_stage(job, normalized)
    chunks = _chunk_stage(job, normalized, regions, speech_onsets)

    provider = asr_provider or OpenAICompatibleASRProvider(
        base_url=job.asr_base_url,
        api_key=job.asr_api_key,
        provider_label=job.asr_provider_label,
    )

    asr_results = _asr_stage(job, chunks, provider)
    write_json(
        {"results": [r.model_dump() for r in asr_results]},
        Path(job.work_dir) / "asr_normalized.json",
    )

    segments, words = stitch_results(asr_results)
    weak = any(r.weak_timestamps for r in asr_results)
    turns = _diarize_stage(job, normalized)
    transcript_segments = _assign_stage(job, segments, words, turns)
    transcript_segments = _align_stage(job, normalized, transcript_segments, weak, aligner)
    transcript_segments = _repair_stage(job, transcript_segments)

    detected_language = job.language
    for r in asr_results:
        if r.language:
            detected_language = r.language
            break

    document = TranscriptDocument(
        audio_file=str(job.audio_path),
        duration=duration,
        language=detected_language,
        asr_provider=job.asr_provider_label,
        asr_model=job.asr_model,
        segments=transcript_segments,
    )
    export_all(document, job.output_dir)
    logger.info("[export] wrote transcript JSON/SRT/VTT under %s", job.output_dir)
    return document
