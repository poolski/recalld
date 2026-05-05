from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator


DEFAULT_SCRATCH_ROOT = Path.home() / ".local" / "share" / "recalld" / "jobs"


class JobStage(str, Enum):
    ingest = "ingest"
    transcribe = "transcribe"
    diarise = "diarise"
    align = "align"
    postprocess = "postprocess"
    vault = "vault"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    failed = "failed"
    complete = "complete"


STAGE_NAMES = tuple(stage.value for stage in JobStage)
STAGE_ORDER = {stage.value: index for index, stage in enumerate(JobStage)}


def default_stage_statuses() -> dict[str, str]:
    return {stage: "pending" for stage in STAGE_NAMES}


def _infer_stage_statuses(current_stage: JobStage, status: JobStatus) -> dict[str, str]:
    stage_statuses = default_stage_statuses()
    current_index = STAGE_NAMES.index(current_stage.value)

    for stage in STAGE_NAMES[:current_index]:
        stage_statuses[stage] = "done"

    if status == JobStatus.complete:
        for stage in STAGE_NAMES:
            stage_statuses[stage] = "done"
    elif status == JobStatus.failed:
        stage_statuses[current_stage.value] = "failed"
    elif status == JobStatus.running:
        stage_statuses[current_stage.value] = "running"

    return stage_statuses


def can_restart_from_stage(job: "Job", stage: JobStage) -> bool:
    if stage == JobStage.ingest:
        return True
    if stage == JobStage.transcribe:
        return job.wav_path is not None
    if stage == JobStage.diarise:
        return job.wav_path is not None
    if stage == JobStage.align:
        return job.transcript_path is not None and job.diarisation_path is not None
    if stage == JobStage.postprocess:
        return job.aligned_path is not None
    if stage == JobStage.vault:
        return job.aligned_path is not None and job.postprocess_path is not None
    return False


class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category_id: str
    original_filename: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_stage: JobStage = JobStage.ingest
    status: JobStatus = JobStatus.pending
    error: Optional[str] = None
    stage_statuses: dict[str, str] = Field(default_factory=default_stage_statuses)
    # Paths to stage outputs (set as each stage completes)
    wav_path: Optional[str] = None
    transcript_path: Optional[str] = None
    diarisation_path: Optional[str] = None
    aligned_path: Optional[str] = None
    postprocess_path: Optional[str] = None
    # Speaker assignment (filled after align stage)
    speaker_00: Optional[str] = None
    speaker_01: Optional[str] = None
    # LLM chunking info for UI display
    topic_count: Optional[int] = None
    chunk_strategy: Optional[str] = None
    filename: Optional[str] = None
    vault_write_mode: Optional[str] = None
    vault_conflict_path: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def ensure_stage_statuses(cls, data):
        if isinstance(data, dict) and "stage_statuses" not in data:
            current_stage = JobStage(data.get("current_stage", JobStage.ingest.value))
            status = JobStatus(data.get("status", JobStatus.pending.value))
            data["stage_statuses"] = _infer_stage_statuses(current_stage, status)
        return data


def _job_dir(job_id: str, scratch_root: Path) -> Path:
    return scratch_root / job_id


def _remove_artifact(path_value: Optional[str]) -> None:
    if not path_value:
        return
    try:
        Path(path_value).unlink(missing_ok=True)
    except OSError:
        pass


def _clear_outputs_from_stage(job: "Job", stage: JobStage) -> None:
    if stage == JobStage.transcribe:
        _remove_artifact(job.transcript_path)
        job.transcript_path = None
    if STAGE_ORDER[stage.value] <= STAGE_ORDER[JobStage.diarise.value]:
        _remove_artifact(job.diarisation_path)
        job.diarisation_path = None
    if STAGE_ORDER[stage.value] <= STAGE_ORDER[JobStage.align.value]:
        _remove_artifact(job.aligned_path)
        job.aligned_path = None
    if STAGE_ORDER[stage.value] <= STAGE_ORDER[JobStage.postprocess.value]:
        _remove_artifact(job.postprocess_path)
        job.postprocess_path = None
        job.topic_count = None
        job.chunk_strategy = None
    if STAGE_ORDER[stage.value] <= STAGE_ORDER[JobStage.vault.value]:
        job.vault_write_mode = None
        job.vault_conflict_path = None


def create_job(category_id: str, original_filename: str, scratch_root: Path = DEFAULT_SCRATCH_ROOT) -> Job:
    job = Job(category_id=category_id, original_filename=original_filename)
    _job_dir(job.id, scratch_root).mkdir(parents=True, exist_ok=True)
    save_job(job, scratch_root)
    return job


def save_job(job: Job, scratch_root: Path = DEFAULT_SCRATCH_ROOT) -> None:
    path = _job_dir(job.id, scratch_root) / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(job.model_dump_json(indent=2))


def load_job(job_id: str, scratch_root: Path = DEFAULT_SCRATCH_ROOT) -> Job:
    path = _job_dir(job_id, scratch_root) / "job.json"
    return Job.model_validate_json(path.read_text())


def reset_job_for_rerun(job: Job, from_start: bool, restart_stage: JobStage | None = None) -> None:
    job.status = JobStatus.pending
    job.error = None

    if from_start:
        job.current_stage = JobStage.ingest
        job.stage_statuses = default_stage_statuses()
        _remove_artifact(job.wav_path)
        job.wav_path = None
        _clear_outputs_from_stage(job, JobStage.transcribe)
        job.speaker_00 = None
        job.speaker_01 = None
        job.vault_write_mode = None
        job.vault_conflict_path = None
        return

    if restart_stage is not None:
        job.current_stage = restart_stage
        for stage in STAGE_NAMES:
            if STAGE_ORDER[stage] >= STAGE_ORDER[restart_stage.value]:
                job.stage_statuses[stage] = "pending"
        _clear_outputs_from_stage(job, restart_stage)
        if STAGE_ORDER[restart_stage.value] <= STAGE_ORDER[JobStage.align.value]:
            job.speaker_00 = None
            job.speaker_01 = None
        return

    restart_stage = job.current_stage
    for stage in STAGE_NAMES:
        if STAGE_ORDER[stage] >= STAGE_ORDER[restart_stage.value]:
            job.stage_statuses[stage] = "pending"
    _clear_outputs_from_stage(job, restart_stage)
    if STAGE_ORDER[restart_stage.value] <= STAGE_ORDER[JobStage.align.value]:
        job.speaker_00 = None
        job.speaker_01 = None


def delete_job(job_id: str, scratch_root: Path = DEFAULT_SCRATCH_ROOT) -> None:
    shutil.rmtree(_job_dir(job_id, scratch_root), ignore_errors=True)


def list_incomplete_jobs(scratch_root: Path = DEFAULT_SCRATCH_ROOT) -> list[Job]:
    return [
        job for job in list_jobs(scratch_root=scratch_root)
        if job.status not in (JobStatus.complete,)
    ]


def list_jobs(scratch_root: Path = DEFAULT_SCRATCH_ROOT) -> list[Job]:
    if not scratch_root.exists():
        return []
    jobs = []
    for job_dir in scratch_root.iterdir():
        job_file = job_dir / "job.json"
        if not job_file.exists():
            continue
        job = Job.model_validate_json(job_file.read_text())
        jobs.append(job)
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)
