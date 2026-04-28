from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


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


class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category_id: str
    original_filename: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_stage: JobStage = JobStage.ingest
    status: JobStatus = JobStatus.pending
    error: Optional[str] = None
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


def _job_dir(job_id: str, scratch_root: Path) -> Path:
    return scratch_root / job_id


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


def list_incomplete_jobs(scratch_root: Path = DEFAULT_SCRATCH_ROOT) -> list[Job]:
    if not scratch_root.exists():
        return []
    jobs = []
    for job_dir in scratch_root.iterdir():
        job_file = job_dir / "job.json"
        if not job_file.exists():
            continue
        job = Job.model_validate_json(job_file.read_text())
        if job.status not in (JobStatus.complete,):
            jobs.append(job)
    return sorted(jobs, key=lambda j: j.created_at, reverse=True)
