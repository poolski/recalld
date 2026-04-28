import pytest
from pathlib import Path
from recalld.jobs import Job, JobStage, JobStatus, create_job, load_job, save_job, list_incomplete_jobs


def test_create_job(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)
    assert job.status == JobStatus.pending
    assert job.current_stage == JobStage.ingest
    assert (tmp_path / job.id).is_dir()


def test_save_and_load(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)
    job.status = JobStatus.running
    save_job(job, scratch_root=tmp_path)
    loaded = load_job(job.id, scratch_root=tmp_path)
    assert loaded.status == JobStatus.running


def test_advance_stage(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)
    job.current_stage = JobStage.transcribe
    save_job(job, scratch_root=tmp_path)
    loaded = load_job(job.id, scratch_root=tmp_path)
    assert loaded.current_stage == JobStage.transcribe


def test_list_incomplete_jobs(tmp_path):
    j1 = create_job(category_id="adhd", original_filename="a.m4a", scratch_root=tmp_path)
    j2 = create_job(category_id="adhd", original_filename="b.m4a", scratch_root=tmp_path)
    j1.status = JobStatus.running
    save_job(j1, scratch_root=tmp_path)
    j2.status = JobStatus.complete
    save_job(j2, scratch_root=tmp_path)
    incomplete = list_incomplete_jobs(scratch_root=tmp_path)
    assert len(incomplete) == 1
    assert incomplete[0].id == j1.id
