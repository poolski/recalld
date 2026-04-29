from pathlib import Path
import json
from recalld.jobs import (
    Job,
    JobStage,
    JobStatus,
    create_job,
    delete_job,
    list_incomplete_jobs,
    load_job,
    save_job,
)


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


def test_save_job_creates_dir_if_missing(tmp_path):
    job = Job(category_id="x", original_filename="f.m4a")
    # Don't call create_job — call save_job directly on a new scratch root
    new_root = tmp_path / "newroot"
    save_job(job, scratch_root=new_root)
    assert (new_root / job.id / "job.json").exists()


def test_delete_job(tmp_path):
    job = create_job(category_id="test", original_filename="x.m4a", scratch_root=tmp_path)
    assert (tmp_path / job.id).exists()
    delete_job(job.id, scratch_root=tmp_path)
    assert not (tmp_path / job.id).exists()


def test_completed_stage_statuses_are_persisted(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)

    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    job.stage_statuses["diarise"] = "running"
    save_job(job, scratch_root=tmp_path)

    loaded = load_job(job.id, scratch_root=tmp_path)

    assert loaded.stage_statuses["ingest"] == "done"
    assert loaded.stage_statuses["transcribe"] == "done"
    assert loaded.stage_statuses["diarise"] == "running"


def test_load_job_infers_stage_statuses_for_legacy_jobs(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)
    job_file = tmp_path / job.id / "job.json"
    payload = json.loads(job_file.read_text())
    payload.pop("stage_statuses", None)
    payload["current_stage"] = "diarise"
    payload["status"] = "failed"
    job_file.write_text(json.dumps(payload))

    loaded = load_job(job.id, scratch_root=tmp_path)

    assert loaded.stage_statuses["ingest"] == "done"
    assert loaded.stage_statuses["transcribe"] == "done"
    assert loaded.stage_statuses["diarise"] == "failed"
