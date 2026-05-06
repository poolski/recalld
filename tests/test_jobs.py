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
    can_restart_from_stage,
    reset_job_for_rerun,
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


def test_note_target_mode_is_persisted(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)
    job.note_target_mode = "existing"
    save_job(job, scratch_root=tmp_path)

    loaded = load_job(job.id, scratch_root=tmp_path)

    assert loaded.note_target_mode == "existing"


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


def test_reset_job_for_rerun_from_failed_clears_current_stage_failure(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)
    job.current_stage = JobStage.diarise
    job.status = JobStatus.failed
    job.error = "boom"
    job.wav_path = str(tmp_path / job.id / "audio.wav")
    job.transcript_path = str(tmp_path / job.id / "transcript.json")
    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    job.stage_statuses["diarise"] = "failed"

    reset_job_for_rerun(job, from_start=False)

    assert job.current_stage == JobStage.diarise
    assert job.status == JobStatus.pending
    assert job.error is None
    assert job.stage_statuses["ingest"] == "done"
    assert job.stage_statuses["transcribe"] == "done"
    assert job.stage_statuses["diarise"] == "pending"
    assert job.stage_statuses["align"] == "pending"
    assert job.stage_statuses["postprocess"] == "pending"
    assert job.stage_statuses["vault"] == "pending"


def test_reset_job_for_rerun_from_start_clears_outputs_and_statuses(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)
    job.current_stage = JobStage.vault
    job.status = JobStatus.failed
    job.error = "boom"
    job.wav_path = "audio.wav"
    job.transcript_path = "transcript.json"
    job.diarisation_path = "diarisation.json"
    job.aligned_path = "aligned.json"
    job.postprocess_path = "postprocess.json"
    job.topic_count = 3
    job.chunk_strategy = "map_reduce"
    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    job.stage_statuses["diarise"] = "done"
    job.stage_statuses["align"] = "done"
    job.stage_statuses["postprocess"] = "done"
    job.stage_statuses["vault"] = "failed"

    reset_job_for_rerun(job, from_start=True)

    assert job.current_stage == JobStage.ingest
    assert job.status == JobStatus.pending
    assert job.error is None
    assert job.wav_path is None
    assert job.transcript_path is None
    assert job.diarisation_path is None
    assert job.aligned_path is None
    assert job.postprocess_path is None
    assert job.topic_count is None
    assert job.chunk_strategy is None
    assert all(status == "pending" for status in job.stage_statuses.values())


def test_reset_job_for_rerun_from_stage_clears_outputs_from_that_stage_onward(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)
    job.current_stage = JobStage.vault
    job.status = JobStatus.failed
    job.error = "boom"
    job.wav_path = "audio.wav"
    job.transcript_path = "transcript.json"
    job.diarisation_path = "diarisation.json"
    job.aligned_path = "aligned.json"
    job.postprocess_path = "postprocess.json"
    job.topic_count = 2
    job.chunk_strategy = "single"
    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    job.stage_statuses["diarise"] = "done"
    job.stage_statuses["align"] = "done"
    job.stage_statuses["postprocess"] = "done"
    job.stage_statuses["vault"] = "failed"

    reset_job_for_rerun(job, from_start=False, restart_stage=JobStage.diarise)

    assert job.current_stage == JobStage.diarise
    assert job.status == JobStatus.pending
    assert job.error is None
    assert job.wav_path == "audio.wav"
    assert job.transcript_path == "transcript.json"
    assert job.diarisation_path is None
    assert job.aligned_path is None
    assert job.postprocess_path is None
    assert job.topic_count is None
    assert job.chunk_strategy is None
    assert job.stage_statuses["ingest"] == "done"
    assert job.stage_statuses["transcribe"] == "done"
    assert job.stage_statuses["diarise"] == "pending"
    assert job.stage_statuses["align"] == "pending"
    assert job.stage_statuses["postprocess"] == "pending"
    assert job.stage_statuses["vault"] == "pending"


def test_reset_job_for_rerun_restart_from_transcribe_clears_transcript_cache(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)
    transcript_file = tmp_path / "transcript.json"
    transcript_file.write_text("[]")
    diarisation_file = tmp_path / "diarisation.json"
    diarisation_file.write_text("[]")
    aligned_file = tmp_path / "aligned.json"
    aligned_file.write_text("[]")
    postprocess_file = tmp_path / "postprocess.json"
    postprocess_file.write_text("{}")
    job.transcript_path = str(transcript_file)
    job.diarisation_path = str(diarisation_file)
    job.aligned_path = str(aligned_file)
    job.postprocess_path = str(postprocess_file)
    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    job.stage_statuses["diarise"] = "done"
    job.stage_statuses["align"] = "done"
    job.stage_statuses["postprocess"] = "done"
    job.stage_statuses["vault"] = "failed"

    reset_job_for_rerun(job, from_start=False, restart_stage=JobStage.transcribe)

    assert job.transcript_path is None
    assert job.diarisation_path is None
    assert job.aligned_path is None
    assert job.postprocess_path is None
    assert not transcript_file.exists()
    assert not diarisation_file.exists()
    assert not aligned_file.exists()
    assert not postprocess_file.exists()


def test_can_restart_from_stage_checks_prerequisites(tmp_path):
    job = create_job(category_id="adhd", original_filename="session.m4a", scratch_root=tmp_path)
    job.wav_path = "audio.wav"
    job.transcript_path = "transcript.json"

    assert can_restart_from_stage(job, JobStage.ingest) is True
    assert can_restart_from_stage(job, JobStage.transcribe) is True
    assert can_restart_from_stage(job, JobStage.diarise) is True
    assert can_restart_from_stage(job, JobStage.align) is False
