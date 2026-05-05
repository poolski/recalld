import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from recalld.config import Category, Config
from recalld.app import create_app
from recalld.jobs import JobStage, JobStatus, create_job, load_job, save_job


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.routers.jobs.DEFAULT_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr("recalld.app.DEFAULT_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr("recalld.config.DEFAULT_CONFIG_PATH", tmp_path / "config.json")
    return tmp_path


@pytest.fixture
def client(scratch):
    return TestClient(create_app())


def test_delete_removes_directory(scratch, client):
    job = create_job(category_id="test", original_filename="x.m4a", scratch_root=scratch)
    assert (scratch / job.id).exists()
    resp = client.delete(f"/jobs/{job.id}")
    assert resp.status_code == 200
    assert resp.text == ""
    assert not (scratch / job.id).exists()


def test_confirm_delete_returns_html(scratch, client):
    job = create_job(category_id="test", original_filename="session.m4a", scratch_root=scratch)
    resp = client.get(f"/jobs/{job.id}/confirm-delete")
    assert resp.status_code == 200
    assert "session.m4a" in resp.text
    assert "Delete" in resp.text
    assert "Cancel" in resp.text


def test_job_row_returns_html(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    resp = client.get(f"/jobs/{job.id}/row")
    assert resp.status_code == 200
    assert "audio.m4a" in resp.text
    assert "View details" in resp.text
    assert "stage-restart-btn" in resp.text
    assert "Remove" in resp.text


def test_job_detail_renders_persisted_stage_statuses(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    job.current_stage = JobStage.diarise
    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    job.stage_statuses["diarise"] = "running"
    save_job(job, scratch_root=scratch)

    resp = client.get(f"/jobs/{job.id}")

    assert resp.status_code == 200
    assert '"ingest": "done"' in resp.text
    assert '"transcribe": "done"' in resp.text
    assert '"diarise": "running"' in resp.text


def test_job_detail_shows_vault_confirmation_button(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    job.current_stage = JobStage.vault
    job.status = JobStatus.pending
    job.stage_statuses["postprocess"] = "done"
    job.stage_statuses["vault"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    resp = client.get(f"/jobs/{job.id}")

    assert resp.status_code == 200
    assert "Confirm and write to vault" in resp.text


def test_job_detail_shows_speaker_confirmation_controls(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    job.current_stage = JobStage.align
    job.status = JobStatus.pending
    job.stage_statuses["align"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    resp = client.get(f"/jobs/{job.id}")

    assert resp.status_code == 200
    assert "Confirm speakers" in resp.text
    assert "Swap speakers" in resp.text


def test_job_detail_shows_rerun_buttons_for_failed_job(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    job.current_stage = JobStage.diarise
    job.status = JobStatus.failed
    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    job.stage_statuses["diarise"] = "failed"
    save_job(job, scratch_root=scratch)

    resp = client.get(f"/jobs/{job.id}")

    assert resp.status_code == 200
    assert "Restart" in resp.text
    assert "row-restart-btn" in resp.text


def test_job_detail_does_not_render_error_message_in_stage_header(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    job.current_stage = JobStage.align
    job.status = JobStatus.failed
    job.stage_statuses["align"] = "failed"
    job.error = "WordSegment.__init__() got an unexpected keyword argument 'segment'"
    save_job(job, scratch_root=scratch)

    resp = client.get(f"/jobs/{job.id}")

    assert resp.status_code == 200
    assert job.error not in resp.text


def test_confirm_vault_write_updates_job_and_resumes_pipeline(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    job.current_stage = JobStage.vault
    job.status = JobStatus.pending
    job.stage_statuses["vault"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    scheduled = {}

    async def fake_run_pipeline(job_arg, source_arg, cfg_arg):
        return None

    def fake_create_task(coro):
        scheduled["coro"] = coro
        coro.close()
        return None

    monkeypatch.setattr("recalld.routers.jobs.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("asyncio.create_task", fake_create_task)
    config = Config(categories=[Category(id="test", name="Coaching", vault_path="Life/Sessions")])
    monkeypatch.setattr("recalld.config.load_config", lambda path=None: config)
    async def fake_exists(self, vault_path):
        return False
    monkeypatch.setattr("recalld.pipeline.vault.VaultWriter.note_exists", fake_exists)

    resp = client.post(f"/jobs/{job.id}/confirm-vault-write")

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    assert updated.status == JobStatus.running
    assert updated.stage_statuses["vault"] == "pending"
    assert "coro" in scheduled


def test_confirm_vault_write_prompts_overwrite_or_append_when_note_exists(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    job.current_stage = JobStage.vault
    job.status = JobStatus.pending
    session_date = job.created_at.date()
    job.filename = f"{session_date.isoformat()} Coaching.md"
    job.stage_statuses["vault"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    config = Config(categories=[Category(id="test", name="Coaching", vault_path="Life/Sessions")])
    monkeypatch.setattr("recalld.config.load_config", lambda path=None: config)

    async def fake_exists(self, vault_path):
        return True

    monkeypatch.setattr("recalld.pipeline.vault.VaultWriter.note_exists", fake_exists)

    scheduled = {}

    def fake_create_task(coro):
        scheduled["coro"] = coro
        coro.close()
        return None

    monkeypatch.setattr("asyncio.create_task", fake_create_task)

    resp = client.post(
        f"/jobs/{job.id}/confirm-vault-write",
        data={"filename": f"{session_date.isoformat()} Coaching.md"},
    )

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    assert updated.status == JobStatus.pending
    assert updated.stage_statuses["vault"] == "awaiting_confirmation"
    assert updated.vault_conflict_path == f"Life/Sessions/{session_date.isoformat()} Coaching.md"
    assert "Overwrite existing note" in resp.text
    assert "Append to existing note" in resp.text
    assert "coro" not in scheduled


def test_confirm_vault_write_with_append_mode_resumes_pipeline(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    job.current_stage = JobStage.vault
    job.status = JobStatus.pending
    job.filename = "2025-04-29 Coaching.md"
    job.stage_statuses["vault"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    config = Config(categories=[Category(id="test", name="Coaching", vault_path="Life/Sessions")])
    monkeypatch.setattr("recalld.config.load_config", lambda path=None: config)

    async def fake_exists(self, vault_path):
        return True

    monkeypatch.setattr("recalld.pipeline.vault.VaultWriter.note_exists", fake_exists)

    scheduled = {}

    async def fake_run_pipeline(job_arg, source_arg, cfg_arg):
        return None

    def fake_create_task(coro):
        scheduled["coro"] = coro
        coro.close()
        return None

    monkeypatch.setattr("recalld.routers.jobs.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("asyncio.create_task", fake_create_task)

    resp = client.post(
        f"/jobs/{job.id}/confirm-vault-write",
        data={"filename": "2025-04-29 Coaching.md", "write_mode": "append"},
    )

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    assert updated.status == JobStatus.running
    assert updated.stage_statuses["vault"] == "pending"
    assert updated.vault_write_mode == "append"
    assert updated.vault_conflict_path is None
    assert "coro" in scheduled


def test_confirm_vault_write_with_append_mode_falls_back_when_note_missing(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    job.current_stage = JobStage.vault
    job.status = JobStatus.pending
    job.filename = "2025-04-29 Coaching.md"
    job.stage_statuses["vault"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    config = Config(categories=[Category(id="test", name="Coaching", vault_path="Life/Sessions")])
    monkeypatch.setattr("recalld.config.load_config", lambda path=None: config)

    async def fake_exists(self, vault_path):
        return False

    monkeypatch.setattr("recalld.pipeline.vault.VaultWriter.note_exists", fake_exists)

    scheduled = {}

    async def fake_run_pipeline(job_arg, source_arg, cfg_arg):
        return None

    def fake_create_task(coro):
        scheduled["coro"] = coro
        coro.close()
        return None

    monkeypatch.setattr("recalld.routers.jobs.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("asyncio.create_task", fake_create_task)

    resp = client.post(
        f"/jobs/{job.id}/confirm-vault-write",
        data={"filename": "2025-04-29 Coaching.md", "write_mode": "append"},
    )

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    assert updated.status == JobStatus.running
    assert updated.stage_statuses["vault"] == "pending"
    assert updated.vault_write_mode == "overwrite"
    assert updated.vault_conflict_path is None
    assert "coro" in scheduled


def test_confirm_vault_write_rejects_unknown_write_mode(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    job.current_stage = JobStage.vault
    job.status = JobStatus.pending
    job.filename = "2025-04-29 Coaching.md"
    job.stage_statuses["vault"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    config = Config(categories=[Category(id="test", name="Coaching", vault_path="Life/Sessions")])
    monkeypatch.setattr("recalld.config.load_config", lambda path=None: config)

    scheduled = {}

    def fake_create_task(coro):
        scheduled["coro"] = coro
        coro.close()
        return None

    monkeypatch.setattr("asyncio.create_task", fake_create_task)

    resp = client.post(
        f"/jobs/{job.id}/confirm-vault-write",
        data={"filename": "2025-04-29 Coaching.md", "write_mode": "delete"},
    )

    assert resp.status_code == 400
    assert "coro" not in scheduled


def test_confirm_vault_write_sanitizes_traversal_in_filename(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    job.current_stage = JobStage.vault
    job.status = JobStatus.pending
    job.stage_statuses["vault"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    config = Config(categories=[Category(id="test", name="Coaching", vault_path="Life/Sessions")])
    monkeypatch.setattr("recalld.config.load_config", lambda path=None: config)

    async def fake_exists(self, vault_path):
        return False

    monkeypatch.setattr("recalld.pipeline.vault.VaultWriter.note_exists", fake_exists)

    async def fake_run_pipeline(job_arg, source_arg, cfg_arg):
        return None

    def fake_create_task(coro):
        coro.close()
        return None

    monkeypatch.setattr("recalld.routers.jobs.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("asyncio.create_task", fake_create_task)

    resp = client.post(
        f"/jobs/{job.id}/confirm-vault-write",
        data={"filename": "../../etc/passwd"},
    )

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    # Path separators must be stripped; traversal sequences must not appear in the filename
    assert "/" not in (updated.filename or "")
    assert "\\" not in (updated.filename or "")
    assert ".." not in (updated.filename or "")


def test_open_in_obsidian_uses_full_vault_note_path(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    job.current_stage = JobStage.vault
    job.status = JobStatus.complete
    job.filename = "2025-04-29 Coaching.md"
    job.stage_statuses["vault"] = "done"
    save_job(job, scratch_root=scratch)
    config = Config(vault_name="Personal", categories=[Category(id="test", name="Coaching", vault_path="Life/Sessions")])
    monkeypatch.setattr("recalld.config.load_config", lambda path=None: config)

    resp = client.get(f"/jobs/{job.id}/open-in-obsidian", follow_redirects=False)

    assert resp.status_code == 302
    assert resp.headers["location"] == "obsidian://open?vault=Personal&file=Life/Sessions/2025-04-29%20Coaching.md"


def test_job_state_returns_latest_stage_statuses(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    save_job(job, scratch_root=scratch)

    resp = client.get(f"/jobs/{job.id}/state")

    assert resp.status_code == 200
    assert resp.json()["stage_statuses"]["ingest"] == "done"
    assert resp.json()["stage_statuses"]["transcribe"] == "done"


def test_job_state_returns_speaker_confirmation_flags(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    aligned_path = scratch / job.id / "aligned.json"
    aligned_path.write_text(json.dumps([
        {"speaker": "Speaker A", "start": 0.0, "end": 1.0, "text": "Hello"},
        {"speaker": "Speaker B", "start": 1.0, "end": 2.0, "text": "Hi"},
    ]))
    job.current_stage = JobStage.align
    job.aligned_path = str(aligned_path)
    job.stage_statuses["align"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    resp = client.get(f"/jobs/{job.id}/state")

    assert resp.status_code == 200
    assert resp.json()["can_confirm_speakers"] is True
    assert resp.json()["can_swap_speakers"] is True
    assert "Speaker A" in resp.json()["preview"]


def test_job_state_returns_vault_preview(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    aligned_path = scratch / job.id / "aligned.json"
    aligned_path.write_text(json.dumps([
        {"speaker": "You", "start": 0.0, "end": 1.0, "text": "Hello"},
        {"speaker": "Coach", "start": 1.0, "end": 2.0, "text": "Hi"},
    ]))
    postprocess_path = scratch / job.id / "postprocess.json"
    postprocess_path.write_text(json.dumps({
        "summary": "A productive session.",
        "focus_points": ["Follow up"],
        "strategy": "single",
        "topic_count": 1,
    }))
    job.current_stage = JobStage.vault
    job.aligned_path = str(aligned_path)
    job.postprocess_path = str(postprocess_path)
    job.stage_statuses["postprocess"] = "done"
    job.stage_statuses["vault"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)
    config = Config(categories=[Category(id="test", name="Coaching", vault_path="Life/Sessions")])
    monkeypatch.setattr("recalld.config.load_config", lambda path=None: config)

    resp = client.get(f"/jobs/{job.id}/state")

    assert resp.status_code == 200
    assert "A productive session." in resp.json()["vault_preview"]
    assert "date:" not in resp.json()["vault_preview"]


def test_confirm_speakers_resumes_pipeline(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    aligned_path = scratch / job.id / "aligned.json"
    aligned_path.write_text(json.dumps([
        {"speaker": "Speaker A", "start": 0.0, "end": 1.0, "text": "Hello"},
    ]))
    job.current_stage = JobStage.align
    job.status = JobStatus.pending
    job.aligned_path = str(aligned_path)
    job.stage_statuses["align"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    scheduled = {}

    async def fake_run_pipeline(job_arg, source_arg, cfg_arg):
        return None

    def fake_create_task(coro):
        scheduled["coro"] = coro
        coro.close()
        return None

    monkeypatch.setattr("recalld.routers.jobs.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("asyncio.create_task", fake_create_task)

    resp = client.post(f"/jobs/{job.id}/confirm-speakers")

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    assert updated.current_stage == JobStage.postprocess
    assert updated.status == JobStatus.running
    assert updated.stage_statuses["align"] == "done"
    assert "coro" in scheduled


def test_swap_speakers_updates_aligned_transcript(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    aligned_path = scratch / job.id / "aligned.json"
    aligned_path.write_text(json.dumps([
        {"speaker": "Speaker A", "start": 0.0, "end": 1.0, "text": "Hello"},
        {"speaker": "Speaker B", "start": 1.0, "end": 2.0, "text": "Hi"},
    ]))
    job.current_stage = JobStage.align
    job.aligned_path = str(aligned_path)
    job.stage_statuses["align"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    resp = client.post(f"/jobs/{job.id}/swap-speakers")

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    swapped = json.loads(Path(updated.aligned_path).read_text())
    assert swapped[0]["speaker"] == "Speaker B"
    assert swapped[1]["speaker"] == "Speaker A"
    assert updated.stage_statuses["align"] == "awaiting_confirmation"


def test_rerun_from_failed_updates_job_and_resumes_pipeline(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    job.current_stage = JobStage.diarise
    job.status = JobStatus.failed
    job.error = "boom"
    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    job.stage_statuses["diarise"] = "failed"
    save_job(job, scratch_root=scratch)

    scheduled = {}

    async def fake_run_pipeline(job_arg, source_arg, cfg_arg):
        return None

    def fake_create_task(coro):
        scheduled["coro"] = coro
        coro.close()
        return None

    monkeypatch.setattr("recalld.routers.jobs.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("asyncio.create_task", fake_create_task)

    resp = client.post(f"/jobs/{job.id}/rerun-from-failed")

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    assert updated.status == JobStatus.running
    assert updated.error is None
    assert updated.current_stage == JobStage.diarise
    assert updated.stage_statuses["diarise"] == "pending"
    assert "coro" in scheduled


def test_rerun_from_start_resets_job_and_resumes_pipeline(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
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
    save_job(job, scratch_root=scratch)

    scheduled = {}

    async def fake_run_pipeline(job_arg, source_arg, cfg_arg):
        return None

    def fake_create_task(coro):
        scheduled["coro"] = coro
        coro.close()
        return None

    monkeypatch.setattr("recalld.routers.jobs.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("asyncio.create_task", fake_create_task)

    resp = client.post(f"/jobs/{job.id}/rerun-from-start")

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    assert updated.status == JobStatus.running
    assert updated.current_stage == JobStage.ingest
    assert updated.wav_path is None
    assert updated.transcript_path is None
    assert updated.diarisation_path is None
    assert updated.aligned_path is None
    assert updated.postprocess_path is None
    assert all(status == "pending" for status in updated.stage_statuses.values())
    assert "coro" in scheduled


def test_restart_from_stage_updates_job_and_resumes_pipeline(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    job.current_stage = JobStage.vault
    job.status = JobStatus.failed
    job.error = "boom"
    job.wav_path = "audio.wav"
    job.transcript_path = "transcript.json"
    job.diarisation_path = "diarisation.json"
    job.aligned_path = "aligned.json"
    job.postprocess_path = "postprocess.json"
    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    job.stage_statuses["diarise"] = "done"
    job.stage_statuses["align"] = "done"
    job.stage_statuses["postprocess"] = "done"
    job.stage_statuses["vault"] = "failed"
    save_job(job, scratch_root=scratch)

    scheduled = {}

    async def fake_run_pipeline(job_arg, source_arg, cfg_arg):
        return None

    def fake_create_task(coro):
        scheduled["coro"] = coro
        coro.close()
        return None

    monkeypatch.setattr("recalld.routers.jobs.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("asyncio.create_task", fake_create_task)

    resp = client.post(f"/jobs/{job.id}/restart-from/diarise")

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    assert updated.current_stage == JobStage.diarise
    assert updated.status == JobStatus.running
    assert updated.diarisation_path is None
    assert updated.aligned_path is None
    assert updated.postprocess_path is None
    assert updated.stage_statuses["diarise"] == "pending"
    assert "coro" in scheduled


def test_confirm_speakers_emits_done_event(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    aligned_path = scratch / job.id / "aligned.json"
    aligned_path.write_text(json.dumps([{"speaker": "A", "text": "hi"}]))
    job.aligned_path = str(aligned_path)
    job.stage_statuses["align"] = "awaiting_confirmation"
    save_job(job, scratch_root=scratch)

    published = []

    def fake_publish(job_id, data):
        published.append((job_id, data))

    monkeypatch.setattr("recalld.routers.jobs.bus.publish", fake_publish)
    monkeypatch.setattr("recalld.routers.jobs.run_pipeline", AsyncMock())

    client.post(f"/jobs/{job.id}/confirm-speakers")

    assert (job.id, {"stage": "align", "status": "done"}) in published


def test_skip_diarise_emits_done_event(scratch, client, monkeypatch):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    (scratch / job.id / job.original_filename).write_bytes(b"audio")
    transcript_path = scratch / job.id / "transcript.json"
    transcript_path.write_text(json.dumps([{"start": 0, "end": 1, "word": "hi"}]))
    job.transcript_path = str(transcript_path)
    job.stage_statuses["diarise"] = "failed"
    save_job(job, scratch_root=scratch)

    published = []

    def fake_publish(job_id, data):
        published.append((job_id, data))

    monkeypatch.setattr("recalld.routers.jobs.bus.publish", fake_publish)
    monkeypatch.setattr("recalld.routers.jobs.run_pipeline", AsyncMock())

    client.post(f"/jobs/{job.id}/skip-diarise")

    assert (job.id, {"stage": "align", "status": "done"}) in published
