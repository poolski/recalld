import pytest
from pathlib import Path
from fastapi.testclient import TestClient
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
    assert "Resume" in resp.text
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

    resp = client.post(f"/jobs/{job.id}/confirm-vault-write")

    assert resp.status_code == 200
    updated = load_job(job.id, scratch_root=scratch)
    assert updated.status == JobStatus.running
    assert updated.stage_statuses["vault"] == "pending"
    assert "coro" in scheduled


def test_job_state_returns_latest_stage_statuses(scratch, client):
    job = create_job(category_id="test", original_filename="audio.m4a", scratch_root=scratch)
    job.stage_statuses["ingest"] = "done"
    job.stage_statuses["transcribe"] = "done"
    save_job(job, scratch_root=scratch)

    resp = client.get(f"/jobs/{job.id}/state")

    assert resp.status_code == 200
    assert resp.json()["stage_statuses"]["ingest"] == "done"
    assert resp.json()["stage_statuses"]["transcribe"] == "done"
