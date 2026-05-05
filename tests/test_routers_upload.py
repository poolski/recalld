from fastapi.testclient import TestClient

from recalld.app import create_app
from recalld.config import Category, Config, save_config
from recalld.jobs import JobStatus, create_job, list_jobs, save_job


def test_index_defaults_to_new_recording_tab(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.routers.upload.DEFAULT_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr("recalld.app.DEFAULT_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr("recalld.config.DEFAULT_CONFIG_PATH", tmp_path / "config.json")
    client = TestClient(create_app())

    resp = client.get("/")

    assert resp.status_code == 200
    assert "Conversation Processing" in resp.text
    assert "New Recording" in resp.text
    assert "Queue snapshot" in resp.text


def test_index_jobs_tab_renders_all_jobs_table(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.routers.upload.DEFAULT_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr("recalld.app.DEFAULT_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr("recalld.config.DEFAULT_CONFIG_PATH", tmp_path / "config.json")
    client = TestClient(create_app())

    running = create_job(category_id="test", original_filename="running.m4a", scratch_root=tmp_path)
    running.status = JobStatus.running
    save_job(running, scratch_root=tmp_path)

    complete = create_job(category_id="test", original_filename="done.m4a", scratch_root=tmp_path)
    complete.status = JobStatus.complete
    complete.current_stage = complete.current_stage.vault
    complete.stage_statuses["vault"] = "done"
    save_job(complete, scratch_root=tmp_path)

    resp = client.get("/?tab=jobs")

    assert resp.status_code == 200
    assert "All Jobs" in resp.text
    assert "running.m4a" in resp.text
    assert "done.m4a" in resp.text
    assert "Awaiting confirmation" in resp.text
    assert "Selected job" in resp.text


def test_upload_uses_user_provided_note_title(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.routers.upload.DEFAULT_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr("recalld.app.DEFAULT_SCRATCH_ROOT", tmp_path)
    config_path = tmp_path / "config.json"
    monkeypatch.setattr("recalld.config.DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr("recalld.routers.upload.run_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr("asyncio.create_task", lambda coro: None)

    cfg = Config(categories=[Category(id="cat-1", name="Coaching", vault_path="Notes/Sessions")])
    save_config(cfg, path=config_path)

    client = TestClient(create_app())
    resp = client.post(
        "/upload",
        data={"category_id": "cat-1", "note_title": "Project Starfish Meeting"},
        files={"file": ("session.m4a", b"audio", "audio/mp4")},
    )
    assert resp.status_code == 200
    assert "Project Starfish Meeting.md" in resp.text


def test_upload_sanitizes_user_provided_note_title(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.routers.upload.DEFAULT_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr("recalld.app.DEFAULT_SCRATCH_ROOT", tmp_path)
    config_path = tmp_path / "config.json"
    monkeypatch.setattr("recalld.config.DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr("recalld.routers.upload.run_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr("asyncio.create_task", lambda coro: None)

    cfg = Config(categories=[Category(id="cat-1", name="Coaching", vault_path="Notes/Sessions")])
    save_config(cfg, path=config_path)

    client = TestClient(create_app())
    resp = client.post(
        "/upload",
        data={"category_id": "cat-1", "note_title": "../Project / Starfish\\.."},
        files={"file": ("session.m4a", b"audio", "audio/mp4")},
    )

    assert resp.status_code == 200
    job = list_jobs(scratch_root=tmp_path)[0]
    assert job.filename is not None
    assert "/" not in job.filename
    assert "\\" not in job.filename
    assert ".." not in job.filename
    assert job.filename.endswith(".md")
