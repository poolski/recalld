import pytest
from fastapi.testclient import TestClient
from recalld.app import create_app
from recalld.jobs import create_job


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
