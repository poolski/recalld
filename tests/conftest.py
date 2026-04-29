from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def disposable_scratch_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    scratch_root = tmp_path / "scratch"
    config_path = tmp_path / "config.json"

    monkeypatch.setattr("recalld.jobs.DEFAULT_SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr("recalld.app.DEFAULT_SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr("recalld.routers.jobs.DEFAULT_SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr("recalld.routers.upload.DEFAULT_SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", scratch_root)
    monkeypatch.setattr("recalld.config.DEFAULT_CONFIG_PATH", config_path)

    return scratch_root
