from __future__ import annotations

import pytest
from fastapi import FastAPI

from recalld.app import lifespan


@pytest.mark.asyncio
async def test_lifespan_shuts_down_tracing(monkeypatch, tmp_path):
    calls: list[str] = []

    monkeypatch.setattr("recalld.app.shutdown_tracing", lambda: calls.append("shutdown"))
    monkeypatch.setattr("recalld.config.DEFAULT_CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr("recalld.jobs.DEFAULT_SCRATCH_ROOT", tmp_path / "scratch")

    async with lifespan(FastAPI()):
        pass

    assert calls == ["shutdown"]
