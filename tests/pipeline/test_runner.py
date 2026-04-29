import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from recalld.config import Category, Config
from recalld.jobs import JobStage, JobStatus, create_job, load_job, save_job
from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.postprocess import PostProcessResult
from recalld.pipeline.runner import run_pipeline


@pytest.mark.asyncio
async def test_pipeline_waits_for_vault_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    scratch = tmp_path / job.id
    aligned_path = scratch / "aligned.json"
    aligned_path.write_text(json.dumps([
        LabelledTurn(speaker="You", start=0.0, end=1.0, text="Hello").__dict__,
    ]))
    job.aligned_path = str(aligned_path)
    job.current_stage = JobStage.postprocess
    save_job(job, scratch_root=tmp_path)

    cfg = Config(
        llm_model="test-model",
        categories=[Category(id="cat-1", name="Coaching", vault_path="Life/Sessions")],
    )
    result = PostProcessResult(
        summary="Summary",
        focus_points=["Follow up"],
        raw_response="",
        strategy="single",
        topic_count=1,
    )

    with patch("recalld.pipeline.runner.detect_context_length", AsyncMock(return_value=1000)), \
         patch("recalld.pipeline.runner.postprocess", AsyncMock(return_value=result)), \
         patch("recalld.pipeline.runner.VaultWriter") as MockWriter:
        await run_pipeline(job, scratch / "session.m4a", cfg)

    MockWriter.assert_not_called()
    reloaded = load_job(job.id, scratch_root=tmp_path)
    assert reloaded.current_stage == JobStage.vault
    assert reloaded.status == JobStatus.pending
    assert reloaded.stage_statuses["postprocess"] == "done"
    assert reloaded.stage_statuses["vault"] == "awaiting_confirmation"
    assert reloaded.postprocess_path is not None
