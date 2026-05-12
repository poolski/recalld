from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from recalld.config import Config
from recalld.jobs import Job, save_job
from recalld.experiments.langfuse_style import (
    load_style_experiment_context,
    run_style_prompt_experiment,
)


class _FakeRunResult:
    def __init__(self, calls, item_results):
        self.calls = calls
        self.item_results = item_results

    def format(self):
        return "formatted"


class _FakeDataset:
    def __init__(self, input_data, dataset_id="dataset-1"):
        self.input_data = input_data
        self.id = dataset_id
        self.calls = []
        self.trace_score_batches = [
            [
                SimpleNamespace(
                    id="trace-score-1",
                    name="reference_alignment",
                    value=0.32,
                    comment="grounding",
                    metadata={},
                    data_type="NUMERIC",
                    config_id=None,
                )
            ],
            [
                SimpleNamespace(
                    id="trace-score-1",
                    name="reference_alignment",
                    value=0.32,
                    comment="grounding",
                    metadata={},
                    data_type="NUMERIC",
                    config_id=None,
                ),
                SimpleNamespace(
                    id="trace-score-2",
                    name="style-evaluator",
                    value=0.78,
                    comment="grounding",
                    metadata={},
                    data_type="NUMERIC",
                    config_id=None,
                ),
            ],
        ]
        self.trace_scores = self.trace_score_batches[-1]
        self.trace_calls = 0

    def run_experiment(self, **kwargs):
        self.calls.append(kwargs)
        task = kwargs["task"]
        item = SimpleNamespace(input=self.input_data)
        output = task(item=item)
        if asyncio.iscoroutine(output):
            output = asyncio.run(output)
        item_results = [
            SimpleNamespace(
                item=item,
                output=output,
                evaluations=[],
                trace_id=f"trace-{len(self.calls)}",
                dataset_run_id=kwargs.get("name"),
            )
        ]
        return _FakeRunResult(self.calls, item_results)


class _FakeClient:
    def __init__(self, dataset):
        self.dataset = dataset
        self.created_dataset = None
        self.created_item = None
        self.dataset_created = False
        self.created_scores = []
        self.api = SimpleNamespace(trace=SimpleNamespace(get=self.get_trace))

    def get_dataset(self, name):
        if not self.dataset_created:
            raise RuntimeError("dataset missing")
        return self.dataset

    def create_dataset(self, **kwargs):
        self.created_dataset = kwargs
        self.dataset_created = True
        return self.dataset

    def create_dataset_item(self, **kwargs):
        self.created_item = kwargs
        return kwargs

    def get_prompt(self, name, **kwargs):
        return SimpleNamespace(version=1)

    def get_trace(self, trace_id, fields=None):
        batch = self.dataset.trace_score_batches[min(self.dataset.trace_calls, len(self.dataset.trace_score_batches) - 1)]
        self.dataset.trace_calls += 1
        return SimpleNamespace(scores=batch)

    def create_score(self, **kwargs):
        self.created_scores.append(kwargs)
        return kwargs


class _FakeSessionContext:
    def __init__(self, calls, session_id):
        self.calls = calls
        self.session_id = session_id

    def __enter__(self):
        self.calls.append(("session_enter", self.session_id))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append(("session_exit", self.session_id))
        return False


def _write_job_artifacts(tmp_path: Path) -> Job:
    job = Job(category_id="cat-1", original_filename="session.m4a")
    job.id = "job-123"
    job.speaker_00 = "You"
    job.speaker_01 = "Lizzie"
    job.aligned_path = str(tmp_path / "aligned.json")
    job.postprocess_path = str(tmp_path / "postprocess.json")
    aligned = [
        {"speaker": "You", "text": "Hello", "start": 0.0, "end": 1.0},
        {"speaker": "Lizzie", "text": "Hi", "start": 1.0, "end": 2.0},
    ]
    (tmp_path / "aligned.json").write_text(json.dumps(aligned))
    (tmp_path / "postprocess.json").write_text(
        json.dumps(
            {
                "summary": "A short summary",
                "focus_points": ["Keep working"],
                "strategy": "single",
                "topic_count": 1,
            }
        )
    )
    save_job(job, scratch_root=tmp_path)
    return job


def test_run_style_prompt_experiment_registers_dataset_filter(tmp_path, monkeypatch):
    job = _write_job_artifacts(tmp_path)
    context = load_style_experiment_context(job.id, scratch_root=tmp_path)
    dataset = _FakeDataset(context.input_data, dataset_id="dataset-style")
    client = _FakeClient(dataset)
    captured: list[tuple[str, object]] = []

    captured_envs: list[str | None] = []

    def fake_get_client():
        captured_envs.append(os.getenv("LANGFUSE_TRACING_ENVIRONMENT"))
        return client

    monkeypatch.setattr("recalld.experiments.langfuse_style.get_langfuse_client", fake_get_client)
    monkeypatch.setattr("recalld.experiments.langfuse_style.shutdown_tracing", lambda: None)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_style.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="fallback-model"),
    )

    async def fake_context_length(*args, **kwargs):
        return 1000

    monkeypatch.setattr(
        "recalld.experiments.langfuse_style.ensure_loaded_context_length",
        fake_context_length,
    )
    monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)

    def fake_register(dataset_id: str, *, client=None, **kwargs):
        captured.append((dataset_id, client))
        return ["rule-1"]

    monkeypatch.setattr(
        "recalld.experiments.langfuse_style.ensure_evaluator_rules_include_dataset",
        fake_register,
    )

    async def fake_build_style_profile(*args, **kwargs):
        return "- direct\n- concise"

    monkeypatch.setattr("recalld.experiments.langfuse_style.build_style_profile", fake_build_style_profile)

    run_style_prompt_experiment(
        job_id=job.id,
        prompt_labels=["production"],
        scratch_root=tmp_path,
    )

    assert captured == [("dataset-style", client)]
    assert captured_envs == ["experiments"]
    assert "LANGFUSE_TRACING_ENVIRONMENT" not in os.environ


def test_run_style_prompt_experiment_wraps_session_context(tmp_path, monkeypatch):
    job = _write_job_artifacts(tmp_path)
    context = load_style_experiment_context(job.id, scratch_root=tmp_path)
    dataset = _FakeDataset(context.input_data, dataset_id="dataset-style")
    client = _FakeClient(dataset)
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr("recalld.experiments.langfuse_style.get_langfuse_client", lambda: client)
    monkeypatch.setattr("recalld.experiments.langfuse_style.shutdown_tracing", lambda: None)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_style.session_context",
        lambda session_id: _FakeSessionContext(calls, session_id),
    )
    monkeypatch.setattr(
        "recalld.experiments.langfuse_style.make_session_id",
        lambda *args, **kwargs: f"run-session-{args[1]}",
    )
    monkeypatch.setattr(
        "recalld.experiments.langfuse_style.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="fallback-model"),
    )

    async def fake_context_length(*args, **kwargs):
        return 1000

    monkeypatch.setattr(
        "recalld.experiments.langfuse_style.ensure_loaded_context_length",
        fake_context_length,
    )

    async def fake_build_style_profile(*args, **kwargs):
        return "- direct\n- concise"

    monkeypatch.setattr("recalld.experiments.langfuse_style.build_style_profile", fake_build_style_profile)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_style.ensure_evaluator_rules_include_dataset",
        lambda *args, **kwargs: [],
    )

    run_style_prompt_experiment(
        job_id=job.id,
        prompt_labels=["production"],
        scratch_root=tmp_path,
    )

    assert dataset.calls[0]["run_name"].startswith("style · session · production · fallback-model · ctx1000")
    assert len(client.created_scores) == 2
    assert [score["name"] for score in client.created_scores] == ["reference_alignment", "style-evaluator"]
    expected_session = f"run-session-{job.id[-6:]}"
    assert all(score["session_id"] == expected_session for score in client.created_scores)
    assert calls == [("session_enter", expected_session), ("session_exit", expected_session)]
