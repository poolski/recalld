from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from recalld.config import Config
from recalld.jobs import Job, save_job
from recalld.llm.context import ProviderModel
from recalld.pipeline.themes import ThemeProposalResult, ThemeSuggestion
from recalld.experiments.langfuse_themes import (
    load_themes_experiment_context,
    run_themes_prompt_experiment,
)


class _FakeRunResult:
    def __init__(self, calls, item_results):
        self.calls = calls
        self.item_results = item_results

    def format(self):
        return "formatted"


class _FakeDataset:
    def __init__(self, input_data, expected_output):
        self.input_data = input_data
        self.expected_output = expected_output
        self.calls = []
        self.evaluations = []
        self.trace_scores = [
            SimpleNamespace(
                id="trace-score-1",
                name="reference_alignment",
                value=0.66,
                comment="grounding",
                metadata={},
                data_type="NUMERIC",
                config_id=None,
            )
        ]

    def run_experiment(self, **kwargs):
        self.calls.append(kwargs)
        task = kwargs["task"]
        item = SimpleNamespace(input=self.input_data, expected_output=self.expected_output)
        output = task(item=item)
        if asyncio.iscoroutine(output):
            output = asyncio.run(output)
        run_evaluations = []
        for evaluator in kwargs["evaluators"]:
            evaluation = evaluator(
                input=item.input,
                output=output,
                expected_output=item.expected_output,
                metadata=kwargs["metadata"],
            )
            if asyncio.iscoroutine(evaluation):
                evaluation = asyncio.run(evaluation)
            self.evaluations.append(evaluation)
            run_evaluations.append(evaluation)
        item_results = [
            SimpleNamespace(
                item=item,
                output=output,
                evaluations=run_evaluations,
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
        self.updated_prompts = []
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

    def update_prompt(self, **kwargs):
        self.updated_prompts.append(kwargs)
        return kwargs

    def get_trace(self, trace_id, fields=None):
        return SimpleNamespace(scores=self.dataset.trace_scores)

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
    job.themes_path = str(tmp_path / "themes.json")
    aligned = [
        {"speaker": "You", "text": "Hello", "start": 0.0, "end": 1.0},
        {"speaker": "Lizzie", "text": "Hi", "start": 1.0, "end": 2.0},
    ]
    (tmp_path / "aligned.json").write_text(json.dumps(aligned))
    (tmp_path / "themes.json").write_text(
        json.dumps(
            [
                {
                    "id": "theme-1",
                    "title": "Active listening",
                    "notes": "Remember small details",
                    "enabled": True,
                    "order": 1,
                    "source": "manual",
                }
            ]
        )
    )
    save_job(job, scratch_root=tmp_path)
    return job


def test_load_themes_experiment_context_reads_artifacts(tmp_path):
    job = _write_job_artifacts(tmp_path)

    context = load_themes_experiment_context(job.id, scratch_root=tmp_path)

    assert context.job_id == job.id
    assert context.dataset_name == f"recalld/jobs/{job.id}/themes"
    assert context.dataset_item_id == f"job-{job.id}-themes"
    assert context.input_data["speaker_b_name"] == "Lizzie"
    assert context.expected_output["themes"][0]["title"] == "Active listening"


def test_run_themes_prompt_experiment_uses_each_requested_label(tmp_path, monkeypatch):
    job = _write_job_artifacts(tmp_path)
    context = load_themes_experiment_context(job.id, scratch_root=tmp_path)
    dataset = _FakeDataset(context.input_data, context.expected_output)
    client = _FakeClient(dataset)
    captured_calls: list[dict] = []

    captured_envs: list[str | None] = []

    def fake_get_client():
        captured_envs.append(os.getenv("LANGFUSE_TRACING_ENVIRONMENT"))
        return client

    monkeypatch.setattr("recalld.experiments.langfuse_themes.get_langfuse_client", fake_get_client)
    monkeypatch.setattr("recalld.experiments.langfuse_themes.shutdown_tracing", lambda: None)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="test-model"),
    )

    async def fake_context_length(*args, **kwargs):
        return 1000

    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.ensure_loaded_context_length",
        fake_context_length,
    )
    monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)

    async def fake_propose_themes(**kwargs):
        captured_calls.append(kwargs)
        return ThemeProposalResult(
            themes=[
                ThemeSuggestion(
                    id="theme-1",
                    title="Active listening",
                    notes="Remember small details",
                    enabled=True,
                    order=1,
                    source="transcript",
                )
            ],
            raw_response="{}",
            strategy="single",
            topic_count=1,
        )

    monkeypatch.setattr("recalld.experiments.langfuse_themes.propose_themes", fake_propose_themes)

    results = run_themes_prompt_experiment(
        job_id=job.id,
        prompt_labels=["production", "candidate"],
        scratch_root=tmp_path,
    )

    assert [call["metadata"]["prompt_label"] for call in dataset.calls] == ["production", "candidate"]
    assert dataset.calls[0]["run_name"].startswith("themes · session · production · test-model · ctx1000")
    assert [call["prompt_label"] for call in captured_calls] == ["production", "candidate"]
    assert [result["prompt_label"] for result in results] == ["production", "candidate"]
    assert client.created_dataset["name"] == context.dataset_name
    assert client.created_item["id"] == context.dataset_item_id
    assert captured_envs == ["experiments"]
    assert "LANGFUSE_TRACING_ENVIRONMENT" not in os.environ
    assert len(client.created_scores) == 2
    assert all(score["session_id"].startswith("run-themes-") for score in client.created_scores)
    assert len(dataset.evaluations) == 2


def test_run_themes_prompt_experiment_clones_requested_labels(tmp_path, monkeypatch):
    job = _write_job_artifacts(tmp_path)
    context = load_themes_experiment_context(job.id, scratch_root=tmp_path)
    dataset = _FakeDataset(context.input_data, context.expected_output)
    client = _FakeClient(dataset)

    monkeypatch.setattr("recalld.experiments.langfuse_themes.get_langfuse_client", lambda: client)
    monkeypatch.setattr("recalld.experiments.langfuse_themes.shutdown_tracing", lambda: None)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="test-model"),
    )

    async def fake_context_length(*args, **kwargs):
        return 1000

    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.ensure_loaded_context_length",
        fake_context_length,
    )

    async def fake_propose_themes(**kwargs):
        return ThemeProposalResult(
            themes=[],
            raw_response="{}",
            strategy="single",
            topic_count=0,
        )

    monkeypatch.setattr("recalld.experiments.langfuse_themes.propose_themes", fake_propose_themes)

    run_themes_prompt_experiment(
        job_id=job.id,
        prompt_labels=["production", "candidate"],
        scratch_root=tmp_path,
        clone_from_label="production",
    )

    assert len(client.updated_prompts) == 2
    assert all(call["new_labels"] == ["candidate"] for call in client.updated_prompts)


def test_run_themes_prompt_experiment_waits_for_each_model_before_continuing(tmp_path, monkeypatch):
    job = _write_job_artifacts(tmp_path)
    context = load_themes_experiment_context(job.id, scratch_root=tmp_path)
    dataset = _FakeDataset(context.input_data, context.expected_output)
    client = _FakeClient(dataset)
    order: list[tuple[str, str, int | None]] = []

    monkeypatch.setattr("recalld.experiments.langfuse_themes.get_langfuse_client", lambda: client)
    monkeypatch.setattr("recalld.experiments.langfuse_themes.shutdown_tracing", lambda: None)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="fallback-model"),
    )

    async def fake_list_models(base_url: str, selected_model: str):
        return [
            ProviderModel(id="google/gemma-4-e4b", max_context_length=131072, selected=selected_model == "google/gemma-4-e4b"),
            ProviderModel(id="google/gemma-4-e2b", max_context_length=131072, selected=selected_model == "google/gemma-4-e2b"),
        ]

    monkeypatch.setattr("recalld.experiments.langfuse_themes.list_available_models", fake_list_models)

    async def fake_context_length(base_url: str, model: str, requested_context_length: int | None = None):
        order.append(("load", model, requested_context_length))
        return requested_context_length or 131072

    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.ensure_loaded_context_length",
        fake_context_length,
    )

    async def fake_propose_themes(**kwargs):
        order.append(("run", kwargs["llm_model"], kwargs["token_budget"]))
        return ThemeProposalResult(
            themes=[],
            raw_response="{}",
            strategy="single",
            topic_count=0,
        )

    monkeypatch.setattr("recalld.experiments.langfuse_themes.propose_themes", fake_propose_themes)

    run_themes_prompt_experiment(
        job_id=job.id,
        prompt_labels=["production"],
        llm_models=["google/gemma-4-e4b", "google/gemma-4-e2b"],
        context_length=32768,
        scratch_root=tmp_path,
    )

    assert order == [
        ("load", "google/gemma-4-e4b", 32768),
        ("run", "google/gemma-4-e4b", 26214),
        ("load", "google/gemma-4-e2b", 32768),
        ("run", "google/gemma-4-e2b", 26214),
    ]


def test_run_themes_prompt_experiment_wraps_session_context(tmp_path, monkeypatch):
    job = _write_job_artifacts(tmp_path)
    context = load_themes_experiment_context(job.id, scratch_root=tmp_path)
    dataset = _FakeDataset(context.input_data, context.expected_output)
    client = _FakeClient(dataset)
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr("recalld.experiments.langfuse_themes.get_langfuse_client", lambda: client)
    monkeypatch.setattr("recalld.experiments.langfuse_themes.shutdown_tracing", lambda: None)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.session_context",
        lambda session_id: _FakeSessionContext(calls, session_id),
    )
    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.make_session_id",
        lambda *args, **kwargs: f"run-session-{args[1]}",
    )
    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="fallback-model"),
    )

    async def fake_context_length(*args, **kwargs):
        return 1000

    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.ensure_loaded_context_length",
        fake_context_length,
    )

    async def fake_propose_themes(**kwargs):
        return ThemeProposalResult(
            themes=[],
            raw_response="{}",
            strategy="single",
            topic_count=0,
        )

    monkeypatch.setattr("recalld.experiments.langfuse_themes.propose_themes", fake_propose_themes)

    run_themes_prompt_experiment(
        job_id=job.id,
        prompt_labels=["production"],
        scratch_root=tmp_path,
    )

    expected_session = f"run-session-{job.id[-6:]}"
    assert calls == [("session_enter", expected_session), ("session_exit", expected_session)]
