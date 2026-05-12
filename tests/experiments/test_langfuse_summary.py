from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from recalld.config import Config
from recalld.jobs import Job, save_job
from recalld.llm.context import ProviderModel
from recalld.pipeline.postprocess import PostProcessResult
from recalld.experiments.langfuse_summary import (
    clone_prompt_label,
    load_summary_experiment_context,
    run_summary_prompt_experiment,
)


class _FakeRunResult:
    def __init__(self, calls, item_results):
        self.calls = calls
        self.item_results = item_results

    def format(self):
        return "formatted"


class _FakeDataset:
    def __init__(self, input_data, expected_output, dataset_id="dataset-1"):
        self.input_data = input_data
        self.expected_output = expected_output
        self.id = dataset_id
        self.calls = []
        self.evaluations = []
        self.trace_score_batches = [
            [
                SimpleNamespace(
                    id="trace-score-1",
                    name="reference_alignment",
                    value=0.88,
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
                    value=0.88,
                    comment="grounding",
                    metadata={},
                    data_type="NUMERIC",
                    config_id=None,
                ),
                SimpleNamespace(
                    id="trace-score-2",
                    name="summary-evaluator",
                    value=0.88,
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
    job.theme_guidance = [{"id": "theme-1", "title": "Example", "notes": "Notes", "enabled": True, "order": 1, "source": "manual"}]
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


def test_load_summary_experiment_context_reads_artifacts(tmp_path):
    job = _write_job_artifacts(tmp_path)

    context = load_summary_experiment_context(job.id, scratch_root=tmp_path)

    assert context.job_id == job.id
    assert context.dataset_name == f"recalld/jobs/{job.id}/summary"
    assert context.dataset_item_id == f"job-{job.id}-summary"
    assert context.input_data["speaker_b_name"] == "Lizzie"
    assert context.expected_output["summary"] == "A short summary"
    assert context.expected_output["focus_points"] == ["Keep working"]


def test_run_summary_prompt_experiment_uses_each_requested_label(tmp_path, monkeypatch):
    job = _write_job_artifacts(tmp_path)
    context = load_summary_experiment_context(job.id, scratch_root=tmp_path)
    dataset = _FakeDataset(context.input_data, context.expected_output)
    client = _FakeClient(dataset)
    captured_calls: list[dict] = []

    monkeypatch.setattr("recalld.experiments.langfuse_summary.get_langfuse_client", lambda: client)
    monkeypatch.setattr("recalld.experiments.langfuse_summary.shutdown_tracing", lambda: None)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.ensure_evaluator_rules_include_dataset",
        lambda *args, **kwargs: [],
    )
    monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)
    captured_envs: list[str | None] = []

    def fake_get_client():
        captured_envs.append(os.getenv("LANGFUSE_TRACING_ENVIRONMENT"))
        return client

    monkeypatch.setattr("recalld.experiments.langfuse_summary.get_langfuse_client", fake_get_client)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="test-model"),
    )
    async def fake_context_length(*args, **kwargs):
        return 1000

    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.ensure_loaded_context_length",
        fake_context_length,
    )

    async def fake_generate_summary(**kwargs):
        captured_calls.append(kwargs)
        return PostProcessResult(
            summary="candidate summary",
            focus_points=["candidate focus"],
            raw_response="",
            strategy="single",
            topic_count=1,
        )

    monkeypatch.setattr("recalld.experiments.langfuse_summary.generate_summary", fake_generate_summary)

    results = run_summary_prompt_experiment(
        job_id=job.id,
        prompt_labels=["production", "candidate"],
        scratch_root=tmp_path,
    )

    assert [call["metadata"]["prompt_label"] for call in dataset.calls] == ["production", "candidate"]
    assert dataset.calls[0]["run_name"].startswith("summary · session · production · test-model · ctx1000")
    assert [call["prompt_label"] for call in captured_calls] == ["production", "candidate"]
    assert [result["prompt_label"] for result in results] == ["production", "candidate"]
    assert client.created_dataset["name"] == context.dataset_name
    assert client.created_item["id"] == context.dataset_item_id
    assert captured_envs == ["experiments"]
    assert "LANGFUSE_TRACING_ENVIRONMENT" not in os.environ
    assert len(client.created_scores) == 4
    assert all(score["session_id"].startswith("run-summary-") for score in client.created_scores)
    assert len(dataset.evaluations) == 2


def test_clone_prompt_label_updates_all_summary_prompts(monkeypatch):
    client = _FakeClient(None)

    clone_prompt_label(source_label="production", target_label="candidate", client=client)

    assert len(client.updated_prompts) == 4
    assert all(call["new_labels"] == ["candidate"] for call in client.updated_prompts)


def test_run_summary_prompt_experiment_waits_for_each_model_before_continuing(tmp_path, monkeypatch):
    job = _write_job_artifacts(tmp_path)
    context = load_summary_experiment_context(job.id, scratch_root=tmp_path)
    dataset = _FakeDataset(context.input_data, context.expected_output)
    client = _FakeClient(dataset)
    order: list[tuple[str, str, int | None]] = []

    monkeypatch.setattr("recalld.experiments.langfuse_summary.get_langfuse_client", lambda: client)
    monkeypatch.setattr("recalld.experiments.langfuse_summary.shutdown_tracing", lambda: None)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.ensure_evaluator_rules_include_dataset",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="fallback-model"),
    )

    async def fake_list_models(base_url: str, selected_model: str):
        return [
            ProviderModel(id="google/gemma-4-e4b", max_context_length=131072, selected=selected_model == "google/gemma-4-e4b"),
            ProviderModel(id="google/gemma-4-e2b", max_context_length=131072, selected=selected_model == "google/gemma-4-e2b"),
        ]

    monkeypatch.setattr("recalld.experiments.langfuse_summary.list_available_models", fake_list_models)

    async def fake_context_length(base_url: str, model: str, requested_context_length: int | None = None):
        order.append(("load", model, requested_context_length))
        return requested_context_length or 131072

    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.ensure_loaded_context_length",
        fake_context_length,
    )

    async def fake_generate_summary(**kwargs):
        order.append(("run", kwargs["llm_model"], kwargs["token_budget"]))
        return PostProcessResult(
            summary="candidate summary",
            focus_points=["candidate focus"],
            raw_response="",
            strategy="single",
            topic_count=1,
        )

    monkeypatch.setattr("recalld.experiments.langfuse_summary.generate_summary", fake_generate_summary)

    run_summary_prompt_experiment(
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


def test_run_summary_prompt_experiment_registers_dataset_filter(tmp_path, monkeypatch):
    job = _write_job_artifacts(tmp_path)
    context = load_summary_experiment_context(job.id, scratch_root=tmp_path)
    dataset = _FakeDataset(context.input_data, context.expected_output, dataset_id="dataset-summary")
    client = _FakeClient(dataset)
    captured: list[tuple[str, object]] = []

    monkeypatch.setattr("recalld.experiments.langfuse_summary.get_langfuse_client", lambda: client)
    monkeypatch.setattr("recalld.experiments.langfuse_summary.shutdown_tracing", lambda: None)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.ensure_evaluator_rules_include_dataset",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="fallback-model"),
    )

    async def fake_context_length(*args, **kwargs):
        return 1000

    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.ensure_loaded_context_length",
        fake_context_length,
    )

    def fake_register(dataset_id: str, *, client=None, **kwargs):
        captured.append((dataset_id, client))
        return ["rule-1"]

    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.ensure_evaluator_rules_include_dataset",
        fake_register,
    )

    async def fake_generate_summary(**kwargs):
        return PostProcessResult(
            summary="candidate summary",
            focus_points=["candidate focus"],
            raw_response="",
            strategy="single",
            topic_count=1,
        )

    monkeypatch.setattr("recalld.experiments.langfuse_summary.generate_summary", fake_generate_summary)

    run_summary_prompt_experiment(
        job_id=job.id,
        prompt_labels=["production"],
        scratch_root=tmp_path,
    )

    assert captured == [("dataset-summary", client)]


def test_run_summary_prompt_experiment_wraps_session_context(tmp_path, monkeypatch):
    job = _write_job_artifacts(tmp_path)
    context = load_summary_experiment_context(job.id, scratch_root=tmp_path)
    dataset = _FakeDataset(context.input_data, context.expected_output, dataset_id="dataset-summary")
    client = _FakeClient(dataset)
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr("recalld.experiments.langfuse_summary.get_langfuse_client", lambda: client)
    monkeypatch.setattr("recalld.experiments.langfuse_summary.shutdown_tracing", lambda: None)
    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.session_context",
        lambda session_id: _FakeSessionContext(calls, session_id),
    )
    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.make_session_id",
        lambda *args, **kwargs: f"run-session-{args[1]}",
    )
    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.ensure_evaluator_rules_include_dataset",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="fallback-model"),
    )

    async def fake_context_length(*args, **kwargs):
        return 1000

    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.ensure_loaded_context_length",
        fake_context_length,
    )

    async def fake_generate_summary(**kwargs):
        return PostProcessResult(
            summary="candidate summary",
            focus_points=["candidate focus"],
            raw_response="",
            strategy="single",
            topic_count=1,
        )

    monkeypatch.setattr("recalld.experiments.langfuse_summary.generate_summary", fake_generate_summary)

    run_summary_prompt_experiment(
        job_id=job.id,
        prompt_labels=["production"],
        scratch_root=tmp_path,
    )

    expected_session = f"run-session-{job.id[-6:]}"
    assert calls == [("session_enter", expected_session), ("session_exit", expected_session)]
