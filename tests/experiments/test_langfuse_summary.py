from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from recalld.config import Config
from recalld.jobs import Job, save_job
from recalld.pipeline.postprocess import PostProcessResult
from recalld.experiments.langfuse_summary import (
    clone_prompt_label,
    load_summary_experiment_context,
    run_summary_prompt_experiment,
)


class _FakeRunResult:
    def __init__(self, calls):
        self.calls = calls

    def format(self):
        return "formatted"


class _FakeDataset:
    def __init__(self, input_data, expected_output):
        self.input_data = input_data
        self.expected_output = expected_output
        self.calls = []
        self.evaluations = []

    def run_experiment(self, **kwargs):
        self.calls.append(kwargs)
        task = kwargs["task"]
        item = SimpleNamespace(input=self.input_data, expected_output=self.expected_output)
        output = task(item=item)
        if asyncio.iscoroutine(output):
            output = asyncio.run(output)
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
        return _FakeRunResult(self.calls)


class _FakeClient:
    def __init__(self, dataset):
        self.dataset = dataset
        self.created_dataset = None
        self.created_item = None
        self.dataset_created = False
        self.updated_prompts = []

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
        "recalld.experiments.langfuse_summary.load_config",
        lambda: Config(llm_base_url="http://localhost:1234/v1", llm_model="test-model"),
    )
    async def fake_context_length(*args, **kwargs):
        return 1000

    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.ensure_loaded_context_length",
        fake_context_length,
    )

    async def fake_postprocess(**kwargs):
        captured_calls.append(kwargs)
        return PostProcessResult(
            summary="candidate summary",
            focus_points=["candidate focus"],
            raw_response="",
            strategy="single",
            topic_count=1,
        )

    monkeypatch.setattr("recalld.experiments.langfuse_summary.postprocess", fake_postprocess)

    results = run_summary_prompt_experiment(
        job_id=job.id,
        prompt_labels=["production", "candidate"],
        scratch_root=tmp_path,
    )

    assert [call["metadata"]["prompt_label"] for call in dataset.calls] == ["production", "candidate"]
    assert [call["prompt_label"] for call in captured_calls] == ["production", "candidate"]
    assert [result["prompt_label"] for result in results] == ["production", "candidate"]
    assert client.created_dataset["name"] == context.dataset_name
    assert client.created_item["id"] == context.dataset_item_id
    assert len(dataset.evaluations) == 2


def test_clone_prompt_label_updates_all_summary_prompts(monkeypatch):
    client = _FakeClient(None)

    clone_prompt_label(source_label="production", target_label="candidate", client=client)

    assert len(client.updated_prompts) == 4
    assert all(call["new_labels"] == ["candidate"] for call in client.updated_prompts)
