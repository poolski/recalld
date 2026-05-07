from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace


def _load_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "langfuse"
    loader = SourceFileLoader("langfuse_script", str(path))
    spec = importlib.util.spec_from_loader("langfuse_script", loader)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_prompt_create_uses_file_and_labels(tmp_path, monkeypatch):
    script = _load_script()
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Hello {{name}}")

    class _FakeClient:
        def __init__(self):
            self.calls = []

        def create_prompt(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(version=2)

    client = _FakeClient()
    monkeypatch.setattr(script, "_get_client", lambda: client)

    args = SimpleNamespace(
        name="recalld/example",
        type="text",
        prompt=None,
        prompt_file=str(prompt_file),
        label=["candidate"],
        commit_message="test",
    )

    assert script._cmd_prompt_create(args) == 0
    assert client.calls[0]["name"] == "recalld/example"
    assert client.calls[0]["prompt"] == "Hello {{name}}"
    assert client.calls[0]["labels"] == ["candidate"]


def test_summary_experiment_dispatches_to_recalld_runner(monkeypatch):
    script = _load_script()
    captured = {}

    def fake_run_summary_prompt_experiment(**kwargs):
        captured.update(kwargs)
        return [{"prompt_label": "candidate", "run_name": "run-1", "result": SimpleNamespace(format=lambda: "ok")}]

    monkeypatch.setattr(
        "recalld.experiments.langfuse_summary.run_summary_prompt_experiment",
        fake_run_summary_prompt_experiment,
    )

    args = SimpleNamespace(
        job_id="job-123",
        prompt_label=["candidate"],
        clone_from_label="production",
        dataset_name="dataset-1",
        scratch_root="/tmp/scratch",
    )

    assert script._cmd_summary_experiment(args) == 0
    assert captured["job_id"] == "job-123"
    assert captured["prompt_labels"] == ["candidate"]
    assert captured["clone_from_label"] == "production"


def test_themes_experiment_dispatches_to_recalld_runner(monkeypatch):
    script = _load_script()
    captured = {}

    def fake_run_themes_prompt_experiment(**kwargs):
        captured.update(kwargs)
        return [{"prompt_label": "candidate", "run_name": "run-1", "result": SimpleNamespace(format=lambda: "ok")}]

    monkeypatch.setattr(
        "recalld.experiments.langfuse_themes.run_themes_prompt_experiment",
        fake_run_themes_prompt_experiment,
    )

    args = SimpleNamespace(
        job_id="job-123",
        prompt_label=["candidate"],
        clone_from_label="production",
        dataset_name="dataset-1",
        scratch_root="/tmp/scratch",
    )

    assert script._cmd_themes_experiment(args) == 0
    assert captured["job_id"] == "job-123"
    assert captured["prompt_labels"] == ["candidate"]
    assert captured["clone_from_label"] == "production"


def test_focus_experiment_dispatches_to_recalld_runner(monkeypatch):
    script = _load_script()
    captured = {}

    def fake_run_focus_prompt_experiment(**kwargs):
        captured.update(kwargs)
        return [{"prompt_label": "candidate", "run_name": "run-1", "result": SimpleNamespace(format=lambda: "ok")}]

    monkeypatch.setattr(
        "recalld.experiments.langfuse_focus.run_focus_prompt_experiment",
        fake_run_focus_prompt_experiment,
    )

    args = SimpleNamespace(
        job_id="job-123",
        prompt_label=["candidate"],
        clone_from_label="production",
        dataset_name="dataset-1",
        scratch_root="/tmp/scratch",
    )

    assert script._cmd_focus_experiment(args) == 0
    assert captured["job_id"] == "job-123"
    assert captured["prompt_labels"] == ["candidate"]
    assert captured["clone_from_label"] == "production"
