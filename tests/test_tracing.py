from __future__ import annotations

import os
from datetime import datetime, timezone

from recalld.tracing import experiment_tracing_environment, job_session_token, shutdown_tracing, start_observation
from recalld.tracing import get_tracing_environment, make_session_id
from recalld.tracing import update_current_span


class _FakeObservation:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    def __enter__(self):
        self._calls.append(("enter", {}))
        return self

    def __exit__(self, exc_type, exc, tb):
        self._calls.append(("exit", {"exc_type": exc_type}))
        return False

    def update(self, **kwargs):
        self._calls.append(("update", kwargs))


class _FakeClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.shutdown_called = False
        self.updated_spans: list[dict] = []

    def start_as_current_observation(self, **kwargs):
        self.calls.append(("start", kwargs))
        return _FakeObservation(self.calls)

    def shutdown(self):
        self.shutdown_called = True

    def update_current_span(self, **kwargs):
        self.updated_spans.append(kwargs)


class _FakePropagation:
    def __init__(self, calls: list[tuple[str, dict]], kwargs: dict):
        self.calls = calls
        self.kwargs = kwargs

    def __enter__(self):
        self.calls.append(("propagate_enter", self.kwargs))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append(("propagate_exit", {"exc_type": exc_type}))
        return False


def test_start_observation_delegates_to_langfuse_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr("recalld.tracing._get_client", lambda: client)

    with start_observation(
        name="job-pipeline",
        as_type="span",
        input={"job_id": "job-123"},
        metadata={"feature": "pipeline"},
    ) as observation:
        observation.update(output={"status": "ok"})

    assert client.calls[0][0] == "start"
    assert client.calls[0][1]["name"] == "job-pipeline"
    assert client.calls[0][1]["as_type"] == "span"
    assert client.calls[0][1]["input"] == {"job_id": "job-123"}
    assert client.calls[0][1]["metadata"] == {"feature": "pipeline"}
    assert client.calls[1] == ("enter", {})
    assert client.calls[2] == ("update", {"output": {"status": "ok"}})
    assert client.calls[3][0] == "exit"


def test_start_observation_forwards_prompt_and_session_context(monkeypatch):
    client = _FakeClient()
    prompt = object()
    monkeypatch.setattr("recalld.tracing._get_client", lambda: client)

    with start_observation(
        name="job-pipeline",
        as_type="span",
        prompt=prompt,
        trace_context={"trace_id": "trace-123"},
    ):
        pass

    assert client.calls[0][1]["prompt"] is prompt
    assert client.calls[0][1]["trace_context"] == {"trace_id": "trace-123"}


def test_start_observation_propagates_session_id(monkeypatch):
    calls: list[tuple[str, dict]] = []
    client = _FakeClient()
    client.calls = calls
    monkeypatch.setattr("recalld.tracing._get_client", lambda: client)
    monkeypatch.setattr("recalld.tracing._propagate_attributes", lambda **kwargs: _FakePropagation(calls, kwargs))

    with start_observation(
        name="job-pipeline",
        as_type="span",
        session_id="session-123",
    ):
        pass

    assert calls[0] == ("propagate_enter", {"session_id": "session-123"})
    assert calls[1][0] == "start"
    assert calls[2][0] == "enter"
    assert calls[-1] == ("propagate_exit", {"exc_type": None})


def test_make_session_id_is_ascii_and_readable():
    session_id = make_session_id(
        "style",
        "trimmed.m4a",
        "production",
        "google/gemma-4-e4b",
        "ctx32768",
        timestamp=datetime(2026, 5, 7, 12, 34, 56, tzinfo=timezone.utc),
    )

    assert session_id == "run-style-trimmed-m4a-production-google-gemma-4-e4b-ctx32768-20260507-123456"


def test_job_session_token_uses_last_chars():
    assert job_session_token("b7d9d444-9a67-4dc2-9825-d27f67335ca0") == "335ca0"


def test_shutdown_tracing_calls_langfuse_shutdown(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr("recalld.tracing._get_client", lambda: client)

    shutdown_tracing()

    assert client.shutdown_called is True


def test_update_current_span_forwards_to_langfuse_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr("recalld.tracing._get_client", lambda: client)

    update_current_span(name="summarise · job-123 · production")

    assert client.updated_spans == [{"name": "summarise · job-123 · production"}]


def test_get_tracing_environment_defaults_to_testing_under_pytest(monkeypatch):
    monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_tracing.py::test_case (call)")

    assert get_tracing_environment() == "testing"


def test_experiment_tracing_environment_sets_and_restores_env(monkeypatch):
    monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    with experiment_tracing_environment():
        assert os.environ["LANGFUSE_TRACING_ENVIRONMENT"] == "experiments"
        assert get_tracing_environment() == "experiments"

    assert "LANGFUSE_TRACING_ENVIRONMENT" not in os.environ
