from __future__ import annotations

import itertools
from types import SimpleNamespace

from recalld.experiments.langfuse_session_scores import mirror_experiment_scores_to_session


class _FakeClient:
    def __init__(self):
        self.created_scores = []
        self.api = SimpleNamespace(trace=SimpleNamespace(get=self.get_trace))

    def get_trace(self, trace_id, fields=None):
        return SimpleNamespace(
            scores=[
                SimpleNamespace(
                    id="trace-score-1",
                    name="reference_alignment",
                    value=0.75,
                    comment="ok",
                    metadata={"source": "trace"},
                    data_type="NUMERIC",
                    config_id=None,
                )
            ]
        )

    def create_score(self, **kwargs):
        self.created_scores.append(kwargs)
        return kwargs


class _WaitingFakeClient:
    def __init__(self):
        self.created_scores = []
        self.api = SimpleNamespace(trace=SimpleNamespace(get=self.get_trace))
        self._trace_calls = 0
        self._trace_batches = [
            [
                SimpleNamespace(
                    id="trace-score-1",
                    name="reference_alignment",
                    value=0.75,
                    comment="ok",
                    metadata={"source": "trace"},
                    data_type="NUMERIC",
                    config_id=None,
                )
            ],
            [
                SimpleNamespace(
                    id="trace-score-1",
                    name="reference_alignment",
                    value=0.75,
                    comment="ok",
                    metadata={"source": "trace"},
                    data_type="NUMERIC",
                    config_id=None,
                ),
                SimpleNamespace(
                    id="trace-score-2",
                    name="focus-evaluator",
                    value=0.5,
                    comment="grounded",
                    metadata={"source": "trace"},
                    data_type="NUMERIC",
                    config_id=None,
                ),
            ],
        ]

    def get_trace(self, trace_id, fields=None):
        batch = self._trace_batches[min(self._trace_calls, len(self._trace_batches) - 1)]
        self._trace_calls += 1
        return SimpleNamespace(scores=batch)

    def create_score(self, **kwargs):
        self.created_scores.append(kwargs)
        return kwargs


def test_mirror_experiment_scores_to_session_uses_session_only_scores():
    client = _FakeClient()
    result = SimpleNamespace(
        item_results=[
            SimpleNamespace(
                trace_id="trace-123",
                evaluations=[
                    SimpleNamespace(
                        id="eval-1",
                        name="focus-evaluator",
                        value=0.5,
                        comment="grounded",
                        metadata={"source": "evaluation"},
                        data_type="NUMERIC",
                        config_id=None,
                    )
                ],
            )
        ]
    )

    created = mirror_experiment_scores_to_session(
        client,
        result,
        session_id="run-focus-abc123",
        metadata={"job_id": "job-1"},
    )

    assert created == 2
    assert len(client.created_scores) == 2
    assert all(score["session_id"] == "run-focus-abc123" for score in client.created_scores)
    assert all("trace_id" not in score for score in client.created_scores)
    assert client.created_scores[0]["metadata"] == {
        "source": "trace",
        "job_id": "job-1",
        "trace_id": "trace-123",
    }
    assert client.created_scores[1]["metadata"] == {
        "source": "evaluation",
        "job_id": "job-1",
        "trace_id": "trace-123",
    }


def test_mirror_experiment_scores_to_session_waits_for_additional_scores(monkeypatch):
    client = _WaitingFakeClient()
    result = SimpleNamespace(
        item_results=[
            SimpleNamespace(
                trace_id="trace-123",
                evaluations=[],
            )
        ]
    )
    sleeps: list[float] = []
    monotonic_values = itertools.chain([0.0, 0.1, 0.6, 1.1], itertools.repeat(1.1))

    monkeypatch.setattr("recalld.experiments.langfuse_session_scores.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("recalld.experiments.langfuse_session_scores.time.sleep", lambda seconds: sleeps.append(seconds))

    created = mirror_experiment_scores_to_session(
        client,
        result,
        session_id="run-focus-abc123",
        metadata={"job_id": "job-1"},
        wait_for_scores=True,
        settle_timeout_seconds=2.0,
    )

    assert created == 2
    assert [score["name"] for score in client.created_scores] == [
        "reference_alignment",
        "focus-evaluator",
    ]
    assert sleeps == [0.5]
