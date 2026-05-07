from __future__ import annotations

import time
from typing import Any


def _score_key(score: Any) -> tuple[Any, ...]:
    return (getattr(score, "name", None),)


def _score_kwargs(score: Any, *, session_id: str, trace_id: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "session_id": session_id,
        "name": score.name,
        "value": score.value,
        "comment": getattr(score, "comment", None),
        "metadata": getattr(score, "metadata", None) or {},
    }
    if getattr(score, "data_type", None) is not None:
        kwargs["data_type"] = score.data_type
    if getattr(score, "config_id", None) is not None:
        kwargs["config_id"] = score.config_id
    score_id = getattr(score, "id", None)
    if score_id:
        kwargs["score_id"] = f"{score_id}-session"
    return kwargs


def mirror_experiment_scores_to_session(
    client: Any,
    result: Any,
    *,
    session_id: str,
    metadata: dict[str, Any] | None = None,
    wait_for_scores: bool = False,
    settle_timeout_seconds: float = 5.0,
) -> int:
    create_score = getattr(client, "create_score", None)
    item_results = getattr(result, "item_results", None) or []
    if not callable(create_score) or not session_id or not item_results:
        return 0

    created = 0
    trace_get = getattr(getattr(client, "api", None), "trace", None)
    trace_getter = getattr(trace_get, "get", None)
    for item_result in item_results:
        trace_id = getattr(item_result, "trace_id", None)
        trace_scores = []
        if callable(trace_getter) and trace_id:
            initial_count: int | None = None
            try:
                deadline = time.monotonic() + settle_timeout_seconds
                while True:
                    trace = trace_getter(trace_id, fields="scores")
                    trace_scores = getattr(trace, "scores", None) or []
                    if not wait_for_scores:
                        break
                    if initial_count is None:
                        initial_count = len(trace_scores)
                    elif len(trace_scores) > initial_count:
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.5)
            except Exception:
                trace_scores = []
        seen_keys: set[tuple[Any, ...]] = set()
        for score in trace_scores:
            score_key = _score_key(score)
            if score_key in seen_keys:
                continue
            seen_keys.add(score_key)
            score_kwargs = _score_kwargs(score, session_id=session_id, trace_id=trace_id)
            score_kwargs["metadata"] = {
                **(getattr(score, "metadata", None) or {}),
                **(metadata or {}),
                **({"trace_id": trace_id} if trace_id else {}),
            }
            create_score(**score_kwargs)
            created += 1

        evaluations = getattr(item_result, "evaluations", None) or []
        for evaluation in evaluations:
            score_key = _score_key(evaluation)
            if score_key in seen_keys:
                continue
            seen_keys.add(score_key)
            eval_kwargs = _score_kwargs(evaluation, session_id=session_id, trace_id=trace_id)
            eval_kwargs["metadata"] = {
                **(getattr(evaluation, "metadata", None) or {}),
                **(metadata or {}),
                **({"trace_id": trace_id} if trace_id else {}),
            }
            create_score(**eval_kwargs)
            created += 1
    return created
