from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import os
from typing import Any, Iterator


class _NoopObservation:
    trace_id: str | None = None
    id: str | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, **kwargs):
        return None


def _has_langfuse_credentials() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_tracing_environment() -> str:
    existing = os.getenv("LANGFUSE_TRACING_ENVIRONMENT")
    if existing:
        return existing
    if os.getenv("PYTEST_CURRENT_TEST"):
        return "testing"
    return "default"


@lru_cache(maxsize=1)
def _get_client():
    if not _has_langfuse_credentials():
        return None

    environment = get_tracing_environment()
    if environment:
        os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", environment)

    try:
        from langfuse import get_client
    except Exception:
        return None

    try:
        return get_client()
    except Exception:
        return None


def get_langfuse_client():
    return _get_client()


def create_trace_id(seed: str | None = None) -> str | None:
    client = _get_client()
    if client is None:
        return None

    create = getattr(client, "create_trace_id", None)
    if not callable(create):
        return None

    try:
        return create(seed=seed)
    except Exception:
        return None


def update_current_span(**kwargs: Any) -> None:
    client = _get_client()
    if client is None:
        return

    update = getattr(client, "update_current_span", None)
    if not callable(update):
        return

    try:
        update(**kwargs)
    except Exception:
        return


@contextmanager
def start_observation(
    *,
    name: str,
    as_type: str = "span",
    input: Any | None = None,
    output: Any | None = None,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
    trace_context: dict[str, str] | None = None,
    prompt: Any | None = None,
) -> Iterator[Any]:
    client = _get_client()
    if client is None:
        yield _NoopObservation()
        return

    kwargs: dict[str, Any] = {"name": name, "as_type": as_type}
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    if model is not None:
        kwargs["model"] = model
    if metadata is not None:
        kwargs["metadata"] = metadata
    if trace_context is not None:
        kwargs["trace_context"] = trace_context
    if prompt is not None:
        kwargs["prompt"] = prompt

    with client.start_as_current_observation(**kwargs) as observation:
        yield observation


def shutdown_tracing() -> None:
    client = _get_client()
    if client is None:
        return

    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            return
        return

    flush = getattr(client, "flush", None)
    if callable(flush):
        try:
            flush()
        except Exception:
            return
