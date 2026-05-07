from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import re
import os
import unicodedata
from datetime import datetime, timezone
from typing import Any, Iterator

try:
    from langfuse import propagate_attributes as _propagate_attributes
except Exception:  # pragma: no cover - optional SDK feature
    _propagate_attributes = None


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


def _slugify_session_part(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text)
    return text.strip("-").lower()


def make_session_id(
    step: str | None = None,
    *parts: Any,
    prefix: str = "run",
    timestamp: datetime | None = None,
) -> str | None:
    tokens = []
    if step is not None:
        step_token = _slugify_session_part(step)
        if step_token:
            tokens.append(step_token)
    for part in parts:
        if part is None:
            continue
        token = _slugify_session_part(part)
        if token:
            tokens.append(token)
    if timestamp is not None:
        ts = timestamp.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
        tokens.append(ts)
    if not tokens:
        return None
    session_id = "-".join([prefix, *tokens]) if prefix else "-".join(tokens)
    return session_id[:200]


def job_session_token(job_id: str | None, length: int = 6) -> str | None:
    if not job_id:
        return None
    text = str(job_id).strip()
    if not text:
        return None
    return text[-length:]


@contextmanager
def experiment_tracing_environment(environment: str = "experiments") -> Iterator[None]:
    previous = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT")
    if environment:
        os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = environment
    else:
        os.environ.pop("LANGFUSE_TRACING_ENVIRONMENT", None)
    _get_client.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LANGFUSE_TRACING_ENVIRONMENT", None)
        else:
            os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = previous
        _get_client.cache_clear()


@contextmanager
def session_context(session_id: str | None) -> Iterator[None]:
    if not session_id or _propagate_attributes is None:
        yield
        return

    with _propagate_attributes(session_id=session_id):
        yield


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
    session_id: str | None = None,
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

    with session_context(session_id):
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
