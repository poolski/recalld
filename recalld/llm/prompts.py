from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Any

from recalld.llm.prompt_cache import get_cached_prompt
from recalld.tracing import get_langfuse_client


def _render_fallback(template: str, variables: dict[str, Any]) -> str:
    rendered = template
    for name, value in variables.items():
        rendered = rendered.replace(f"{{{name}}}", str(value))
    return rendered


def _prompt_value(prompt: Any, variables: dict[str, Any]) -> str:
    compiler = getattr(prompt, "compile", None)
    if callable(compiler):
        compiled = compiler(**variables)
        if isinstance(compiled, str):
            return compiled
        return str(compiled)
    if isinstance(prompt, str):
        return prompt
    return str(prompt)


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _prompt_metadata(prompt: Any | None, source: str, prompt_name: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "prompt_name": prompt_name,
        "prompt_source": source,
    }
    if prompt is None:
        return metadata

    version = getattr(prompt, "version", None)
    if version is not None:
        metadata["prompt_version"] = version

    labels = getattr(prompt, "labels", None)
    if labels:
        if isinstance(labels, str):
            metadata["prompt_labels"] = [labels]
        else:
            metadata["prompt_labels"] = list(labels)

    return metadata


@dataclass(frozen=True)
class ResolvedPrompt:
    name: str
    text: str
    prompt: Any | None
    source: str
    metadata: dict[str, Any]


def _resolve_prompt_label(prompt_label: str | None) -> str:
    if prompt_label:
        return prompt_label
    return os.getenv("LANGFUSE_PROMPT_LABEL", "production")


def resolve_text_prompt(
    prompt_name: str,
    fallback: str,
    *,
    prompt_label: str | None = None,
    **variables: Any,
) -> ResolvedPrompt:
    client = get_langfuse_client()
    if client is not None:
        getter = getattr(client, "get_prompt", None)
        if callable(getter):
            label = _resolve_prompt_label(prompt_label)
            for kwargs in (
                {"label": label, "type": "text"},
                {"label": label, "prompt_type": "text"},
                {"label": label},
            ):
                try:
                    prompt = getter(prompt_name, **kwargs)
                except TypeError:
                    continue
                except Exception:
                    break
                if prompt is None:
                    continue
                text = _prompt_value(prompt, variables)
                meta = _prompt_metadata(prompt, "langfuse", prompt_name)
                meta["prompt_hash"] = _prompt_hash(text)
                return ResolvedPrompt(
                    name=prompt_name,
                    text=text,
                    prompt=prompt,
                    source="langfuse",
                    metadata=meta,
                )

    cached_text = get_cached_prompt(prompt_name)
    if cached_text is not None:
        text = _render_fallback(cached_text, variables)
        meta = _prompt_metadata(None, "cache", prompt_name)
        meta["prompt_hash"] = _prompt_hash(text)
        return ResolvedPrompt(
            name=prompt_name,
            text=text,
            prompt=None,
            source="cache",
            metadata=meta,
        )

    text = _render_fallback(fallback, variables)
    meta = _prompt_metadata(None, "fallback", prompt_name)
    meta["prompt_hash"] = _prompt_hash(text)
    return ResolvedPrompt(
        name=prompt_name,
        text=text,
        prompt=None,
        source="fallback",
        metadata=meta,
    )
