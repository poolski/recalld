"""Experiment helpers for Langfuse tuning runs."""
from __future__ import annotations

from typing import Any


def prompt_version(client: Any, prompt_name: str, label: str) -> str | None:
    """Return the version number for a prompt label, or None if unavailable."""
    try:
        p = client.get_prompt(prompt_name, label=label)
        v = getattr(p, "version", None)
        return str(v) if v is not None else None
    except Exception:
        return None


def experiment_description(
    *,
    exp_type: str,
    filename: str,
    prompt_label: str,
    prompt_version: str | None,
    llm_model: str,
    all_labels: list[str],
    run_tag: str | None = None,
) -> str:
    label_str = f"{prompt_label} (v{prompt_version})" if prompt_version else prompt_label
    if len(all_labels) > 1:
        other = [l for l in all_labels if l != prompt_label]
        action = f"Comparing {label_str} against {', '.join(other)}"
    else:
        action = f"Baseline run of {label_str}"
    parts = [f"{action} {exp_type} prompt on \"{filename}\" using {llm_model}."]
    if run_tag:
        parts.append(f"Run tag: {run_tag}.")
    return " ".join(parts)
