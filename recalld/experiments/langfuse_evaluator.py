from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from langfuse import Evaluation

from recalld.llm.client import LLMClient, complete_with_prompt
from recalld.llm.prompts import resolve_text_prompt

EXPERIMENT_QUALITY_SCORE_NAME = "experiment_quality"
EXPERIMENT_EVALUATOR_PROMPT_NAME = "recalld/experiment-evaluator"

FALLBACK_EVALUATOR_PROMPT = """\
You are a strict but fair judge for recalld experiments.

Given the input, expected output, actual output, and metadata, assign one score from 0.0 to 1.0.

Use the following criteria, in order:
- factual fidelity: stay grounded in the provided input and expected output
- coverage: include the important points without unnecessary omissions
- structure: follow the shape and organization that fit the task
- constraint adherence: avoid unsupported content, generic filler, duplication, and formatting mistakes

Task-specific guidance:
- For summaries, reward coherent thematic structure, accurate coverage, and concise but complete language.
- For theme extraction, reward concrete, distinct headings that map cleanly to the transcript and avoid broad umbrella labels.
- For other tasks, emphasize the task-specific criteria while remaining grounded in the provided input and expected output.

Return only JSON in this shape:
{
  "score": 0.0,
  "reason": "short explanation",
  "strengths": ["..."],
  "issues": ["..."]
}

Keep the reason short and concrete. Do not add extra keys, markdown, or commentary.
"""


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _extract_json(raw: str) -> dict[str, Any] | None:
    cleaned = (raw or "").strip()
    if not cleaned:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def build_experiment_quality_evaluator(
    *,
    llm_base_url: str,
    llm_model: str,
    prompt_label: str | None = None,
) -> Callable[..., Any]:
    evaluation_model = os.getenv("LANGFUSE_EVALUATION_MODEL", llm_model)
    evaluation_prompt_label = prompt_label or os.getenv("LANGFUSE_EVALUATION_PROMPT_LABEL", "production")
    client = LLMClient(base_url=llm_base_url, model=evaluation_model)

    async def evaluator(*, input, output, expected_output, metadata, **kwargs) -> Evaluation:
        prompt = resolve_text_prompt(
            EXPERIMENT_EVALUATOR_PROMPT_NAME,
            FALLBACK_EVALUATOR_PROMPT,
            prompt_label=evaluation_prompt_label,
        )
        user_prompt = _json_text(
            {
                "task_name": (metadata or {}).get("experiment_name", "unknown"),
                "prompt_label": (metadata or {}).get("prompt_label", "unknown"),
                "input": input,
                "expected_output": expected_output,
                "output": output,
            }
        )
        try:
            raw = await complete_with_prompt(
                client,
                prompt.text,
                user_prompt,
                prompt=prompt.prompt,
                metadata=prompt.metadata,
            )
            parsed = _extract_json(raw) or {}
            score = _clamp_score(parsed.get("score"))
            reason = str(parsed.get("reason", "")).strip()
            if not reason:
                reason = "No judge reason returned."
            return Evaluation(name=EXPERIMENT_QUALITY_SCORE_NAME, value=score, comment=reason)
        except Exception as exc:
            return Evaluation(
                name=EXPERIMENT_QUALITY_SCORE_NAME,
                value=0.0,
                comment=f"evaluation failed: {exc}",
            )

    return evaluator
