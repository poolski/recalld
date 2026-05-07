from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from recalld.experiments.langfuse_evaluator import (
    EXPERIMENT_QUALITY_SCORE_NAME,
    build_experiment_quality_evaluator,
)


@pytest.mark.asyncio
async def test_experiment_quality_evaluator_uses_runtime_prompt_and_parses_json():
    captured = {}

    resolved = SimpleNamespace(
        text="judge-system",
        prompt="judge-prompt",
        metadata={"prompt_name": "recalld/experiment-evaluator", "prompt_source": "langfuse"},
    )

    def fake_resolve(name, fallback, **variables):
        captured["resolve"] = (name, fallback, variables)
        return resolved

    async def fake_complete(system, user, prompt=None, metadata=None):
        captured["complete"] = (system, user, prompt, metadata)
        return '{"score": 0.83, "reason": "good enough", "strengths": ["clear"], "issues": []}'

    with patch("recalld.experiments.langfuse_evaluator.resolve_text_prompt", side_effect=fake_resolve), \
         patch("recalld.experiments.langfuse_evaluator.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(side_effect=fake_complete)

        evaluator = build_experiment_quality_evaluator(
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
        )
        result = await evaluator(
            input={"task": "summary"},
            output={"summary": "ok"},
            expected_output={"summary": "ref"},
            metadata={"experiment_name": "summary", "prompt_label": "production"},
        )

    assert result.name == EXPERIMENT_QUALITY_SCORE_NAME
    assert result.value == 0.83
    assert result.comment == "good enough"
    assert captured["resolve"][0] == "recalld/experiment-evaluator"
    assert captured["complete"][0] == "judge-system"
    assert '"task_name": "summary"' in captured["complete"][1]
    assert captured["complete"][2] == "judge-prompt"


@pytest.mark.asyncio
async def test_experiment_quality_evaluator_falls_back_to_zero_on_bad_json():
    with patch("recalld.experiments.langfuse_evaluator.resolve_text_prompt") as resolve_prompt, \
         patch("recalld.experiments.langfuse_evaluator.LLMClient") as MockClient:
        resolve_prompt.return_value = SimpleNamespace(
            text="judge-system",
            prompt="judge-prompt",
            metadata={"prompt_name": "recalld/experiment-evaluator", "prompt_source": "langfuse"},
        )
        instance = MockClient.return_value
        instance.complete = AsyncMock(return_value="not json")

        evaluator = build_experiment_quality_evaluator(
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
        )
        result = await evaluator(
            input={"task": "themes"},
            output={"themes": []},
            expected_output={"themes": []},
            metadata={"experiment_name": "themes"},
        )

    assert result.name == EXPERIMENT_QUALITY_SCORE_NAME
    assert result.value == 0.0
    assert result.comment == "No judge reason returned."
