from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.themes import ThemeSuggestion, propose_themes


def _turns(pairs: list[tuple[str, str]]) -> list[LabelledTurn]:
    turns = []
    t = 0.0
    for speaker, text in pairs:
        turns.append(LabelledTurn(speaker=speaker, start=t, end=t + 5, text=text))
        t += 5
    return turns


@pytest.mark.asyncio
async def test_propose_themes_parses_theme_suggestions_from_model_response():
    turns = _turns([("You", "I need to plan the launch"), ("Facilitator", "What are the next steps?")])

    async def fake_complete(system, user):
        return """
        [
          {"id": "theme-1", "title": "Launch planning", "notes": "Timeline and sequencing", "enabled": true},
          {"id": "theme-2", "title": "Open questions", "notes": "", "enabled": true}
        ]
        """

    with patch("recalld.pipeline.themes.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(side_effect=fake_complete)
        result = await propose_themes(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=2000,
        )

    assert result.strategy == "single"
    assert result.topic_count == 2
    assert result.themes == [
        ThemeSuggestion(id="theme-1", title="Launch planning", notes="Timeline and sequencing", enabled=True, order=1, source="transcript"),
        ThemeSuggestion(id="theme-2", title="Open questions", notes="", enabled=True, order=2, source="transcript"),
    ]


@pytest.mark.asyncio
async def test_propose_themes_returns_empty_list_when_model_output_is_invalid():
    turns = _turns([("You", "I need to plan the launch"), ("Facilitator", "What are the next steps?")])

    async def fake_complete(system, user):
        return "not json"

    with patch("recalld.pipeline.themes.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(side_effect=fake_complete)
        result = await propose_themes(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=2000,
        )

    assert result.themes == []
    assert result.topic_count == 0


@pytest.mark.asyncio
async def test_propose_themes_resolves_runtime_prompts_and_forwards_prompt_context():
    turns = _turns([("You", "I need to plan the launch"), ("Facilitator", "What are the next steps?")])
    resolved = {
        "recalld/themes-single": SimpleNamespace(
            text="themes-single-system",
            prompt="themes-single-prompt",
            metadata={"prompt_name": "recalld/themes-single", "prompt_source": "langfuse"},
        ),
        "recalld/themes-map": SimpleNamespace(
            text="themes-map-system",
            prompt="themes-map-prompt",
            metadata={"prompt_name": "recalld/themes-map", "prompt_source": "langfuse"},
        ),
    }
    complete_calls = []
    resolve_calls = []

    def fake_resolve(name, fallback, **variables):
        resolve_calls.append((name, variables))
        return resolved[name]

    async def fake_complete(system, user, prompt=None, metadata=None, preset=None):
        complete_calls.append((system, prompt, metadata, preset))
        return """
        [
          {"id": "theme-1", "title": "Launch planning", "notes": "Timeline and sequencing", "enabled": true},
          {"id": "theme-2", "title": "Open questions", "notes": "", "enabled": true}
        ]
        """

    with patch("recalld.pipeline.themes.resolve_text_prompt", side_effect=fake_resolve), \
         patch("recalld.pipeline.themes.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(side_effect=fake_complete)
        result = await propose_themes(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=2000,
            prompt_label="candidate",
        )

    assert result.strategy == "single"
    assert complete_calls[0][1] == "themes-single-prompt"
    assert complete_calls[0][2]["prompt_name"] == "recalld/themes-single"
    assert complete_calls[0][3] == "@local:transcript-summariser"
    assert all(call[1].get("prompt_label") == "candidate" for call in resolve_calls)
