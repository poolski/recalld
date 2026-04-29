import pytest
from unittest.mock import AsyncMock, patch
from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.postprocess import postprocess, PostProcessResult


def _turns(pairs: list[tuple[str, str]]) -> list[LabelledTurn]:
    turns = []
    t = 0.0
    for speaker, text in pairs:
        turns.append(LabelledTurn(speaker=speaker, start=t, end=t + 5, text=text))
        t += 5
    return turns


FAKE_LLM_RESPONSE = """## Summary

A productive session exploring focus strategies.

## Focus

- [ ] Start mornings with planning
- [ ] Revisit good-enough conversation
"""


@pytest.mark.asyncio
async def test_postprocess_returns_summary_and_focus():
    turns = _turns([("You", "I struggle with mornings"), ("Coach", "Let's explore that")])
    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(return_value=FAKE_LLM_RESPONSE)
        result = await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=10000,
        )
    assert "productive session" in result.summary
    assert len(result.focus_points) == 2
    assert "Start mornings" in result.focus_points[0]


@pytest.mark.asyncio
async def test_postprocess_map_reduce_calls_llm_multiple_times():
    # Create enough turns to trigger map_reduce (budget=50)
    turns = _turns([("You", "word " * 30), ("Coach", "word " * 30)] * 5)
    call_count = 0

    async def fake_complete(system, user):
        nonlocal call_count
        call_count += 1
        return FAKE_LLM_RESPONSE

    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = fake_complete
        await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=50,
        )
    assert call_count > 1


@pytest.mark.asyncio
async def test_postprocess_uses_single_request_when_transcript_fits_provider_budget():
    turns = _turns([("You", "word " * 120), ("Coach", "word " * 120)])
    call_count = 0

    async def fake_complete(system, user):
        nonlocal call_count
        call_count += 1
        return FAKE_LLM_RESPONSE

    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = fake_complete
        result = await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=1000,
        )

    assert call_count == 1
    assert result.strategy == "single"


@pytest.mark.asyncio
async def test_postprocess_includes_configured_speaker_names_in_system_prompt():
    turns = _turns([("Alex", "I struggled with focus this week"), ("Jordan", "What pattern did you notice?")])
    captured = {}

    async def fake_complete(system, user):
        captured["system"] = system
        return FAKE_LLM_RESPONSE

    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = fake_complete
        await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=10000,
            speaker_a_name="Alex",
            speaker_b_name="Jordan",
        )

    assert "Alex" in captured["system"]
    assert "Jordan" in captured["system"]
    assert "Refer to Alex as \"you\"" in captured["system"]


def test_parse_focus_points_from_markdown():
    from recalld.pipeline.postprocess import parse_focus_points
    md = "## Focus\n\n- [ ] Do thing one\n- [ ] Do thing two\n\nExtra text"
    points = parse_focus_points(md)
    assert points == ["Do thing one", "Do thing two"]
