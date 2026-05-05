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


async def _fake_stream(system, user, event_cb=None):
    yield "## Summary\n\n"
    yield "A productive session exploring focus strategies.\n\n"
    yield "## Focus\n\n"
    yield "- [ ] Start mornings with planning\n"
    yield "- [ ] Revisit good-enough conversation\n"


@pytest.mark.asyncio
async def test_postprocess_returns_summary_and_focus():
    turns = _turns([("You", "I struggle with mornings"), ("Coach", "Let's explore that")])
    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(return_value="- direct\n- concise")
        instance.stream = _fake_stream
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

    async def fake_stream(system, user, event_cb=None):
        nonlocal call_count
        call_count += 1
        async for t in _fake_stream(system, user):
            yield t

    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = fake_complete
        instance.stream = fake_stream
        progress = []
        await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=50,
            progress_cb=lambda msg: progress.append(msg),
        )
    # Map (several calls to complete) + Reduce (one call to stream)
    assert call_count > 1
    assert any("Detecting style from transcript sample." in p for p in progress)
    assert any("Selecting summarization strategy." in p for p in progress)
    assert any("Chunking strategy: map-reduce across" in p for p in progress)
    assert any("Reducing chunk summaries into final transcript summary." in p for p in progress)


@pytest.mark.asyncio
async def test_postprocess_uses_single_request_when_transcript_fits_provider_budget():
    turns = _turns([("You", "word " * 120), ("Coach", "word " * 120)])
    call_count = 0

    async def fake_stream(system, user, event_cb=None):
        nonlocal call_count
        call_count += 1
        async for t in _fake_stream(system, user):
            yield t

    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(return_value="- direct\n- concise")
        instance.stream = fake_stream
        progress = []
        result = await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=2000,
            progress_cb=lambda msg: progress.append(msg),
        )

    assert call_count == 1
    assert result.strategy == "single"
    assert any("Chunking strategy: single-pass summary" in p for p in progress)


@pytest.mark.asyncio
async def test_postprocess_includes_configured_speaker_names_in_system_prompt():
    turns = _turns([("Alex", "I struggled with focus this week"), ("Jordan", "What pattern did you notice?")])
    captured = {}

    async def fake_stream(system, user, event_cb=None):
        captured["system"] = system
        async for t in _fake_stream(system, user):
            yield t

    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(return_value="- direct\n- concise")
        instance.stream = fake_stream
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
    assert "Paragraphs must be separated by blank lines." in captured["system"]
    assert "Style profile (from transcript sample):" in captured["system"]


@pytest.mark.asyncio
async def test_postprocess_calls_stream_cb_with_partial_summary():
    turns = _turns([("You", "hi"), ("Coach", "hello")])
    updates = []

    async def fake_stream(system, user, event_cb=None):
        yield "## Summary\n\nPart 1"
        yield " Part 2"

    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(return_value="- direct\n- concise")
        instance.stream = fake_stream
        await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=1000,
            stream_cb=lambda text: updates.append(text)
        )

    assert "Part 1" in updates[0]
    assert "Part 1 Part 2" in updates[1]


@pytest.mark.asyncio
async def test_postprocess_forwards_lmstudio_stream_events():
    turns = _turns([("You", "hi"), ("Coach", "hello")])
    captured = []

    async def fake_stream(system, user, event_cb=None):
        if event_cb:
            event_cb("prompt_processing.start", {"type": "prompt_processing.start"})
            event_cb("reasoning.delta", {"type": "reasoning.delta", "content": "Thinking"})
            event_cb("message.delta", {"type": "message.delta", "content": "## Summary\n\nDone"})
            event_cb(
                "chat.end",
                {"type": "chat.end", "result": {"output": [{"type": "message", "content": "## Summary\n\nDone"}]}},
            )
        yield "## Summary\n\nDone"

    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(return_value="- direct\n- concise")
        instance.stream = fake_stream
        await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=1000,
            event_cb=lambda event_type, data: captured.append(event_type),
        )

    assert captured == [
        "prompt_processing.start",
        "reasoning.delta",
        "message.delta",
        "chat.end",
    ]


def test_parse_focus_points_from_markdown():
    from recalld.pipeline.postprocess import parse_focus_points
    md = "## Focus\n\n- [ ] Do thing one\n- [ ] Do thing two\n\nExtra text"
    points = parse_focus_points(md)
    assert points == ["Do thing one", "Do thing two"]


def test_postprocess_prompts_require_detailed_transcript_grounded_summary():
    from recalld.pipeline.postprocess import (
        SYSTEM_PROMPT_TEMPLATE,
        MAP_SYSTEM_PROMPT,
        REDUCE_SYSTEM_PROMPT_TEMPLATE,
        STYLE_ANALYSIS_SYSTEM_PROMPT,
    )

    assert "Cover the discussion in detail, proportional to transcript depth and duration." in SYSTEM_PROMPT_TEMPLATE
    assert "Extract specific topics discussed, concrete suggestions made, decisions reached, and open questions." in SYSTEM_PROMPT_TEMPLATE
    assert "Include only what is stated in the transcript; do not infer facts or inject opinions." in SYSTEM_PROMPT_TEMPLATE
    assert "Use a direct, pragmatic, no-fluff style." in SYSTEM_PROMPT_TEMPLATE
    assert "Prioritize concrete facts and discussion details over narrative filler." in SYSTEM_PROMPT_TEMPLATE
    assert "Follow the provided style profile closely for wording, cadence, and register." in SYSTEM_PROMPT_TEMPLATE
    assert "The style profile controls phrasing only; it must not change or add facts." in SYSTEM_PROMPT_TEMPLATE
    assert "If style guidance conflicts with transcript fidelity, transcript fidelity wins." in SYSTEM_PROMPT_TEMPLATE
    assert "Use 3-5 concise paragraphs" not in SYSTEM_PROMPT_TEMPLATE
    assert "Paragraphs must be separated by blank lines." in SYSTEM_PROMPT_TEMPLATE

    assert "Use a direct, pragmatic, no-fluff style." in MAP_SYSTEM_PROMPT
    assert "Extract concrete details: topics discussed, suggestions made, decisions, and open follow-ups." in MAP_SYSTEM_PROMPT
    assert "Use only transcript-grounded facts, with no opinions or invented details." in MAP_SYSTEM_PROMPT
    assert "Align phrasing with the provided style profile when present." in MAP_SYSTEM_PROMPT

    assert "Use a direct, pragmatic, no-fluff style." in REDUCE_SYSTEM_PROMPT_TEMPLATE
    assert "Prioritize concrete facts and discussion details over narrative filler." in REDUCE_SYSTEM_PROMPT_TEMPLATE
    assert "Preserve detailed factual coverage from the partial summaries." in REDUCE_SYSTEM_PROMPT_TEMPLATE
    assert "Include only what is evidenced in the summaries; do not add opinions or inferred facts." in REDUCE_SYSTEM_PROMPT_TEMPLATE
    assert "Follow the provided style profile closely for wording, cadence, and register." in REDUCE_SYSTEM_PROMPT_TEMPLATE
    assert "The style profile controls phrasing only; it must not change or add facts." in REDUCE_SYSTEM_PROMPT_TEMPLATE
    assert "extract writing style characteristics" in STYLE_ANALYSIS_SYSTEM_PROMPT.lower()
