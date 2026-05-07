from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.postprocess import _sample_style_window, postprocess, PostProcessResult


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
    turns = _turns([("You", "I struggle with mornings"), ("Facilitator", "Let's explore that")])
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
async def test_postprocess_resolves_runtime_prompts_and_forwards_prompt_context():
    long_sample = " ".join(
        [
            "I need to rethink the morning routine and protect deep work time.",
            "I need to rethink the morning routine and protect deep work time.",
            "I need to rethink the morning routine and protect deep work time.",
            "I need to rethink the morning routine and protect deep work time.",
            "I need to rethink the morning routine and protect deep work time.",
            "I need to rethink the morning routine and protect deep work time.",
        ]
    )
    turns = _turns([
        ("You", long_sample),
        ("Facilitator", "Let's explore that in detail and break down the current routine."),
    ])
    resolved = {
        "recalld/postprocess-style-analysis": SimpleNamespace(
            text="style-analysis-system",
            prompt="style-prompt",
            metadata={"prompt_name": "recalld/postprocess-style-analysis", "prompt_source": "langfuse"},
        ),
        "recalld/postprocess-summary-single": SimpleNamespace(
            text="summary-single-system",
            prompt="summary-single-prompt",
            metadata={"prompt_name": "recalld/postprocess-summary-single", "prompt_source": "langfuse"},
        ),
        "recalld/postprocess-summary-reduce": SimpleNamespace(
            text="summary-reduce-system",
            prompt="summary-reduce-prompt",
            metadata={"prompt_name": "recalld/postprocess-summary-reduce", "prompt_source": "langfuse"},
        ),
        "recalld/postprocess-summary-map": SimpleNamespace(
            text="summary-map-system",
            prompt="summary-map-prompt",
            metadata={"prompt_name": "recalld/postprocess-summary-map", "prompt_source": "langfuse"},
        ),
    }
    complete_calls = []
    stream_calls = []

    def fake_resolve(name, fallback, **variables):
        return resolved[name]

    async def fake_complete(system, user, prompt=None, metadata=None):
        complete_calls.append((system, prompt, metadata))
        return "- direct\n- concise"

    async def fake_stream(system, user, event_cb=None, prompt=None, metadata=None):
        stream_calls.append((system, prompt, metadata))
        yield "## Summary\n\nA productive session exploring focus strategies.\n\n## Focus\n\n- [ ] Start mornings with planning\n"

    with patch("recalld.pipeline.postprocess.resolve_text_prompt", side_effect=fake_resolve), \
         patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = fake_complete
        instance.stream = fake_stream
        result = await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=10000,
        )

    assert result.strategy == "single"
    assert complete_calls[0][1] == "style-prompt"
    assert complete_calls[0][2]["prompt_name"] == "recalld/postprocess-style-analysis"
    assert stream_calls[0][1] == "summary-single-prompt"
    assert stream_calls[0][2]["prompt_name"] == "recalld/postprocess-summary-single"


@pytest.mark.asyncio
async def test_postprocess_map_reduce_calls_llm_multiple_times():
    # Create enough turns to trigger map_reduce (budget=50)
    turns = _turns([("You", "word " * 30), ("Facilitator", "word " * 30)] * 5)
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
    turns = _turns([("You", "word " * 120), ("Facilitator", "word " * 120)])
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
async def test_postprocess_includes_existing_note_scaffold_in_system_prompt():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
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
            existing_note_content="## Summary\n- terse reminder\n\n## Projects\n- follow up",
        )

    assert "Existing note content to expand" in captured["system"]
    assert "Expand shorthand sections" in captured["system"]
    assert "## Projects" in captured["system"]


@pytest.mark.asyncio
async def test_postprocess_includes_confirmed_theme_guidance_in_system_prompt():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
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
            theme_guidance=[
                {"title": "Planning next steps", "notes": "Decisions and follow-ups", "enabled": True, "order": 1},
                {"title": "Risks", "notes": "", "enabled": False, "order": 2},
            ],
        )

    system = captured["system"]
    assert "Confirmed theme guidance" in system
    assert "Planning next steps" in system
    assert "Decisions and follow-ups" in system
    assert "Risks" not in system
    assert "disabled" not in system.lower()


@pytest.mark.asyncio
async def test_postprocess_places_theme_guidance_before_existing_note_scaffold():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
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
            existing_note_content="## Summary\n- terse reminder\n\n## Projects\n- follow up",
            theme_guidance=[
                {"title": "Planning next steps", "notes": "Decisions and follow-ups", "enabled": True, "order": 1},
            ],
        )

    system = captured["system"]
    assert system.index("Confirmed theme guidance") < system.index("Existing note content to expand")
    assert "confirmed theme guidance is the primary organizational guidance" in system.lower()


@pytest.mark.asyncio
async def test_postprocess_says_existing_links_and_embeds_are_untouchable_with_theme_guidance():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
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
            existing_note_content="## Summary\n- terse reminder with [[Project Board]] and ![[Daily Note]]",
            theme_guidance=[
                {"title": "Planning next steps", "notes": "Decisions and follow-ups", "enabled": True, "order": 1},
            ],
        )

    system = captured["system"].lower()
    assert "existing links, embeds, and link targets are authoritative" in system
    assert "do not remove or rewrite them" in system


@pytest.mark.asyncio
async def test_postprocess_tells_model_to_add_other_relevant_points_under_their_own_headings():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
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
            existing_note_content="## Summary\n- terse reminder\n\n## Projects\n- follow up",
        )

    system = captured["system"].lower()
    assert "other relevant discussion points under their own headings" in system


@pytest.mark.asyncio
async def test_postprocess_allows_inline_rewording_while_preserving_markdown_structure():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
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
            existing_note_content="## Summary\n- terse reminder\n\n## For next time\n- revisit planning",
        )

    system = captured["system"].lower()
    assert "preserve the overall markdown structure" in system
    assert "expand or reword text within a section inline" in system


@pytest.mark.asyncio
async def test_postprocess_preserves_links_and_embeds_when_rewording_inline():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
    captured = {}

    async def fake_stream(system, user, event_cb=None):
        captured["system"] = system
        async for t in _fake_stream(system, user):
            yield t

    existing_note = "\n".join(
        [
            "## Summary",
            "- Keep this linked note: [[Project Board]] and [spec](https://example.com/spec).",
            "- Include the embed ![[Daily Note]].",
            "",
            "## For next time",
            "- revisit the plan with [[Alice]] and [notes](https://example.com/notes)",
        ]
    )

    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(return_value="- direct\n- concise")
        instance.stream = fake_stream
        await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=10000,
            existing_note_content=existing_note,
        )

    system = captured["system"].lower()
    assert "preserve links, embeds, and link targets" in system
    assert "keep links and embeds in a context that still makes sense" in system
    assert "[[project board]]" in system
    assert "https://example.com/spec" in system


def test_style_sample_window_uses_a_single_speaker_when_preferred_label_is_missing():
    turns = _turns(
        [
            ("Alice", "I like direct language"),
            ("Bob", "I like long explanations"),
            ("Alice", "Keep it concrete"),
            ("Bob", "Add detail"),
        ]
    )

    sample = _sample_style_window(turns, speaker_a_name="You", seconds=60.0)

    assert "Alice:" in sample
    assert "Bob:" not in sample


@pytest.mark.asyncio
async def test_postprocess_tells_model_to_preserve_existing_note_structure():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
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
            existing_note_content="## Summary\n- terse reminder\n\n## Projects\n- follow up",
        )

    system = captured["system"].lower()
    assert "preserve existing headings" in system
    assert "preserve the note's overview" in system
    assert "continue existing sections" in system
    assert "add new headings when the transcript introduces distinct themes" in system


@pytest.mark.asyncio
async def test_postprocess_tells_model_to_use_thematic_sections_on_blank_slate():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
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
        )

    system = captured["system"].lower()
    assert "blank slate" in system
    assert "thematic sections" in system
    assert "chronological block" in system


@pytest.mark.asyncio
async def test_postprocess_tells_model_to_add_new_headings_and_expand_lists():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
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
            existing_note_content="\n".join(
                [
                    "## Summary",
                    "1. Keep the first point as written.",
                    "2. Keep the second point as written.",
                    "",
                    "## Projects",
                    "- follow up",
                ]
            ),
        )

    system = captured["system"].lower()
    assert "add a new heading for it when helpful" in system
    assert "do not preserve lists verbatim" in system


@pytest.mark.asyncio
async def test_postprocess_expands_existing_followup_heading_instead_of_synthesizing_focus():
    turns = _turns([("You", "I want to keep this note terse"), ("Facilitator", "What belongs in the summary?")])
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
            existing_note_content="## Summary\n- terse reminder\n\n## For next time\n- revisit planning",
        )

    system = captured["system"].lower()
    assert "for next time" in system
    assert "expand it with more detail" in system
    assert "instead of creating a separate" in system


@pytest.mark.asyncio
async def test_postprocess_calls_stream_cb_with_partial_summary():
    turns = _turns([("You", "hi"), ("Facilitator", "hello")])
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
    turns = _turns([("You", "hi"), ("Facilitator", "hello")])
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


def test_parse_summary_preserves_headings_inside_summary_and_stops_before_focus():
    from recalld.pipeline.postprocess import parse_summary

    md = "\n".join(
        [
            "## Summary",
            "",
            "## Overview",
            "The main discussion covered priorities and next steps.",
            "",
            "## Details",
            "A linked note like [project notes](https://example.com/notes) should stay visible.",
            "",
            "## Focus",
            "",
            "- [ ] Follow up on the next steps",
        ]
    )

    summary = parse_summary(md)

    assert "## Overview" in summary
    assert "## Details" in summary
    assert "## Focus" not in summary
    assert "project notes" in summary


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
    assert "use only one speaker's turns" in STYLE_ANALYSIS_SYSTEM_PROMPT.lower()
    assert "extract writing style characteristics" in STYLE_ANALYSIS_SYSTEM_PROMPT.lower()
