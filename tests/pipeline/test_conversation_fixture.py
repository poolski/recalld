from __future__ import annotations

import os
from pathlib import Path

import pytest
from unittest.mock import patch

from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.postprocess import postprocess
from recalld.pipeline.transcribe import transcribe


CONVERSATION_FIXTURE = Path("tests/fixtures/conversation.wav")
CHUNKING_FIXTURE = Path("tests/fixtures/chunking_5min.m4a")
EXPECTED_LINES = [
    ("af_heart", "Hey, did you finish the report?"),
    ("am_adam", "Not yet, I'm adding the final numbers now."),
    ("af_heart", "Alright, send it over when you're done."),
    ("am_adam", "Will do, give me two minutes."),
]


def test_conversation_fixture_exists():
    assert CONVERSATION_FIXTURE.exists()
    assert CONVERSATION_FIXTURE.suffix == ".wav"


def test_chunking_fixture_exists():
    assert CHUNKING_FIXTURE.exists()
    assert CHUNKING_FIXTURE.suffix == ".m4a"


@pytest.mark.skipif(
    os.environ.get("RECALLD_RUN_AUDIO_FIXTURES") != "1",
    reason="Set RECALLD_RUN_AUDIO_FIXTURES=1 to run real transcription fixture checks.",
)
def test_transcribe_conversation_fixture_contains_expected_phrases():
    words = transcribe(CONVERSATION_FIXTURE, model_name="tiny")
    text = " ".join(word.word.strip() for word in words).lower()

    for _, expected in EXPECTED_LINES:
        expected_text = expected.lower().replace(",", "").replace("'", "")
        normalized = text.replace(",", "").replace("'", "")
        assert expected_text in normalized


def _words_to_turns(words: list, words_per_turn: int = 1) -> list[LabelledTurn]:
    turns: list[LabelledTurn] = []
    speakers = ["You", "Coach"]
    for index in range(0, len(words), words_per_turn):
        chunk = words[index:index + words_per_turn]
        if not chunk:
            continue
        text = " ".join(word.word.strip() for word in chunk).strip()
        turns.append(
            LabelledTurn(
                speaker=speakers[(index // words_per_turn) % 2],
                start=chunk[0].start,
                end=chunk[-1].end,
                text=text,
            )
        )
    return turns


@pytest.mark.skipif(
    os.environ.get("RECALLD_RUN_AUDIO_FIXTURES") != "1",
    reason="Set RECALLD_RUN_AUDIO_FIXTURES=1 to run real transcription fixture checks.",
)
@pytest.mark.asyncio
async def test_chunking_fixture_forces_map_reduce():
    words = transcribe(CHUNKING_FIXTURE, model_name="tiny")
    assert words

    turns = _words_to_turns(words)
    calls: list[tuple[str, int]] = []

    async def fake_complete(system, user):
        calls.append(("complete", len(user.split())))
        return "## Summary\n\nChunk summary.\n\n## Focus\n\n- [ ] Follow up"

    async def fake_stream(system, user, event_cb=None):
        calls.append(("stream", len(user.split())))
        yield "## Summary\n\n"
        yield "Combined summary.\n\n"
        yield "## Focus\n\n"
        yield "- [ ] Follow up\n"

    with patch("recalld.pipeline.postprocess.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = fake_complete
        instance.stream = fake_stream
        result = await postprocess(
            turns=turns,
            llm_base_url="http://localhost:1234/v1",
            llm_model="qwen",
            token_budget=1000,
        )

    assert result.strategy == "map_reduce"
    assert len([call for call in calls if call[0] == "complete"]) > 1
    assert len([call for call in calls if call[0] == "stream"]) == 1
