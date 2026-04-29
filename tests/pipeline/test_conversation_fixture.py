from __future__ import annotations

import os
from pathlib import Path

import pytest

from recalld.pipeline.transcribe import transcribe


CONVERSATION_FIXTURE = Path("tests/fixtures/conversation.wav")
EXPECTED_LINES = [
    ("af_heart", "Hey, did you finish the report?"),
    ("am_adam", "Not yet, I'm adding the final numbers now."),
    ("af_heart", "Alright, send it over when you're done."),
    ("am_adam", "Will do, give me two minutes."),
]


def test_conversation_fixture_exists():
    assert CONVERSATION_FIXTURE.exists()
    assert CONVERSATION_FIXTURE.suffix == ".wav"


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
