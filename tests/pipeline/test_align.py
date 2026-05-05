from recalld.pipeline.align import align, LabelledTurn
from recalld.pipeline.transcribe import WordSegment
from recalld.pipeline.diarise import SpeakerTurn


def _words(text: str, start: float, end: float) -> list[WordSegment]:
    words = text.split()
    step = (end - start) / max(len(words), 1)
    return [
        WordSegment(start=start + i * step, end=start + (i + 1) * step, word=w)
        for i, w in enumerate(words)
    ]


def test_single_speaker_turn():
    words = _words("Hello world", 0.0, 2.0)
    turns = [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]
    result = align(words, turns)
    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"
    assert "Hello" in result[0].text


def test_two_speaker_turns():
    words = _words("Hello there how are you", 0.0, 5.0)
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.0, end=5.0, speaker="SPEAKER_01"),
    ]
    result = align(words, turns)
    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].speaker == "SPEAKER_01"


def test_word_assigned_to_overlapping_turn():
    # Word at 1.5s should go to the turn that covers it
    words = [WordSegment(start=1.5, end=2.0, word="hi")]
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=1.0, end=3.0, speaker="SPEAKER_01"),
    ]
    result = align(words, turns)
    assigned = [t for t in result if t.text.strip()]
    assert assigned[0].speaker == "SPEAKER_01"


def test_empty_turns_returns_unlabelled():
    words = _words("some text", 0.0, 2.0)
    result = align(words, [])
    assert len(result) == 1
    assert result[0].speaker == "UNKNOWN"


def test_speaker_name_substitution():
    words = _words("Hello there", 0.0, 2.0)
    turns = [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]
    speaker_map = {"SPEAKER_00": "You", "SPEAKER_01": "Facilitator"}
    result = align(words, turns, speaker_map=speaker_map)
    assert result[0].speaker == "You"
