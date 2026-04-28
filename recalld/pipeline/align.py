from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from recalld.pipeline.diarise import SpeakerTurn
from recalld.pipeline.transcribe import WordSegment


@dataclass
class LabelledTurn:
    speaker: str
    start: float
    end: float
    text: str = ""


def align(
    words: list[WordSegment],
    turns: list[SpeakerTurn],
    speaker_map: Optional[dict[str, str]] = None,
) -> list[LabelledTurn]:
    """
    Assign each word to the speaker turn that contains its midpoint.
    Consecutive words from the same speaker are merged into one LabelledTurn.
    """
    if not turns:
        text = " ".join(w.word for w in words)
        return [LabelledTurn(
            speaker="UNKNOWN",
            start=words[0].start if words else 0,
            end=words[-1].end if words else 0,
            text=text,
        )]

    def speaker_at(t: float) -> str:
        for turn in turns:
            if turn.start <= t < turn.end:
                label = turn.speaker
                if speaker_map:
                    label = speaker_map.get(label, label)
                return label
        # Assign to nearest turn
        nearest = min(turns, key=lambda s: abs(s.start - t))
        label = nearest.speaker
        if speaker_map:
            label = speaker_map.get(label, label)
        return label

    labelled: list[LabelledTurn] = []
    for word in words:
        mid = (word.start + word.end) / 2
        spk = speaker_at(mid)
        if labelled and labelled[-1].speaker == spk:
            labelled[-1].text += " " + word.word
            labelled[-1].end = word.end
        else:
            labelled.append(LabelledTurn(
                speaker=spk,
                start=word.start,
                end=word.end,
                text=word.word,
            ))

    return labelled
