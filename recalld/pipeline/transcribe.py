from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class WordSegment:
    start: float   # seconds
    end: float
    word: str


class TranscribeError(Exception):
    pass


def transcribe(
    wav_path: Path,
    model_name: str = "small",
    progress_cb: Optional[Callable[[float], None]] = None,
) -> list[WordSegment]:
    """
    Transcribe wav_path using whisper.cpp via pywhispercpp.
    Returns list of word-level timestamped segments.
    progress_cb called with 0.0–1.0 as transcription progresses.
    """
    try:
        from pywhispercpp.model import Model
    except ImportError:
        raise TranscribeError("pywhispercpp not installed")

    model = Model(model_name, print_realtime=False, print_progress=False)
    segments = model.transcribe(str(wav_path))

    words: list[WordSegment] = []
    for seg in segments:
        # pywhispercpp segments have .words list with .start/.end in ms, .word
        for w in getattr(seg, "words", []):
            words.append(WordSegment(
                start=w.start / 1000.0,
                end=w.end / 1000.0,
                word=w.word,
            ))

    if not words:
        # Fall back to segment-level if no word timestamps available
        for seg in segments:
            words.append(WordSegment(
                start=seg.t0 / 100.0,
                end=seg.t1 / 100.0,
                word=seg.text.strip(),
            ))

    if progress_cb:
        progress_cb(1.0)

    return words
