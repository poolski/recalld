from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class SpeakerTurn:
    start: float   # seconds
    end: float
    speaker: str   # e.g. "SPEAKER_00"


class DiariseError(Exception):
    pass


def diarise(
    wav_path: Path,
    huggingface_token: str,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> list[SpeakerTurn]:
    """
    Run speaker diarisation on wav_path using pyannote.audio on MPS.
    Returns list of speaker turns.
    Requires HuggingFace token with pyannote licence accepted.
    """
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        raise DiariseError("pyannote.audio not installed")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=huggingface_token,
        )
        pipeline.to(torch.device(device))
    except Exception as exc:
        raise DiariseError(f"Failed to load pyannote pipeline: {exc}") from exc

    diarization = pipeline(str(wav_path))

    turns: list[SpeakerTurn] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(SpeakerTurn(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
        ))

    if progress_cb:
        progress_cb(1.0)

    return turns
