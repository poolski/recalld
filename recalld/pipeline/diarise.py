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


def _turns_from_itertracks(diarization) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker))
    return turns


def _turns_from_speaker_diarization(diarization) -> list[SpeakerTurn]:
    raw_turns = getattr(diarization, "speaker_diarization", None)
    if raw_turns is None:
        raise DiariseError("Unsupported diarization output: missing speaker turns")

    turns: list[SpeakerTurn] = []
    for raw_turn in raw_turns:
        if isinstance(raw_turn, dict):
            segment = raw_turn.get("segment")
            speaker = raw_turn.get("speaker")
        else:
            segment = getattr(raw_turn, "segment", None)
            speaker = getattr(raw_turn, "speaker", None)

        if segment is None or speaker is None:
            continue

        turns.append(SpeakerTurn(start=segment.start, end=segment.end, speaker=speaker))

    if not turns:
        raise DiariseError("Unsupported diarization output: no usable speaker turns")

    return turns


def _extract_annotation(diarization):
    exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
    if hasattr(exclusive, "itertracks"):
        return exclusive

    speaker_diarization = getattr(diarization, "speaker_diarization", None)
    if hasattr(speaker_diarization, "itertracks"):
        return speaker_diarization

    return None


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
            token=huggingface_token,
        )
        if pipeline is None:
            raise DiariseError("Failed to load pyannote pipeline: returned None")
        pipeline.to(torch.device(device))
    except Exception as exc:
        raise DiariseError(f"Failed to load pyannote pipeline: {exc}") from exc

    diarization = pipeline(str(wav_path))

    if hasattr(diarization, "itertracks"):
        turns = _turns_from_itertracks(diarization)
    elif (annotation := _extract_annotation(diarization)) is not None:
        turns = _turns_from_itertracks(annotation)
    else:
        turns = _turns_from_speaker_diarization(diarization)

    if progress_cb:
        progress_cb(1.0)

    return turns
