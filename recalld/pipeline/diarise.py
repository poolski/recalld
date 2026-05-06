from __future__ import annotations

import audioop
import tempfile
import time
import wave
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


SILENCE_WINDOW_SECONDS = 0.5
MIN_SILENCE_SECONDS = 1.5
MAX_CHUNK_SECONDS = 900.0
SILENCE_RMS_THRESHOLD = 500


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


def _wav_metadata(wav_path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(wav_path), "rb") as wav:
        return wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes):02d}m {remainder:04.1f}s"


def _detect_silence_ranges(
    wav_path: Path,
    *,
    silence_window_seconds: float = SILENCE_WINDOW_SECONDS,
    min_silence_seconds: float = MIN_SILENCE_SECONDS,
    silence_rms_threshold: int = SILENCE_RMS_THRESHOLD,
) -> list[tuple[int, int]]:
    channels, sample_width, frame_rate, total_frames = _wav_metadata(wav_path)
    window_frames = max(1, int(frame_rate * silence_window_seconds))
    min_silence_frames = max(1, int(frame_rate * min_silence_seconds))

    silence_ranges: list[tuple[int, int]] = []
    silence_start: int | None = None

    with wave.open(str(wav_path), "rb") as wav:
        frame_cursor = 0
        while frame_cursor < total_frames:
            chunk = wav.readframes(window_frames)
            if not chunk:
                break
            read_frames = len(chunk) // (channels * sample_width)
            if read_frames <= 0:
                break
            is_silent = audioop.rms(chunk, sample_width) < silence_rms_threshold
            if is_silent:
                if silence_start is None:
                    silence_start = frame_cursor
            elif silence_start is not None:
                if frame_cursor - silence_start >= min_silence_frames:
                    silence_ranges.append((silence_start, frame_cursor))
                silence_start = None
            frame_cursor += read_frames

        if silence_start is not None and frame_cursor - silence_start >= min_silence_frames:
            silence_ranges.append((silence_start, frame_cursor))

    return silence_ranges


def _choose_split_point(
    start_frame: int,
    target_frame: int,
    silence_ranges: list[tuple[int, int]],
) -> int | None:
    best: int | None = None
    for silence_start, silence_end in silence_ranges:
        if silence_end <= start_frame:
            continue
        if silence_start >= target_frame:
            break
        if silence_start <= target_frame <= silence_end:
            return target_frame
        if silence_end <= target_frame:
            best = silence_end
    return best


def _build_chunk_ranges(
    total_frames: int,
    silence_ranges: list[tuple[int, int]],
    *,
    frame_rate: int,
    max_chunk_seconds: float = MAX_CHUNK_SECONDS,
) -> list[tuple[int, int]]:
    max_chunk_frames = max(1, int(frame_rate * max_chunk_seconds))
    if total_frames <= max_chunk_frames:
        return [(0, total_frames)]

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        target = min(start + max_chunk_frames, total_frames)
        if target == total_frames:
            ranges.append((start, total_frames))
            break
        split_point = _choose_split_point(start, target, silence_ranges) or target
        if split_point <= start:
            split_point = target
        ranges.append((start, split_point))
        start = split_point
    return ranges


def _write_wav_segment(source: Path, destination: Path, start_frame: int, end_frame: int) -> None:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        with wave.open(str(destination), "wb") as writer:
            writer.setparams(params)
            reader.setpos(start_frame)
            frames_remaining = max(0, end_frame - start_frame)
            frames_per_read = 4096
            while frames_remaining > 0:
                frames_to_read = min(frames_per_read, frames_remaining)
                data = reader.readframes(frames_to_read)
                if not data:
                    break
                frames_written = len(data) // (params.nchannels * params.sampwidth)
                frames_remaining -= frames_written
                writer.writeframes(data)


def _chunk_wav_for_diarisation(wav_path: Path, work_dir: Path) -> list[tuple[Path, float, float]]:
    channels, sample_width, frame_rate, total_frames = _wav_metadata(wav_path)
    if total_frames == 0:
        return [(wav_path, 0.0, 0.0)]

    silence_ranges = _detect_silence_ranges(wav_path)
    chunk_ranges = _build_chunk_ranges(
        total_frames,
        silence_ranges,
        frame_rate=frame_rate,
        max_chunk_seconds=MAX_CHUNK_SECONDS,
    )
    if len(chunk_ranges) == 1:
        return [(wav_path, 0.0, total_frames / frame_rate)]

    chunks: list[tuple[Path, float, float]] = []
    for index, (start_frame, end_frame) in enumerate(chunk_ranges):
        chunk_path = work_dir / f"{wav_path.stem}-chunk-{index:03d}.wav"
        _write_wav_segment(wav_path, chunk_path, start_frame, end_frame)
        chunks.append((chunk_path, start_frame / frame_rate, (end_frame - start_frame) / frame_rate))
    return chunks


def _offset_turns(turns: list[SpeakerTurn], offset: float) -> list[SpeakerTurn]:
    return [
        SpeakerTurn(start=turn.start + offset, end=turn.end + offset, speaker=turn.speaker)
        for turn in turns
    ]


def _emit_progress(progress_cb: Optional[Callable[[str], None]], message: str) -> None:
    if progress_cb:
        progress_cb(message)


def diarise(
    wav_path: Path,
    huggingface_token: str,
    progress_cb: Optional[Callable[[str], None]] = None,
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

    total_started_at = time.perf_counter()

    _emit_progress(progress_cb, "Loading diarisation model.")

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=huggingface_token,
        )
        if pipeline is None:
            raise DiariseError("Failed to load pyannote pipeline: returned None")
        pipeline.to(torch.device(device))
    except Exception as exc:
        raise DiariseError(f"Failed to load pyannote pipeline: {exc}") from exc

    _emit_progress(progress_cb, "Diarisation model loaded.")

    all_turns: list[SpeakerTurn] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_plan_started_at = time.perf_counter()
        chunk_specs = _chunk_wav_for_diarisation(wav_path, Path(tmpdir))
        total_duration = sum(duration for _, _, duration in chunk_specs)
        _emit_progress(
            progress_cb,
            f"Planned {len(chunk_specs)} diarisation chunk(s) for {_format_duration(total_duration)} of audio in "
            f"{time.perf_counter() - chunk_plan_started_at:.1f}s.",
        )
        for index, (chunk_path, offset, duration) in enumerate(chunk_specs, start=1):
            chunk_started_at = time.perf_counter()
            _emit_progress(
                progress_cb,
                f"Diarising chunk {index}/{len(chunk_specs)} ({_format_duration(duration)}).",
            )
            diarization = pipeline(str(chunk_path))

            if hasattr(diarization, "itertracks"):
                turns = _turns_from_itertracks(diarization)
            elif (annotation := _extract_annotation(diarization)) is not None:
                turns = _turns_from_itertracks(annotation)
            else:
                turns = _turns_from_speaker_diarization(diarization)

            all_turns.extend(_offset_turns(turns, offset))
            _emit_progress(
                progress_cb,
                f"Finished chunk {index}/{len(chunk_specs)} in {time.perf_counter() - chunk_started_at:.1f}s.",
            )

    _emit_progress(
        progress_cb,
        f"Completed diarisation in {time.perf_counter() - total_started_at:.1f}s.",
    )

    return all_turns
