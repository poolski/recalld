import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from recalld.pipeline.ingest import ingest, IngestError


def test_ingest_wav_passthrough(tmp_path):
    # A .wav file should be copied as-is without calling ffmpeg
    src = tmp_path / "audio.wav"
    src.write_bytes(b"RIFF" + b"\x00" * 40)  # minimal fake wav
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    result = ingest(src, out_dir)
    assert result.name == "audio.wav"
    assert result.exists()


def test_ingest_non_wav_calls_ffmpeg(tmp_path):
    src = tmp_path / "recording.m4a"
    src.write_bytes(b"fake m4a")
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    with patch("recalld.pipeline.ingest.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = ingest(src, out_dir)
    mock_run.assert_called_once()
    assert result.suffix == ".wav"


def test_ingest_ffmpeg_failure_raises(tmp_path):
    src = tmp_path / "recording.mp4"
    src.write_bytes(b"fake mp4")
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    with patch("recalld.pipeline.ingest.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        with pytest.raises(IngestError):
            ingest(src, out_dir)


def test_ingest_missing_ffmpeg_raises(tmp_path):
    src = tmp_path / "recording.mp4"
    src.write_bytes(b"fake mp4")
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    with patch("recalld.pipeline.ingest.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(IngestError, match="ffmpeg not found"):
            ingest(src, out_dir)
