from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class IngestError(Exception):
    pass


def ingest(source: Path, job_dir: Path) -> Path:
    """Extract audio to wav. Returns path to wav file in job_dir."""
    if source.suffix.lower() == ".wav":
        dest = job_dir / source.name
        shutil.copy2(source, dest)
        return dest

    dest = job_dir / (source.stem + ".wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(dest),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise IngestError("ffmpeg not found — install it with: brew install ffmpeg")

    if result.returncode != 0:
        raise IngestError(f"ffmpeg failed: {result.stderr}")

    return dest
