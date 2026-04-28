from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "recalld" / "config.json"


class Category(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    vault_path: str
    focus_note_path: Optional[str] = None
    speaker_a: str = "Speaker A"
    speaker_b: str = "Speaker B"


class Config(BaseModel):
    obsidian_api_url: str = "https://127.0.0.1:27124"
    obsidian_api_key: str = ""
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = ""
    llm_context_headroom: float = 0.8
    log_level: str = "info"
    last_used_category: Optional[str] = None
    categories: list[Category] = Field(default_factory=list)
    whisper_model: str = "small"
    huggingface_token: str = ""
    scratch_retention_days: int = 30


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    if path.exists():
        return Config.model_validate_json(path.read_text())
    return Config()


def save_config(cfg: Config, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.model_dump_json(indent=2))
