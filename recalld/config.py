from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "recalld" / "config.json"


class Category(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    vault_path: str
    focus_note_path: Optional[str] = None
    speaker_a: str = "Speaker A"
    speaker_b: str = "Speaker B"


class Config(BaseModel):
    vault_name: str = "Personal"
    obsidian_api_url: str = "https://127.0.0.1:27124"
    obsidian_api_key: str = ""
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = ""
    llm_context_headroom: float = 0.8
    llm_reasoning: str = "off"
    log_level: str = "info"
    last_used_category: Optional[str] = None
    categories: list[Category] = Field(default_factory=list)
    whisper_model: str = "small"
    huggingface_token: str = ""
    scratch_retention_days: int = 30

    @field_validator("llm_reasoning", mode="before")
    @classmethod
    def _normalize_llm_reasoning(cls, value):
        if isinstance(value, str) and value.strip().lower() == "on":
            return "on"
        return "off"


def _resolve_config_path(path: Path | None) -> Path:
    return path or DEFAULT_CONFIG_PATH


def load_config(path: Path | None = None) -> Config:
    path = _resolve_config_path(path)
    if path.exists():
        try:
            return Config.model_validate_json(path.read_text())
        except Exception as exc:
            raise ValueError(
                f"Config file at {path} is invalid: {exc}\n"
                "Fix or delete the file and restart recalld."
            ) from exc
    return Config()


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = _resolve_config_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.model_dump_json(indent=2))
    os.chmod(path, 0o600)
