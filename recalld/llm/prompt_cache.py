from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_CACHE_PATH = Path.home() / ".config" / "recalld" / "prompts.json"

KNOWN_PROMPT_NAMES: tuple[str, ...] = (
    "recalld/postprocess-summary-single",
    "recalld/postprocess-summary-map",
    "recalld/postprocess-summary-reduce",
    "recalld/postprocess-style-analysis",
    "recalld/themes-single",
    "recalld/themes-map",
    "recalld/theme-guidance-instructions",
    "recalld/focus-instructions",
    "recalld/note-title",
)


def _resolve_cache_path(cache_path: Path | None) -> Path:
    return cache_path or DEFAULT_PROMPT_CACHE_PATH


def get_cached_prompt(prompt_name: str, cache_path: Path | None = None) -> str | None:
    path = _resolve_cache_path(cache_path)
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text())
        entry = data.get(prompt_name)
        if entry is None:
            return None
        return entry.get("text")
    except Exception:
        return None


def save_prompt_cache(data: dict[str, Any], cache_path: Path | None = None) -> None:
    path = _resolve_cache_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def sync_prompt_cache(
    client: Any,
    prompt_names: tuple[str, ...] = KNOWN_PROMPT_NAMES,
    cache_path: Path | None = None,
) -> int:
    path = _resolve_cache_path(cache_path)
    try:
        existing: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        existing = {}

    synced = 0
    for name in prompt_names:
        try:
            try:
                prompt = client.get_prompt(name, label="production", type="text")
            except TypeError:
                prompt = client.get_prompt(name, label="production")
        except Exception:
            continue
        raw = getattr(prompt, "prompt", None) or str(prompt)
        version = getattr(prompt, "version", None)
        updated = getattr(prompt, "updatedAt", None) or getattr(prompt, "updated_at", None)
        existing[name] = {
            "text": raw,
            "version": version,
            "updated_at": str(updated) if updated else None,
            "hash": hashlib.sha256(raw.encode()).hexdigest()[:12],
        }
        synced += 1

    if synced:
        save_prompt_cache(existing, cache_path=cache_path)
    return synced
