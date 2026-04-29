from __future__ import annotations

import math
from dataclasses import dataclass

import httpx

FALLBACK_CONTEXT_LENGTH = 6000


@dataclass(eq=True, frozen=True)
class ProviderModel:
    id: str
    context_length: int | None
    selected: bool = False


def _models_urls(base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    if base.endswith("/api/v1"):
        root = base[:-7]
        return [f"{root}/api/v1/models", f"{root}/v1/models"]
    if base.endswith("/v1"):
        root = base[:-3]
        return [f"{root}/api/v1/models", f"{base}/models"]
    return [f"{base}/api/v1/models", f"{base}/v1/models"]


def _as_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _context_from_loaded_instances(entry: dict) -> int | None:
    for instance in entry.get("loaded_instances", []):
        if not isinstance(instance, dict):
            continue
        config = instance.get("config")
        if not isinstance(config, dict):
            continue
        length = _as_int(config.get("context_length"))
        if length:
            return length
    return None


def _normalize_model_entries(data: dict) -> list[ProviderModel]:
    if isinstance(data.get("data"), list):
        models: list[ProviderModel] = []
        for entry in data["data"]:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if not model_id:
                continue
            length = _as_int(entry.get("context_length")) or _as_int(entry.get("max_context_length"))
            models.append(ProviderModel(id=str(model_id), context_length=length))
        return models

    if isinstance(data.get("models"), list):
        models = []
        for entry in data["models"]:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "llm":
                continue
            model_id = entry.get("key") or entry.get("id")
            if not model_id:
                continue
            length = _context_from_loaded_instances(entry) or _as_int(entry.get("context_length")) or _as_int(entry.get("max_context_length"))
            models.append(ProviderModel(id=str(model_id), context_length=length))
        return models

    return []


async def list_available_models(base_url: str, selected_model: str) -> list[ProviderModel]:
    """Query the provider model list and normalize it for settings and chunking decisions."""
    data = None
    async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
        for url in _models_urls(base_url):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                continue
    if data is None:
        return []

    return [
        ProviderModel(
            id=model.id,
            context_length=model.context_length,
            selected=model.id == selected_model,
        )
        for model in _normalize_model_entries(data)
    ]


async def detect_context_length(base_url: str, model: str) -> int:
    """Query provider models for context length. Falls back to 6000 on any error."""
    models = await list_available_models(base_url, selected_model=model)
    for entry in models:
        if entry.selected and entry.context_length:
            return entry.context_length
    for entry in models:
        if entry.context_length:
            return entry.context_length
    return FALLBACK_CONTEXT_LENGTH


def token_budget(context_length: int, headroom: float = 0.8) -> int:
    return math.floor(context_length * headroom)


def estimate_tokens(text: str) -> int:
    """Heuristic: 1 token ≈ 0.75 words."""
    words = len(text.split())
    return math.ceil(words / 0.75)
