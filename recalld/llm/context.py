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


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/api/v1"):
        return f"{base[:-7]}/v1/models"
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


async def list_available_models(base_url: str, selected_model: str) -> list[ProviderModel]:
    """Query the provider model list and normalize it for settings and chunking decisions."""
    try:
        async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
            resp = await client.get(_models_url(base_url))
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return []

    models: list[ProviderModel] = []
    for entry in data.get("data", []):
        model_id = entry.get("id")
        if not model_id:
            continue
        length = entry.get("context_length") or entry.get("max_context_length")
        models.append(ProviderModel(
            id=str(model_id),
            context_length=int(length) if length else None,
            selected=str(model_id) == selected_model,
        ))
    return models


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
