from __future__ import annotations

import math
from dataclasses import dataclass

import httpx

from recalld.llm.client import LLMClient

FALLBACK_CONTEXT_LENGTH = 6000


@dataclass(eq=True, frozen=True)
class ProviderModel:
    id: str
    max_context_length: int | None
    loaded_context_length: int | None = None
    loaded_instance_id: str | None = None
    selected: bool = False

    @property
    def context_length(self) -> int | None:
        if self.loaded_context_length is not None:
            return self.loaded_context_length
        return self.max_context_length

    @property
    def is_loaded(self) -> bool:
        return self.loaded_context_length is not None


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


def _loaded_instance_id(entry: dict) -> str | None:
    for instance in entry.get("loaded_instances", []):
        if not isinstance(instance, dict):
            continue
        instance_id = instance.get("id")
        if isinstance(instance_id, str) and instance_id:
            return instance_id
    return None


def _max_context_length(entry: dict) -> int | None:
    return _as_int(entry.get("max_context_length")) or _as_int(entry.get("context_length"))


def _selected_model(models: list[ProviderModel], selected_model: str) -> ProviderModel | None:
    for model in models:
        if model.id == selected_model:
            return model
    return None


def _loaded_models(models: list[ProviderModel]) -> list[ProviderModel]:
    return [model for model in models if model.loaded_context_length is not None]


def _normalize_model_entries(data: dict) -> list[ProviderModel]:
    if isinstance(data.get("data"), list):
        models: list[ProviderModel] = []
        for entry in data["data"]:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if not model_id:
                continue
            length = _max_context_length(entry)
            models.append(ProviderModel(id=str(model_id), max_context_length=length))
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
            loaded_length = _context_from_loaded_instances(entry)
            max_length = _max_context_length(entry)
            models.append(
                ProviderModel(
                    id=str(model_id),
                    max_context_length=max_length,
                    loaded_context_length=loaded_length,
                    loaded_instance_id=_loaded_instance_id(entry),
                )
            )
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
                max_context_length=model.max_context_length,
                loaded_context_length=model.loaded_context_length,
                loaded_instance_id=model.loaded_instance_id,
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


def _context_length_from_load_response(data: dict) -> int | None:
    if not isinstance(data, dict):
        return None
    load_config = data.get("load_config")
    if isinstance(load_config, dict):
        length = _as_int(load_config.get("context_length"))
        if length:
            return length
    return _as_int(data.get("context_length"))


async def ensure_loaded_context_length(
    base_url: str,
    model: str,
    requested_context_length: int | None = None,
) -> int:
    """Ensure the selected LM Studio model is loaded, then return its runtime context length."""
    models = await list_available_models(base_url, selected_model=model)
    selected = _selected_model(models, model)
    if selected is None:
        return await detect_context_length(base_url, model)

    loaded_models = _loaded_models(models)

    if selected.loaded_context_length and requested_context_length is None:
        return selected.loaded_context_length

    if selected.loaded_context_length and requested_context_length is not None:
        if selected.loaded_context_length == requested_context_length and len(loaded_models) == 1:
            return selected.loaded_context_length

    max_length = requested_context_length or selected.max_context_length or selected.context_length or FALLBACK_CONTEXT_LENGTH
    client = LLMClient(base_url=base_url, model=model)
    if requested_context_length is not None and loaded_models:
        for loaded in loaded_models:
            await client.unload_model(instance_id=loaded.loaded_instance_id or loaded.id)
    load_response = await client.load_model(context_length=max_length)
    loaded_length = _context_length_from_load_response(load_response)
    if requested_context_length is not None:
        if loaded_length is not None:
            if loaded_length != requested_context_length:
                raise RuntimeError(
                    f"LM Studio loaded {model} with context length {loaded_length}, expected {requested_context_length}"
                )
            return loaded_length

        refreshed = await list_available_models(base_url, selected_model=model)
        refreshed_selected = _selected_model(refreshed, model)
        if refreshed_selected and refreshed_selected.context_length == requested_context_length:
            return refreshed_selected.context_length

        raise RuntimeError(f"LM Studio did not report context length {requested_context_length} after loading {model}")

    refreshed = await list_available_models(base_url, selected_model=model)
    refreshed_selected = _selected_model(refreshed, model)
    if refreshed_selected and refreshed_selected.context_length:
        return refreshed_selected.context_length

    if loaded_length:
        return loaded_length

    return max_length


def token_budget(context_length: int, headroom: float = 0.8) -> int:
    return math.floor(context_length * headroom)


def estimate_tokens(text: str) -> int:
    """Heuristic: 1 token ≈ 0.75 words."""
    words = len(text.split())
    return math.ceil(words / 0.75)
