from __future__ import annotations

import math

import httpx

FALLBACK_CONTEXT_LENGTH = 6000


async def detect_context_length(base_url: str, model: str) -> int:
    """Query /v1/models to get the context length for the loaded model. Falls back to 6000 on any error."""
    try:
        async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
            resp = await client.get(f"{base_url}/models")
            resp.raise_for_status()
            data = resp.json()
            for entry in data.get("data", []):
                if entry.get("id") == model:
                    length = entry.get("context_length") or entry.get("max_context_length")
                    if length:
                        return int(length)
            # If model not found by name, use first entry
            entries = data.get("data", [])
            if entries:
                length = entries[0].get("context_length") or entries[0].get("max_context_length")
                if length:
                    return int(length)
    except Exception:
        pass
    return FALLBACK_CONTEXT_LENGTH


def token_budget(context_length: int, headroom: float = 0.8) -> int:
    return math.floor(context_length * headroom)


def estimate_tokens(text: str) -> int:
    """Heuristic: 1 token ≈ 0.75 words."""
    words = len(text.split())
    return math.ceil(words / 0.75)
