import pytest
import httpx
import respx
from recalld.llm.context import detect_context_length, token_budget, estimate_tokens

FAKE_MODELS_RESPONSE = {
    "data": [{"id": "qwen2.5-7b", "context_length": 32768}]
}


@respx.mock
async def test_detect_context_length_success():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json=FAKE_MODELS_RESPONSE)
    )
    length = await detect_context_length("http://localhost:1234/v1", "qwen2.5-7b")
    assert length == 32768


@respx.mock
async def test_detect_context_length_fallback_on_error():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(500)
    )
    length = await detect_context_length("http://localhost:1234/v1", "qwen2.5-7b")
    assert length == 6000  # fallback default


@respx.mock
async def test_detect_context_length_fallback_on_unreachable():
    respx.get("http://localhost:1234/v1/models").mock(
        side_effect=httpx.ConnectError("unreachable")
    )
    length = await detect_context_length("http://localhost:1234/v1", "qwen2.5-7b")
    assert length == 6000


def test_token_budget():
    assert token_budget(32768, headroom=0.8) == 26214


def test_estimate_tokens():
    # ~0.75 words per token heuristic, so 100 words ≈ 133 tokens
    text = " ".join(["word"] * 100)
    tokens = estimate_tokens(text)
    assert 120 <= tokens <= 150
