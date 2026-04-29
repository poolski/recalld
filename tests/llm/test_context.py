import pytest
import httpx
import respx
from recalld.llm.context import ProviderModel, detect_context_length, estimate_tokens, list_available_models, token_budget

FAKE_MODELS_RESPONSE = {
    "data": [
        {"id": "qwen/qwen3-4b", "context_length": 32768},
        {"id": "qwen/qwen3-8b", "max_context_length": 65536},
    ]
}


@respx.mock
async def test_detect_context_length_success():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json=FAKE_MODELS_RESPONSE)
    )
    length = await detect_context_length("http://localhost:1234/v1", "qwen/qwen3-4b")
    assert length == 32768


@respx.mock
async def test_detect_context_length_fallback_on_error():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(500)
    )
    length = await detect_context_length("http://localhost:1234/v1", "qwen/qwen3-4b")
    assert length == 6000  # fallback default


@respx.mock
async def test_detect_context_length_fallback_on_unreachable():
    respx.get("http://localhost:1234/v1/models").mock(
        side_effect=httpx.ConnectError("unreachable")
    )
    length = await detect_context_length("http://localhost:1234/v1", "qwen/qwen3-4b")
    assert length == 6000


@respx.mock
async def test_detect_context_length_normalizes_base_url_without_v1_suffix():
    route = respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json=FAKE_MODELS_RESPONSE)
    )

    length = await detect_context_length("http://localhost:1234", "qwen/qwen3-8b")

    assert route.called
    assert length == 65536


@respx.mock
async def test_list_available_models_returns_provider_metadata():
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json=FAKE_MODELS_RESPONSE)
    )

    models = await list_available_models("http://localhost:1234", selected_model="qwen/qwen3-8b")

    assert models == [
        ProviderModel(id="qwen/qwen3-4b", context_length=32768, selected=False),
        ProviderModel(id="qwen/qwen3-8b", context_length=65536, selected=True),
    ]


def test_token_budget():
    assert token_budget(32768, headroom=0.8) == 26214


def test_estimate_tokens():
    # ~0.75 words per token heuristic, so 100 words ≈ 133 tokens
    text = " ".join(["word"] * 100)
    tokens = estimate_tokens(text)
    assert 120 <= tokens <= 150
