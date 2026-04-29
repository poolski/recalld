import pytest
import httpx
import respx
from recalld.llm.context import ProviderModel, detect_context_length, ensure_loaded_context_length, estimate_tokens, list_available_models, token_budget

FAKE_MODELS_RESPONSE = {
    "data": [
        {"id": "qwen/qwen3-4b", "context_length": 32768},
        {"id": "qwen/qwen3-8b", "max_context_length": 65536},
    ]
}

FAKE_LMSTUDIO_MODELS_RESPONSE = {
    "models": [
        {
            "type": "llm",
            "key": "gemma-3-270m-it-qat",
            "display_name": "Gemma 3 270m Instruct Qat",
            "loaded_instances": [
                {
                    "id": "gemma-3-270m-it-qat",
                    "config": {"context_length": 4096},
                }
            ],
            "max_context_length": 32768,
        },
        {
            "type": "llm",
            "key": "qwen/qwen3-4b",
            "display_name": "Qwen 3 4B",
            "loaded_instances": [],
            "max_context_length": 65536,
        },
        {
            "type": "embedding",
            "key": "text-embedding-nomic-embed-text-v1.5-embedding",
            "display_name": "Nomic Embed Text v1.5",
            "loaded_instances": [],
            "max_context_length": 2048,
        },
    ]
}

FAKE_LMSTUDIO_LOAD_RESPONSE = {
    "type": "llm",
    "instance_id": "qwen/qwen3-4b",
    "status": "loaded",
    "load_config": {"context_length": 65536},
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
        ProviderModel(id="qwen/qwen3-4b", max_context_length=32768, selected=False),
        ProviderModel(id="qwen/qwen3-8b", max_context_length=65536, selected=True),
    ]


@respx.mock
async def test_list_available_models_supports_lmstudio_models_payload():
    respx.get("http://localhost:1234/api/v1/models").mock(
        return_value=httpx.Response(200, json=FAKE_LMSTUDIO_MODELS_RESPONSE)
    )

    models = await list_available_models("http://localhost:1234/api/v1", selected_model="gemma-3-270m-it-qat")

    assert models == [
        ProviderModel(id="gemma-3-270m-it-qat", max_context_length=32768, loaded_context_length=4096, selected=True),
        ProviderModel(id="qwen/qwen3-4b", max_context_length=65536, selected=False),
    ]


@respx.mock
async def test_detect_context_length_prefers_loaded_instance_context_for_lmstudio():
    respx.get("http://localhost:1234/api/v1/models").mock(
        return_value=httpx.Response(200, json=FAKE_LMSTUDIO_MODELS_RESPONSE)
    )

    length = await detect_context_length("http://localhost:1234/api/v1", "gemma-3-270m-it-qat")

    assert length == 4096


@respx.mock
async def test_detect_context_length_uses_max_context_length_when_lmstudio_model_is_not_loaded():
    respx.get("http://localhost:1234/api/v1/models").mock(
        return_value=httpx.Response(200, json=FAKE_LMSTUDIO_MODELS_RESPONSE)
    )

    length = await detect_context_length("http://localhost:1234/api/v1", "qwen/qwen3-4b")

    assert length == 65536


@respx.mock
async def test_ensure_loaded_context_length_loads_unloaded_model_with_max_context_length():
    list_route = respx.get("http://localhost:1234/api/v1/models").mock(
        side_effect=[
            httpx.Response(200, json=FAKE_LMSTUDIO_MODELS_RESPONSE),
            httpx.Response(200, json={
                "models": [
                    {
                        "type": "llm",
                        "key": "qwen/qwen3-4b",
                        "display_name": "Qwen 3 4B",
                        "loaded_instances": [
                            {
                                "id": "qwen/qwen3-4b",
                                "config": {"context_length": 65536},
                            }
                        ],
                        "max_context_length": 65536,
                    },
                ]
            }),
        ],
    )
    load_route = respx.post("http://localhost:1234/api/v1/models/load").mock(
        return_value=httpx.Response(200, json=FAKE_LMSTUDIO_LOAD_RESPONSE)
    )

    length = await ensure_loaded_context_length("http://localhost:1234/api/v1", "qwen/qwen3-4b")

    assert list_route.call_count == 2
    assert load_route.called
    assert load_route.calls[0].request.content
    assert length == 65536


@respx.mock
async def test_ensure_loaded_context_length_uses_existing_loaded_instance():
    respx.get("http://localhost:1234/api/v1/models").mock(
        return_value=httpx.Response(200, json=FAKE_LMSTUDIO_MODELS_RESPONSE)
    )

    length = await ensure_loaded_context_length("http://localhost:1234/api/v1", "gemma-3-270m-it-qat")

    assert length == 4096


def test_token_budget():
    assert token_budget(32768, headroom=0.8) == 26214


def test_estimate_tokens():
    # ~0.75 words per token heuristic, so 100 words ≈ 133 tokens
    text = " ".join(["word"] * 100)
    tokens = estimate_tokens(text)
    assert 120 <= tokens <= 150
