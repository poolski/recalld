import json

import httpx
import pytest
import respx

from recalld.llm.client import LLMClient


@pytest.mark.asyncio
@respx.mock
async def test_llm_client_uses_lmstudio_chat_endpoint_and_payload():
    route = respx.post("http://localhost:1234/api/v1/chat").mock(
        return_value=httpx.Response(200, json={"output": "## Summary\n\nHello\n\n## Focus\n\n- [ ] Test"})
    )

    client = LLMClient(base_url="http://localhost:1234", model="qwen/qwen3-4b")
    result = await client.complete("You answer only in rhymes.", "What is your favorite color?")

    assert route.called
    request = route.calls[0].request
    assert request.content
    assert request.headers["content-type"].startswith("application/json")
    assert json.loads(request.content.decode()) == {
        "model": "qwen/qwen3-4b",
        "system_prompt": "You answer only in rhymes.",
        "input": "What is your favorite color?",
    }
    assert "## Summary" in result


@pytest.mark.asyncio
@respx.mock
async def test_llm_client_strips_trailing_v1_from_base_url():
    route = respx.post("http://localhost:1234/api/v1/chat").mock(
        return_value=httpx.Response(200, json={"output": "ok"})
    )

    client = LLMClient(base_url="http://localhost:1234/v1", model="qwen/qwen3-4b")
    result = await client.complete("system", "user")

    assert route.called
    assert result == "ok"


@pytest.mark.asyncio
@respx.mock
async def test_llm_client_extracts_message_content_from_structured_output():
    route = respx.post("http://localhost:1234/api/v1/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model_instance_id": "qwen/qwen3-4b",
                "output": [
                    {"type": "reasoning", "content": "internal reasoning"},
                    {"type": "message", "content": "## Summary\n\nHello\n\n## Focus\n\n- [ ] Test"},
                ],
            },
        )
    )

    client = LLMClient(base_url="http://localhost:1234", model="qwen/qwen3-4b")
    result = await client.complete("system", "user")

    assert route.called
    assert result == "## Summary\n\nHello\n\n## Focus\n\n- [ ] Test"
