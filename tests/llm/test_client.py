import json

import httpx
import pytest
import respx

from recalld.llm.client import LLMClient
from recalld.llm.client import LLMRequestError


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
        "stream": False,
    }
    assert "## Summary" in result


@pytest.mark.asyncio
@respx.mock
async def test_llm_client_loads_model_with_context_length_and_auth_header(monkeypatch):
    route = respx.post("http://localhost:1234/api/v1/models/load").mock(
        return_value=httpx.Response(
            200,
            json={
                "type": "llm",
                "instance_id": "qwen/qwen3-4b",
                "status": "loaded",
                "load_config": {"context_length": 16384},
            },
        )
    )
    monkeypatch.setenv("LM_API_TOKEN", "secret-token")

    client = LLMClient(base_url="http://localhost:1234", model="qwen/qwen3-4b")
    result = await client.load_model(context_length=16384)

    assert route.called
    request = route.calls[0].request
    assert json.loads(request.content.decode()) == {
        "model": "qwen/qwen3-4b",
        "context_length": 16384,
        "echo_load_config": True,
    }
    assert request.headers["authorization"] == "Bearer secret-token"
    assert result["status"] == "loaded"


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
async def test_llm_client_raises_lmstudio_error_message_on_failed_request():
    respx.post("http://localhost:1234/api/v1/chat").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "Reasoning setting 'low' is not supported by model 'google/gemma-4-e4b'. Supported settings: 'off', 'on'.",
                    "type": "invalid_request",
                    "param": "reasoning",
                    "code": "invalid_value",
                }
            },
        )
    )

    client = LLMClient(base_url="http://localhost:1234", model="qwen/qwen3-4b")

    with pytest.raises(LLMRequestError, match="Reasoning setting 'low' is not supported"):
        await client.complete("system", "user")


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


@pytest.mark.asyncio
@respx.mock
async def test_llm_client_stream_yields_partial_tokens():
    sse_content = (
        'event: chat.start\n'
        'data: {"type":"chat.start","model_instance_id":"m"}\n\n'
        'event: message.delta\n'
        'data: {"type":"message.delta","content":"Part 1 "}\n\n'
        'event: message.delta\n'
        'data: {"type":"message.delta","content":"Part 2"}\n\n'
        'event: chat.end\n'
        'data: {"type":"chat.end","result":{"model_instance_id":"m","output":[{"type":"message","content":"Part 1 Part 2"}]}}\n\n'
        'data: [DONE]\n\n'
    ).encode()
    route = respx.post("http://localhost:1234/api/v1/chat").mock(
        return_value=httpx.Response(200, content=sse_content)
    )

    client = LLMClient(base_url="http://localhost:1234", model="m")
    tokens = []
    async for t in client.stream("sys", "user"):
        tokens.append(t)

    assert route.called
    request = route.calls[0].request
    assert request.content
    assert json.loads(request.content.decode()) == {
        "model": "m",
        "system_prompt": "sys",
        "input": "user",
        "stream": True,
    }
    assert tokens == ["Part 1 ", "Part 2"]


@pytest.mark.asyncio
@respx.mock
async def test_llm_client_stream_falls_back_to_chat_end_when_no_deltas():
    sse_content = (
        'event: chat.start\n'
        'data: {"type":"chat.start","model_instance_id":"m"}\n\n'
        'event: chat.end\n'
        'data: {"type":"chat.end","result":{"model_instance_id":"m","output":[{"type":"message","content":"Done"}]}}\n\n'
        'data: [DONE]\n\n'
    ).encode()
    respx.post("http://localhost:1234/api/v1/chat").mock(
        return_value=httpx.Response(200, content=sse_content)
    )

    client = LLMClient(base_url="http://localhost:1234", model="m")
    tokens = []
    async for t in client.stream("sys", "user"):
        tokens.append(t)

    assert tokens == ["Done"]


@pytest.mark.asyncio
@respx.mock
async def test_llm_client_stream_reports_lmstudio_streaming_events():
    sse_content = (
        'event: chat.start\n'
        'data: {"type":"chat.start","model_instance_id":"m"}\n\n'
        'event: prompt_processing.start\n'
        'data: {"type":"prompt_processing.start"}\n\n'
        'event: prompt_processing.progress\n'
        'data: {"type":"prompt_processing.progress","progress":0.5}\n\n'
        'event: prompt_processing.end\n'
        'data: {"type":"prompt_processing.end"}\n\n'
        'event: reasoning.start\n'
        'data: {"type":"reasoning.start"}\n\n'
        'event: reasoning.delta\n'
        'data: {"type":"reasoning.delta","content":"Need to"}\n\n'
        'event: reasoning.end\n'
        'data: {"type":"reasoning.end"}\n\n'
        'event: message.start\n'
        'data: {"type":"message.start"}\n\n'
        'event: message.delta\n'
        'data: {"type":"message.delta","content":"Done"}\n\n'
        'event: message.end\n'
        'data: {"type":"message.end"}\n\n'
        'event: chat.end\n'
        'data: {"type":"chat.end","result":{"model_instance_id":"m","output":[{"type":"message","content":"Done"}]}}\n\n'
        'data: [DONE]\n\n'
    ).encode()
    respx.post("http://localhost:1234/api/v1/chat").mock(
        return_value=httpx.Response(200, content=sse_content)
    )

    client = LLMClient(base_url="http://localhost:1234", model="m")
    tokens = []
    events = []

    async for t in client.stream("sys", "user", event_cb=lambda event_type, data: events.append(event_type)):
        tokens.append(t)

    assert tokens == ["Done"]
    assert events == [
        "chat.start",
        "prompt_processing.start",
        "prompt_processing.progress",
        "prompt_processing.end",
        "reasoning.start",
        "reasoning.delta",
        "reasoning.end",
        "message.start",
        "message.delta",
        "message.end",
        "chat.end",
    ]
