from __future__ import annotations

from recalld.llm.prompts import resolve_text_prompt


def test_resolve_text_prompt_langfuse_includes_prompt_hash(monkeypatch):
    class _FakePrompt:
        version = 1
        labels = ["production"]

        def compile(self, **variables):
            return "exact text"

    class _FakeClient:
        def get_prompt(self, name, **kwargs):
            return _FakePrompt()

    monkeypatch.setattr("recalld.llm.prompts.get_langfuse_client", lambda: _FakeClient())
    monkeypatch.setattr("recalld.llm.prompts.get_cached_prompt", lambda name: None)

    prompt = resolve_text_prompt("recalld/example-prompt", "fallback")

    assert "prompt_hash" in prompt.metadata
    import hashlib
    expected = hashlib.sha256("exact text".encode()).hexdigest()[:12]
    assert prompt.metadata["prompt_hash"] == expected


def test_resolve_text_prompt_cache_includes_prompt_hash(monkeypatch):
    monkeypatch.setattr("recalld.llm.prompts.get_langfuse_client", lambda: None)
    monkeypatch.setattr(
        "recalld.llm.prompts.get_cached_prompt",
        lambda name: "cached text",
    )

    prompt = resolve_text_prompt("recalld/example-prompt", "fallback")

    assert "prompt_hash" in prompt.metadata
    import hashlib
    expected = hashlib.sha256("cached text".encode()).hexdigest()[:12]
    assert prompt.metadata["prompt_hash"] == expected


def test_resolve_text_prompt_uses_cache_when_langfuse_unavailable(monkeypatch):
    monkeypatch.setattr("recalld.llm.prompts.get_langfuse_client", lambda: None)
    monkeypatch.setattr(
        "recalld.llm.prompts.get_cached_prompt",
        lambda name: "Cached prompt for {name}",
    )

    prompt = resolve_text_prompt(
        "recalld/example-prompt",
        "Fallback for {name}",
        name="Alex",
    )

    assert prompt.text == "Cached prompt for Alex"
    assert prompt.source == "cache"
    assert prompt.prompt is None
    assert prompt.metadata["prompt_source"] == "cache"
    assert prompt.metadata["prompt_name"] == "recalld/example-prompt"


def test_resolve_text_prompt_falls_back_to_hardcoded_when_cache_misses(monkeypatch):
    monkeypatch.setattr("recalld.llm.prompts.get_langfuse_client", lambda: None)
    monkeypatch.setattr("recalld.llm.prompts.get_cached_prompt", lambda name: None)

    prompt = resolve_text_prompt(
        "recalld/example-prompt",
        "Fallback for {name}",
        name="Alex",
    )

    assert prompt.text == "Fallback for Alex"
    assert prompt.source == "fallback"


def test_resolve_text_prompt_prefers_langfuse_over_cache(monkeypatch):
    class _FakePrompt:
        version = 3
        labels = ["production"]

        def compile(self, **variables):
            return f"Langfuse for {variables['name']}"

    class _FakeClient:
        def get_prompt(self, name, **kwargs):
            return _FakePrompt()

    monkeypatch.setattr("recalld.llm.prompts.get_langfuse_client", lambda: _FakeClient())
    cache_calls: list[str] = []
    monkeypatch.setattr(
        "recalld.llm.prompts.get_cached_prompt",
        lambda name: cache_calls.append(name) or "Cached",
    )

    prompt = resolve_text_prompt("recalld/example-prompt", "Fallback", name="Alex")

    assert prompt.source == "langfuse"
    assert prompt.text == "Langfuse for Alex"
    assert cache_calls == []


def test_resolve_text_prompt_uses_fallback_and_preserves_non_variable_braces(monkeypatch):
    monkeypatch.setattr("recalld.llm.prompts.get_langfuse_client", lambda: None)

    prompt = resolve_text_prompt(
        "recalld/example-prompt",
        'Hello {name}! JSON example: {"items": []}',
        name="Alex",
    )

    assert prompt.text == 'Hello Alex! JSON example: {"items": []}'
    assert prompt.prompt is None
    assert prompt.source == "fallback"
    assert prompt.metadata["prompt_name"] == "recalld/example-prompt"
    assert prompt.metadata["prompt_source"] == "fallback"


def test_resolve_text_prompt_uses_requested_prompt_label(monkeypatch):
    class _FakePrompt:
        version = 7
        labels = ["candidate"]

        def compile(self, **variables):
            return f"hello {variables['name']}"

    class _FakeClient:
        def __init__(self):
            self.calls = []

        def get_prompt(self, name, **kwargs):
            self.calls.append((name, kwargs))
            return _FakePrompt()

    client = _FakeClient()
    monkeypatch.setattr("recalld.llm.prompts.get_langfuse_client", lambda: client)

    prompt = resolve_text_prompt(
        "recalld/example-prompt",
        "Hello {name}",
        prompt_label="candidate",
        name="Alex",
    )

    assert client.calls[0][0] == "recalld/example-prompt"
    assert client.calls[0][1]["label"] == "candidate"
    assert prompt.text == "hello Alex"
    assert prompt.metadata["prompt_version"] == 7
    assert prompt.metadata["prompt_labels"] == ["candidate"]
