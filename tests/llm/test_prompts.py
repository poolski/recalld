from __future__ import annotations

from recalld.llm.prompts import resolve_text_prompt


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
