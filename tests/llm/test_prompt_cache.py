from __future__ import annotations

import json
from pathlib import Path

import pytest

from recalld.llm.prompt_cache import get_cached_prompt, save_prompt_cache, sync_prompt_cache, KNOWN_PROMPT_NAMES, BOOTSTRAP_PROMPT_CACHE_PATH


def test_get_cached_prompt_returns_none_when_no_cache_and_no_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.llm.prompt_cache.BOOTSTRAP_PROMPT_CACHE_PATH", tmp_path / "missing.json")

    result = get_cached_prompt("recalld/focus-instructions", cache_path=tmp_path / "prompts.json")

    assert result is None


def test_get_cached_prompt_returns_none_when_prompt_not_in_any_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "prompts.json"
    cache_path.write_text(json.dumps({"recalld/other-prompt": {"text": "other", "version": 1}}))
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text(json.dumps({"recalld/other-prompt": {"text": "other bootstrap", "version": 1}}))
    monkeypatch.setattr("recalld.llm.prompt_cache.BOOTSTRAP_PROMPT_CACHE_PATH", bootstrap)

    result = get_cached_prompt("recalld/focus-instructions", cache_path=cache_path)

    assert result is None


def test_get_cached_prompt_returns_text_from_cache(tmp_path):
    cache_path = tmp_path / "prompts.json"
    cache_path.write_text(
        json.dumps({"recalld/focus-instructions": {"text": "cached text here", "version": 2}})
    )

    result = get_cached_prompt("recalld/focus-instructions", cache_path=cache_path)

    assert result == "cached text here"


def test_save_prompt_cache_creates_file_and_is_readable(tmp_path):
    cache_path = tmp_path / "prompts.json"
    data = {"recalld/focus-instructions": {"text": "some prompt", "version": 3, "updated_at": "2026-05-07T00:00:00Z"}}

    save_prompt_cache(data, cache_path=cache_path)

    assert get_cached_prompt("recalld/focus-instructions", cache_path=cache_path) == "some prompt"


def test_save_prompt_cache_creates_parent_directory(tmp_path):
    cache_path = tmp_path / "nested" / "dir" / "prompts.json"
    data = {"recalld/note-title": {"text": "hello", "version": 1}}

    save_prompt_cache(data, cache_path=cache_path)

    assert cache_path.exists()


class _FakePrompt:
    def __init__(self, text: str, version: int = 1):
        self.prompt = text
        self.version = version
        self.updatedAt = "2026-05-07T00:00:00Z"


class _FakeClient:
    def __init__(self, prompts: dict[str, _FakePrompt]):
        self._prompts = prompts

    def get_prompt(self, name, **kwargs):
        if name not in self._prompts:
            raise Exception(f"not found: {name}")
        return self._prompts[name]


def test_sync_prompt_cache_writes_fetched_prompts(tmp_path):
    cache_path = tmp_path / "prompts.json"
    client = _FakeClient({"recalld/focus-instructions": _FakePrompt("fetched text", version=3)})

    synced = sync_prompt_cache(client, prompt_names=("recalld/focus-instructions",), cache_path=cache_path)

    assert synced == 1
    assert get_cached_prompt("recalld/focus-instructions", cache_path=cache_path) == "fetched text"


def test_sync_prompt_cache_skips_failed_prompts_and_continues(tmp_path):
    cache_path = tmp_path / "prompts.json"
    client = _FakeClient({"recalld/note-title": _FakePrompt("title text", version=1)})

    synced = sync_prompt_cache(
        client,
        prompt_names=("recalld/missing-prompt", "recalld/note-title"),
        cache_path=cache_path,
    )

    assert synced == 1
    assert get_cached_prompt("recalld/note-title", cache_path=cache_path) == "title text"
    assert get_cached_prompt("recalld/missing-prompt", cache_path=cache_path) is None


def test_sync_prompt_cache_stores_hash_in_cache_entry(tmp_path):
    cache_path = tmp_path / "prompts.json"
    client = _FakeClient({"recalld/note-title": _FakePrompt("some text", version=1)})

    sync_prompt_cache(client, prompt_names=("recalld/note-title",), cache_path=cache_path)

    import json, hashlib
    data = json.loads(cache_path.read_text())
    entry = data["recalld/note-title"]
    assert "hash" in entry
    assert entry["hash"] == hashlib.sha256("some text".encode()).hexdigest()[:12]


def test_sync_prompt_cache_merges_with_existing_cache(tmp_path):
    cache_path = tmp_path / "prompts.json"
    save_prompt_cache({"recalld/existing-prompt": {"text": "old", "version": 1}}, cache_path=cache_path)
    client = _FakeClient({"recalld/note-title": _FakePrompt("new title", version=2)})

    sync_prompt_cache(client, prompt_names=("recalld/note-title",), cache_path=cache_path)

    assert get_cached_prompt("recalld/existing-prompt", cache_path=cache_path) == "old"
    assert get_cached_prompt("recalld/note-title", cache_path=cache_path) == "new title"


def test_get_cached_prompt_falls_back_to_bootstrap_when_user_cache_missing(tmp_path, monkeypatch):
    bootstrap = tmp_path / "bootstrap" / "prompts.json"
    bootstrap.parent.mkdir()
    bootstrap.write_text(json.dumps({"recalld/note-title": {"text": "bootstrap text", "version": 1}}))
    monkeypatch.setattr("recalld.llm.prompt_cache.BOOTSTRAP_PROMPT_CACHE_PATH", bootstrap)

    result = get_cached_prompt("recalld/note-title", cache_path=tmp_path / "missing.json")

    assert result == "bootstrap text"


def test_get_cached_prompt_prefers_user_cache_over_bootstrap(tmp_path, monkeypatch):
    bootstrap = tmp_path / "bootstrap" / "prompts.json"
    bootstrap.parent.mkdir()
    bootstrap.write_text(json.dumps({"recalld/note-title": {"text": "bootstrap text", "version": 1}}))
    monkeypatch.setattr("recalld.llm.prompt_cache.BOOTSTRAP_PROMPT_CACHE_PATH", bootstrap)

    user_cache = tmp_path / "prompts.json"
    user_cache.write_text(json.dumps({"recalld/note-title": {"text": "user text", "version": 2}}))

    result = get_cached_prompt("recalld/note-title", cache_path=user_cache)

    assert result == "user text"


def test_bootstrap_prompt_cache_path_points_to_existing_file():
    assert BOOTSTRAP_PROMPT_CACHE_PATH.exists(), f"Bootstrap cache not found at {BOOTSTRAP_PROMPT_CACHE_PATH}"


def test_known_prompt_names_includes_all_pipeline_prompts():
    expected = {
        "recalld/postprocess-summary-single",
        "recalld/postprocess-summary-map",
        "recalld/postprocess-summary-reduce",
        "recalld/postprocess-style-analysis",
        "recalld/themes-single",
        "recalld/themes-map",
        "recalld/theme-guidance-instructions",
        "recalld/focus-instructions",
        "recalld/note-title",
    }
    assert expected.issubset(set(KNOWN_PROMPT_NAMES))
