import subprocess
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from recalld.app import create_app
from recalld.config import Config, load_config, save_config
from recalld.llm.context import ProviderModel


class _FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str):
        return _FakeResponse(200)


def test_settings_page_shows_provider_models(monkeypatch):
    async def fake_list_models(base_url: str, selected_model: str):
        return [
            ProviderModel(
                id="qwen/qwen3-4b",
                max_context_length=32768,
                loaded_context_length=16384,
                loaded_instance_id="qwen/qwen3-4b",
                selected=True,
            ),
            ProviderModel(id="qwen/qwen3-8b", max_context_length=65536, selected=False),
        ]

    monkeypatch.setattr("recalld.routers.settings.list_available_models", fake_list_models)

    client = TestClient(create_app())
    resp = client.get("/settings/")

    assert resp.status_code == 200
    assert "Available models" in resp.text
    assert "qwen/qwen3-4b" in resp.text
    assert "qwen/qwen3-8b" in resp.text
    assert "32,768" in resp.text
    assert "65,536" in resp.text
    assert "16,384" in resp.text
    assert "Current model" in resp.text
    assert "Loaded" in resp.text
    assert "Loaded context length" in resp.text
    assert "Maximum context length" in resp.text
    assert "Refresh models" in resp.text
    assert 'name="llm_reasoning"' in resp.text
    assert '<option value="off" selected>Off</option>' in resp.text


def test_settings_page_indicates_stored_secrets():
    cfg = Config(obsidian_api_key="obsidian-secret", huggingface_token="hf-secret")
    save_config(cfg)

    client = TestClient(create_app())
    resp = client.get("/settings/")

    assert resp.status_code == 200
    assert "Stored secret detected." in resp.text
    assert 'placeholder="Leave blank to keep existing"' in resp.text


def test_settings_page_shows_vault_name_field():
    client = TestClient(create_app())
    resp = client.get("/settings/")

    assert resp.status_code == 200
    assert 'name="vault_name"' in resp.text
    assert 'value="Personal"' in resp.text


def test_save_settings_persists_selected_provider_model(monkeypatch):
    async def fake_list_models(base_url: str, selected_model: str):
        return [
            ProviderModel(id="qwen/qwen3-4b", max_context_length=32768, selected=selected_model == "qwen/qwen3-4b"),
            ProviderModel(id="qwen/qwen3-8b", max_context_length=65536, selected=selected_model == "qwen/qwen3-8b"),
        ]

    monkeypatch.setattr("recalld.routers.settings.list_available_models", fake_list_models)

    client = TestClient(create_app())
    resp = client.post("/settings/", data={
        "vault_name": "Personal",
        "obsidian_api_url": "https://127.0.0.1:27124",
        "obsidian_api_key": "",
        "llm_base_url": "http://localhost:1234",
        "llm_model": "qwen/qwen3-8b",
        "llm_context_headroom": "0.75",
        "llm_reasoning": "off",
        "log_level": "info",
        "whisper_model": "small",
        "huggingface_token": "",
        "scratch_retention_days": "30",
    })

    assert resp.status_code == 200
    cfg = load_config()
    assert cfg.llm_base_url == "http://localhost:1234"
    assert cfg.llm_model == "qwen/qwen3-8b"
    assert cfg.llm_context_headroom == 0.75
    assert cfg.llm_reasoning == "off"
    assert "Settings saved." in resp.text
    assert "qwen/qwen3-8b" in resp.text
    assert load_config().vault_name == "Personal"


def test_save_settings_preserves_existing_secrets_when_blank(monkeypatch):
    cfg = Config(
        obsidian_api_key="obsidian-secret",
        huggingface_token="hf-secret",
        llm_base_url="http://localhost:1234",
        llm_model="qwen/qwen3-4b",
    )
    save_config(cfg)

    async def fake_list_models(base_url: str, selected_model: str):
        return []

    monkeypatch.setattr("recalld.routers.settings.list_available_models", fake_list_models)

    client = TestClient(create_app())
    resp = client.post("/settings/", data={
        "vault_name": "Personal",
        "obsidian_api_url": "https://127.0.0.1:27124",
        "obsidian_api_key": "",
        "llm_base_url": "http://localhost:5678",
        "llm_model": "qwen/qwen3-8b",
        "llm_context_headroom": "0.75",
        "llm_reasoning": "off",
        "log_level": "debug",
        "whisper_model": "medium",
        "huggingface_token": "",
        "scratch_retention_days": "14",
    })

    assert resp.status_code == 200
    reloaded = load_config()
    assert reloaded.obsidian_api_key == "obsidian-secret"
    assert reloaded.huggingface_token == "hf-secret"
    assert reloaded.llm_base_url == "http://localhost:5678"
    assert reloaded.whisper_model == "medium"
    assert "Stored secret detected." in resp.text


def test_settings_page_shows_notice_when_provider_unavailable_but_model_selected(monkeypatch):
    async def fake_list_models(base_url: str, selected_model: str):
        return []

    monkeypatch.setattr("recalld.routers.settings.list_available_models", fake_list_models)

    client = TestClient(create_app())
    client.post("/settings/", data={
        "vault_name": "Personal",
        "obsidian_api_url": "https://127.0.0.1:27124",
        "obsidian_api_key": "",
        "llm_base_url": "http://localhost:1234/api/v1",
        "llm_model": "qwen/qwen3-4b",
        "llm_context_headroom": "0.8",
        "llm_reasoning": "off",
        "log_level": "info",
        "whisper_model": "small",
        "huggingface_token": "",
        "scratch_retention_days": "30",
    })

    resp = client.get("/settings/")

    assert resp.status_code == 200
    assert "Could not refresh models from LM Studio" in resp.text


def test_status_bar_renders_clickable_indicators(monkeypatch):
    monkeypatch.setattr("recalld.routers.settings.httpx.AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr("recalld.routers.settings.detect_context_length", AsyncMock(return_value=8192))
    monkeypatch.setattr("recalld.routers.settings.shutil.which", lambda name: "/opt/homebrew/bin/ffmpeg" if name == "ffmpeg" else None)

    client = TestClient(create_app())
    resp = client.get("/settings/status")

    assert resp.status_code == 200
    assert 'type="button"' in resp.text
    assert 'data-status-kind="llm"' in resp.text
    assert 'data-status-kind="obsidian"' in resp.text
    assert 'data-status-kind="ffmpeg"' in resp.text
    assert 'data-status-kind="pyannote"' in resp.text
    assert 'data-status-title="Obsidian API"' in resp.text


def test_status_details_returns_selected_model_and_tooling(monkeypatch):
    async def fake_list_models(base_url: str, selected_model: str):
        return [
            ProviderModel(
                id="qwen/qwen3-4b",
                max_context_length=32768,
                loaded_context_length=16384,
                loaded_instance_id="qwen/qwen3-4b",
                selected=True,
            ),
        ]

    def fake_run(cmd, capture_output, text, check, timeout):
        return subprocess.CompletedProcess(cmd, 0, stdout="ffmpeg version 7.1\n", stderr="")

    monkeypatch.setattr("recalld.routers.settings.list_available_models", fake_list_models)
    monkeypatch.setattr("recalld.routers.settings.detect_context_length", AsyncMock(return_value=32768))
    monkeypatch.setattr("recalld.routers.settings.httpx.AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr("recalld.routers.settings.shutil.which", lambda name: "/opt/homebrew/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr("recalld.routers.settings.subprocess.run", fake_run)
    monkeypatch.setattr("recalld.routers.settings.load_config", lambda path=None: Config(llm_model="qwen/qwen3-4b", llm_base_url="http://localhost:1234"))

    client = TestClient(create_app())
    resp = client.get("/settings/status/details?kind=llm")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["title"] == "LLM"
    assert payload["items"][0]["label"] == "Selected model"
    assert payload["items"][0]["value"] == "qwen/qwen3-4b"
    assert any(item["label"] == "Loaded" and item["value"] == "Yes" for item in payload["items"])
    assert any(item["label"] == "Loaded context length" and item["value"] == "16,384" for item in payload["items"])
    assert any(item["label"] == "Maximum context length" and item["value"] == "32,768" for item in payload["items"])


def test_status_details_returns_vault_name_for_obsidian(monkeypatch):
    async def fake_list_models(base_url: str, selected_model: str):
        return []

    monkeypatch.setattr("recalld.routers.settings.httpx.AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr("recalld.routers.settings.list_available_models", fake_list_models)
    monkeypatch.setattr("recalld.routers.settings.detect_context_length", AsyncMock(return_value=8192))
    monkeypatch.setattr("recalld.routers.settings.shutil.which", lambda name: "/opt/homebrew/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr("recalld.routers.settings.load_config", lambda path=None: Config(vault_name="Personal", obsidian_api_url="https://127.0.0.1:27124"))

    client = TestClient(create_app())
    resp = client.get("/settings/status/details?kind=obsidian")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["title"] == "Obsidian API"
    assert any(item["label"] == "Vault name" for item in payload["items"])
    assert payload["items"][0]["value"] == "Personal"
