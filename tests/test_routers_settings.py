from fastapi.testclient import TestClient

from recalld.app import create_app
from recalld.config import load_config
from recalld.llm.context import ProviderModel


def test_settings_page_shows_provider_models(monkeypatch):
    async def fake_list_models(base_url: str, selected_model: str):
        return [
            ProviderModel(id="qwen/qwen3-4b", context_length=32768, selected=True),
            ProviderModel(id="qwen/qwen3-8b", context_length=65536, selected=False),
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
    assert "Current model" in resp.text


def test_save_settings_persists_selected_provider_model(monkeypatch):
    async def fake_list_models(base_url: str, selected_model: str):
        return [
            ProviderModel(id="qwen/qwen3-4b", context_length=32768, selected=selected_model == "qwen/qwen3-4b"),
            ProviderModel(id="qwen/qwen3-8b", context_length=65536, selected=selected_model == "qwen/qwen3-8b"),
        ]

    monkeypatch.setattr("recalld.routers.settings.list_available_models", fake_list_models)

    client = TestClient(create_app())
    resp = client.post("/settings/", data={
        "obsidian_api_url": "https://127.0.0.1:27124",
        "obsidian_api_key": "",
        "llm_base_url": "http://localhost:1234",
        "llm_model": "qwen/qwen3-8b",
        "llm_context_headroom": "0.75",
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
    assert "Settings saved." in resp.text
    assert "qwen/qwen3-8b" in resp.text

