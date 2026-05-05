import json
import pytest
from pathlib import Path
from recalld.config import Config, Category, load_config, save_config


def test_load_defaults(tmp_path):
    cfg = load_config(tmp_path / "config.json")
    assert cfg.obsidian_api_url == "https://127.0.0.1:27124"
    assert cfg.llm_base_url == "http://localhost:1234/v1"
    assert cfg.llm_context_headroom == 0.8
    assert cfg.log_level == "info"
    assert cfg.last_used_category is None
    assert cfg.categories == []


def test_save_and_reload(tmp_path):
    path = tmp_path / "config.json"
    cfg = load_config(path)
    cfg.obsidian_api_key = "abc123"
    save_config(cfg, path)
    reloaded = load_config(path)
    assert reloaded.obsidian_api_key == "abc123"


def test_add_category(tmp_path):
    path = tmp_path / "config.json"
    cfg = load_config(path)
    cat = Category(
        id="coaching",
        name="Weekly Coaching",
        vault_path="Notes/Coaching/Sessions",
        focus_note_path="Notes/Coaching/Focus.md",
        speaker_a="You",
        speaker_b="Coach",
    )
    cfg.categories.append(cat)
    save_config(cfg, path)
    reloaded = load_config(path)
    assert len(reloaded.categories) == 1
    assert reloaded.categories[0].name == "Weekly Coaching"


def test_config_file_is_human_readable(tmp_path):
    path = tmp_path / "config.json"
    cfg = load_config(path)
    save_config(cfg, path)
    raw = path.read_text()
    parsed = json.loads(raw)
    assert "obsidian_api_url" in parsed


def test_corrupt_config_raises_clear_error(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("not valid json {{{")
    with pytest.raises(ValueError, match="is invalid"):
        load_config(path)


def test_default_config_path_is_resolved_at_call_time(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr("recalld.config.DEFAULT_CONFIG_PATH", path)

    cfg = load_config()
    cfg.llm_model = "qwen/qwen3-4b"
    save_config(cfg)

    assert path.exists()
    assert load_config().llm_model == "qwen/qwen3-4b"


def test_config_defaults_to_personal_vault_name():
    cfg = Config()
    assert cfg.vault_name == "Personal"
