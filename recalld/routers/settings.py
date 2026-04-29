from __future__ import annotations

import shutil

import httpx
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from recalld.app import templates
from recalld.config import DEFAULT_CONFIG_PATH, load_config, save_config
from recalld.llm.context import detect_context_length, list_available_models

router = APIRouter(prefix="/settings")


async def _settings_context(request: Request, cfg, *, saved: bool = False) -> dict:
    provider_models = await list_available_models(cfg.llm_base_url, cfg.llm_model)
    current_model = next((model for model in provider_models if model.selected), None)
    detected_context_length = current_model.context_length if current_model and current_model.context_length else await detect_context_length(cfg.llm_base_url, cfg.llm_model)
    return {
        "request": request,
        "cfg": cfg,
        "provider_models": provider_models,
        "current_model": current_model,
        "detected_context_length": detected_context_length,
        "saved": saved,
    }


@router.get("/", response_class=HTMLResponse)
async def settings_page(request: Request):
    cfg = load_config()
    return templates.TemplateResponse(request, "settings.html", await _settings_context(request, cfg))


@router.post("/", response_class=HTMLResponse)
async def save_settings(
    request: Request,
    obsidian_api_url: str = Form(...),
    obsidian_api_key: str = Form(""),
    llm_base_url: str = Form(...),
    llm_model: str = Form(""),
    llm_context_headroom: float = Form(0.8),
    log_level: str = Form("info"),
    whisper_model: str = Form("small"),
    huggingface_token: str = Form(""),
    scratch_retention_days: int = Form(30),
):
    cfg = load_config()
    cfg.obsidian_api_url = obsidian_api_url
    cfg.obsidian_api_key = obsidian_api_key
    cfg.llm_base_url = llm_base_url
    cfg.llm_model = llm_model
    cfg.llm_context_headroom = llm_context_headroom
    cfg.log_level = log_level
    cfg.whisper_model = whisper_model
    cfg.huggingface_token = huggingface_token
    cfg.scratch_retention_days = scratch_retention_days
    save_config(cfg)
    return templates.TemplateResponse(request, "settings.html", await _settings_context(request, cfg, saved=True))


@router.get("/status", response_class=HTMLResponse)
async def status_bar(request: Request):
    cfg = load_config()

    # Obsidian check
    try:
        async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
            r = await client.get(cfg.obsidian_api_url + "/")
        obsidian_ok = r.status_code < 500
    except Exception:
        obsidian_ok = False

    # LLM check
    ctx_len = await detect_context_length(cfg.llm_base_url, cfg.llm_model)
    llm_ok = ctx_len != 6000 or cfg.llm_model  # rough proxy

    # ffmpeg check
    ffmpeg_ok = shutil.which("ffmpeg") is not None

    # pyannote model check
    try:
        from pyannote.audio import Pipeline
        pyannote_ok = True
    except Exception:
        pyannote_ok = False

    indicators = [
        ("Obsidian API", obsidian_ok, "Start Obsidian with Local REST API plugin enabled"),
        ("LLM", llm_ok, "Start LM Studio and load a model"),
        ("ffmpeg", ffmpeg_ok, "Run: brew install ffmpeg"),
        ("pyannote", pyannote_ok, "Set HuggingFace token in Settings"),
    ]

    html = ""
    for name, ok, hint in indicators:
        dot_class = "green" if ok else "red"
        hint_attr = f'title="{hint}"' if not ok else ""
        html += f'<span class="status-indicator" {hint_attr}><span class="dot {dot_class}"></span>{name}</span>'

    return HTMLResponse(html)
