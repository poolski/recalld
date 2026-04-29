from __future__ import annotations

from html import escape
import subprocess
import shutil

import httpx
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from recalld.app import templates
from recalld.config import DEFAULT_CONFIG_PATH, load_config, save_config
from recalld.llm.context import detect_context_length, list_available_models

router = APIRouter(prefix="/settings")


async def _check_obsidian_api(cfg) -> dict:
    try:
        async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
            resp = await client.get(cfg.obsidian_api_url + "/")
        ok = resp.status_code < 500
        status_code = resp.status_code
        error = ""
    except Exception as exc:
        ok = False
        status_code = None
        error = str(exc)
    return {
        "ok": ok,
        "status_code": status_code,
        "error": error,
        "url": cfg.obsidian_api_url,
        "vault_name": cfg.vault_name,
    }


async def _check_llm(cfg) -> dict:
    provider_models = await list_available_models(cfg.llm_base_url, cfg.llm_model)
    current_model = next((model for model in provider_models if model.selected), None)
    detected_context_length = current_model.context_length if current_model and current_model.context_length else await detect_context_length(cfg.llm_base_url, cfg.llm_model)
    return {
        "ok": bool(cfg.llm_model) and bool(provider_models),
        "provider_models": provider_models,
        "current_model": current_model,
        "detected_context_length": detected_context_length,
        "base_url": cfg.llm_base_url,
        "selected_model": cfg.llm_model or "Not set",
    }


def _check_ffmpeg() -> dict:
    path = shutil.which("ffmpeg")
    version = ""
    ok = bool(path)
    if path:
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False, timeout=3)
            version = result.stdout.splitlines()[0] if result.stdout else ""
        except Exception as exc:
            version = str(exc)
            ok = False
    return {
        "ok": ok,
        "path": path or "Not found",
        "version": version or ("Not found" if not path else "Unavailable"),
    }


def _check_pyannote(cfg) -> dict:
    try:
        import pyannote.audio  # noqa: F401
        installed = True
    except Exception:
        installed = False
    ready = installed and bool(cfg.huggingface_token)
    return {
        "ok": ready,
        "installed": installed,
        "token_set": bool(cfg.huggingface_token),
    }


async def _status_payload(cfg) -> dict[str, dict]:
    obsidian = await _check_obsidian_api(cfg)
    llm = await _check_llm(cfg)
    ffmpeg = _check_ffmpeg()
    pyannote = _check_pyannote(cfg)
    return {
        "obsidian": obsidian,
        "llm": llm,
        "ffmpeg": ffmpeg,
        "pyannote": pyannote,
    }


async def _settings_context(request: Request, cfg, *, saved: bool = False) -> dict:
    llm = await _check_llm(cfg)
    model_refresh_unavailable = bool(cfg.llm_model) and not llm["provider_models"]
    return {
        "request": request,
        "cfg": cfg,
        "provider_models": llm["provider_models"],
        "current_model": llm["current_model"],
        "detected_context_length": llm["detected_context_length"],
        "model_refresh_unavailable": model_refresh_unavailable,
        "saved": saved,
        "obsidian_api_key_set": bool(cfg.obsidian_api_key),
        "huggingface_token_set": bool(cfg.huggingface_token),
    }


@router.get("/", response_class=HTMLResponse)
async def settings_page(request: Request):
    cfg = load_config()
    return templates.TemplateResponse(request, "settings.html", await _settings_context(request, cfg))


@router.post("/", response_class=HTMLResponse)
async def save_settings(
    request: Request,
    vault_name: str = Form("Personal"),
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
    cfg.vault_name = vault_name
    cfg.obsidian_api_url = obsidian_api_url
    if obsidian_api_key:
        cfg.obsidian_api_key = obsidian_api_key
    cfg.llm_base_url = llm_base_url
    cfg.llm_model = llm_model
    cfg.llm_context_headroom = llm_context_headroom
    cfg.log_level = log_level
    cfg.whisper_model = whisper_model
    if huggingface_token:
        cfg.huggingface_token = huggingface_token
    cfg.scratch_retention_days = scratch_retention_days
    save_config(cfg)
    return templates.TemplateResponse(request, "settings.html", await _settings_context(request, cfg, saved=True))


@router.get("/status", response_class=HTMLResponse)
async def status_bar(request: Request):
    cfg = load_config()
    payload = await _status_payload(cfg)

    indicators = [
        ("obsidian", "Obsidian API", payload["obsidian"]["ok"], "Start Obsidian with Local REST API plugin enabled"),
        ("llm", "LLM", payload["llm"]["ok"], "Start LM Studio and load a model"),
        ("ffmpeg", "ffmpeg", payload["ffmpeg"]["ok"], "Run: brew install ffmpeg"),
        ("pyannote", "pyannote", payload["pyannote"]["ok"], "Set HuggingFace token in Settings"),
    ]

    status_html = ""
    for kind, name, ok, hint in indicators:
        dot_class = "green" if ok else "red"
        hint_attr = f'title="{hint}"' if not ok else ""
        status_html += (
            f'<button type="button" class="status-indicator" data-status-kind="{escape(kind)}" '
            f'data-status-title="{escape(name)}" {hint_attr}>'
            f'<span class="dot {dot_class}"></span>{escape(name)}</button>'
        )

    return HTMLResponse(status_html)


@router.get("/status/details", response_class=JSONResponse)
async def status_details(kind: str):
    cfg = load_config()
    payload = await _status_payload(cfg)

    if kind not in payload:
        return JSONResponse({"error": f"Unknown status kind: {kind}"}, status_code=400)

    if kind == "llm":
        llm = payload[kind]
        current_model = llm["current_model"]
        loaded_length = current_model.loaded_context_length if current_model else None
        max_length = current_model.max_context_length if current_model else None
        items = [
            {"label": "Selected model", "value": llm["selected_model"]},
            {"label": "Base URL", "value": llm["base_url"]},
            {"label": "Loaded", "value": "Yes" if current_model and current_model.is_loaded else "No"},
            {"label": "Loaded context length", "value": f"{loaded_length:,}" if loaded_length is not None else "Not loaded"},
            {"label": "Maximum context length", "value": f"{max_length:,}" if max_length is not None else "Unknown"},
            {"label": "Available models", "value": str(len(llm["provider_models"]))},
        ]
        return JSONResponse({"title": "LLM", "ok": llm["ok"], "items": items})

    if kind == "obsidian":
        obsidian = payload[kind]
        items = [
            {"label": "Vault name", "value": obsidian["vault_name"]},
            {"label": "API URL", "value": obsidian["url"]},
            {"label": "Health", "value": "Healthy" if obsidian["ok"] else "Unhealthy"},
            {"label": "HTTP status", "value": str(obsidian["status_code"] or "n/a")},
            {"label": "Auth key", "value": "Set" if cfg.obsidian_api_key else "Not set"},
        ]
        if obsidian["error"]:
            items.append({"label": "Error", "value": obsidian["error"]})
        return JSONResponse({"title": "Obsidian API", "ok": obsidian["ok"], "items": items})

    if kind == "ffmpeg":
        ffmpeg = payload[kind]
        items = [
            {"label": "Binary", "value": ffmpeg["path"]},
            {"label": "Version", "value": ffmpeg["version"]},
        ]
        return JSONResponse({"title": "ffmpeg", "ok": ffmpeg["ok"], "items": items})

    pyannote = payload[kind]
    items = [
        {"label": "Installed", "value": "Yes" if pyannote["installed"] else "No"},
        {"label": "HuggingFace token", "value": "Set" if pyannote["token_set"] else "Not set"},
    ]
    return JSONResponse({"title": "pyannote", "ok": pyannote["ok"], "items": items})
