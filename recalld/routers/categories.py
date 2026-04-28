from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from recalld.app import templates
from recalld.config import Category, load_config, save_config

router = APIRouter(prefix="/categories")


@router.post("/", response_class=HTMLResponse)
async def add_category(
    request: Request,
    name: str = Form(...),
    vault_path: str = Form(...),
    focus_note_path: str = Form(""),
    speaker_a: str = Form("Speaker A"),
    speaker_b: str = Form("Speaker B"),
):
    cfg = load_config()
    cat = Category(
        name=name,
        vault_path=vault_path,
        focus_note_path=focus_note_path or None,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )
    cfg.categories.append(cat)
    save_config(cfg)
    return RedirectResponse("/", status_code=303)


@router.post("/{cat_id}/delete")
async def delete_category(cat_id: str):
    cfg = load_config()
    cfg.categories = [c for c in cfg.categories if c.id != cat_id]
    if cfg.last_used_category == cat_id:
        cfg.last_used_category = cfg.categories[0].id if cfg.categories else None
    save_config(cfg)
    return RedirectResponse("/", status_code=303)


@router.post("/{cat_id}/speakers")
async def update_speakers(
    cat_id: str,
    speaker_a: str = Form(...),
    speaker_b: str = Form(...),
):
    cfg = load_config()
    for cat in cfg.categories:
        if cat.id == cat_id:
            cat.speaker_a = speaker_a
            cat.speaker_b = speaker_b
    save_config(cfg)
    return RedirectResponse("/", status_code=303)
