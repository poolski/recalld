from __future__ import annotations

import asyncio
import re
import shutil
from datetime import date, datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from recalld.app import templates
from recalld.config import DEFAULT_CONFIG_PATH, load_config, save_config
from recalld.jobs import DEFAULT_SCRATCH_ROOT, JobStatus, create_job, list_incomplete_jobs, list_jobs
from recalld.pipeline.runner import _normalize_note_title, run_pipeline
from recalld.pipeline.vault import VaultWriter
from recalld.runtime import spawn_pipeline_task

router = APIRouter()


def _canonical_note_title(category_name: str, session_date: date) -> str:
    return _normalize_note_title(f"{session_date.isoformat()} {category_name}", session_date)


def _note_date(filename: str) -> date | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}) ", filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_note_candidate(filename: str, session_date: date, day_window: int = 7) -> bool:
    if not filename.endswith(".md"):
        return False
    note_date = _note_date(filename)
    if note_date is None:
        return False
    return abs((note_date - session_date).days) <= day_window


async def _note_target_state(category_id: str) -> dict:
    cfg = load_config()
    cat = next((c for c in cfg.categories if c.id == category_id), None)
    if not cat:
        return {"category": None, "note_title": "", "note_candidates": []}

    session_date = date.today()
    note_title = _canonical_note_title(cat.name, session_date)
    writer = VaultWriter(cfg.obsidian_api_url, cfg.obsidian_api_key)
    try:
        entries = await writer.list_directory(cat.vault_path)
    except Exception:
        entries = []
    note_candidates = [
        {
            "path": f"{cat.vault_path}/{entry}",
            "name": entry,
        }
        for entry in entries
        if _is_note_candidate(entry, session_date)
    ]
    note_candidates.sort(
        key=lambda item: (
            abs(((_note_date(item["name"]) or session_date) - session_date).days),
            item["name"],
        )
    )
    return {
        "category": cat,
        "note_title": note_title,
        "note_candidates": note_candidates,
    }


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cfg = load_config()
    incomplete = list_incomplete_jobs(scratch_root=DEFAULT_SCRATCH_ROOT)
    all_jobs = list_jobs(scratch_root=DEFAULT_SCRATCH_ROOT)
    active_tab = "jobs" if request.query_params.get("tab") == "jobs" else "new"
    selected_job = all_jobs[0] if all_jobs else None
    stage_counts: dict[str, int] = {}
    for job in all_jobs:
        stage_counts[job.current_stage.value] = stage_counts.get(job.current_stage.value, 0) + 1
    status_counts = {
        "running": sum(1 for job in all_jobs if job.status == JobStatus.running),
        "awaiting_confirmation": sum(
            1 for job in all_jobs if "awaiting_confirmation" in job.stage_statuses.values()
        ),
        "failed": sum(1 for job in all_jobs if job.status == JobStatus.failed),
        "complete": sum(1 for job in all_jobs if job.status == JobStatus.complete),
    }
    return templates.TemplateResponse(request, "index.html", {
        "cfg": cfg,
        "incomplete_jobs": incomplete,
        "all_jobs": all_jobs,
        "active_tab": active_tab,
        "selected_job": selected_job,
        "stage_counts": stage_counts,
        "status_counts": status_counts,
    })


@router.get("/upload/note-target", response_class=HTMLResponse)
async def note_target(request: Request, category_id: str):
    state = await _note_target_state(category_id)
    return templates.TemplateResponse(request, "partials/upload_note_target.html", state)


@router.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    category_id: str = Form(...),
    note_target: str = Form("new"),
):
    cfg = load_config()
    job = create_job(
        category_id=category_id,
        original_filename=file.filename,
        scratch_root=DEFAULT_SCRATCH_ROOT,
    )
    cat = next((c for c in cfg.categories if c.id == category_id), None)
    if note_target and note_target != "new":
        job.note_target_mode = "existing"
        job.note_target_path = note_target
        job.filename = Path(note_target).name
    else:
        job.note_target_mode = "new"
        normalized_note_title = _canonical_note_title(cat.name, job.created_at.date()) if cat else ""
        if normalized_note_title:
            job.filename = normalized_note_title
        if cat and job.filename:
            job.note_target_path = f"{cat.vault_path}/{job.filename}"
    from recalld.jobs import save_job
    save_job(job, scratch_root=DEFAULT_SCRATCH_ROOT)

    # Save uploaded file to scratch
    job_dir = DEFAULT_SCRATCH_ROOT / job.id
    dest = job_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Update last_used_category
    cfg.last_used_category = category_id
    save_config(cfg)

    # Start pipeline in background
    spawn_pipeline_task(run_pipeline(job, dest, cfg))

    return templates.TemplateResponse(request, "processing.html", {
        "job": job,
        "cfg": cfg,
    })
