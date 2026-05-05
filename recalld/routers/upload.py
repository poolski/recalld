from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from recalld.app import templates
from recalld.config import DEFAULT_CONFIG_PATH, load_config, save_config
from recalld.jobs import DEFAULT_SCRATCH_ROOT, JobStatus, create_job, list_incomplete_jobs, list_jobs
from recalld.pipeline.runner import run_pipeline
from recalld.runtime import spawn_pipeline_task

router = APIRouter()


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


@router.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    category_id: str = Form(...),
    note_title: str = Form(""),
):
    cfg = load_config()
    job = create_job(
        category_id=category_id,
        original_filename=file.filename,
        scratch_root=DEFAULT_SCRATCH_ROOT,
    )
    cleaned_note_title = " ".join(note_title.split()).strip()
    if cleaned_note_title:
        if cleaned_note_title.lower().endswith(".md"):
            cleaned_note_title = cleaned_note_title[:-3].strip()
        if not cleaned_note_title.startswith(job.created_at.date().isoformat()):
            cleaned_note_title = f"{job.created_at.date().isoformat()} {cleaned_note_title}".strip()
        job.filename = f"{cleaned_note_title}.md"
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
