from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from recalld.app import templates
from recalld.config import DEFAULT_CONFIG_PATH, load_config, save_config
from recalld.jobs import DEFAULT_SCRATCH_ROOT, create_job, list_incomplete_jobs
from recalld.pipeline.runner import run_pipeline

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cfg = load_config()
    incomplete = list_incomplete_jobs()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cfg": cfg,
        "incomplete_jobs": incomplete,
    })


@router.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    category_id: str = Form(...),
):
    cfg = load_config()
    job = create_job(category_id=category_id, original_filename=file.filename)

    # Save uploaded file to scratch
    job_dir = DEFAULT_SCRATCH_ROOT / job.id
    dest = job_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Update last_used_category
    cfg.last_used_category = category_id
    save_config(cfg)

    # Start pipeline in background
    asyncio.create_task(run_pipeline(job, dest, cfg))

    return templates.TemplateResponse("processing.html", {
        "request": request,
        "job": job,
        "cfg": cfg,
    })
