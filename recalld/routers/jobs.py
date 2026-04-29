from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from recalld.app import templates
from recalld.events import bus
from recalld.jobs import DEFAULT_SCRATCH_ROOT, delete_job, load_job
from recalld.pipeline.runner import run_pipeline, quote_path

router = APIRouter(prefix="/jobs")


@router.get("/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str):
    job = load_job(job_id)
    return templates.TemplateResponse("processing.html", {"request": request, "job": job})


@router.get("/{job_id}/row", response_class=HTMLResponse)
async def job_row(request: Request, job_id: str):
    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    return templates.TemplateResponse(request, "partials/job_row.html", {"job": job})


@router.get("/{job_id}/confirm-delete", response_class=HTMLResponse)
async def confirm_delete(request: Request, job_id: str):
    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    return templates.TemplateResponse(request, "partials/job_confirm_delete.html", {"job": job})


@router.delete("/{job_id}", response_class=HTMLResponse)
async def delete_job_route(job_id: str):
    delete_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    return HTMLResponse("")


@router.get("/{job_id}/events")
async def job_events(job_id: str):
    async def event_generator():
        async for data in bus.subscribe(job_id):
            yield f"data: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{job_id}/resume", response_class=HTMLResponse)
async def resume_job(request: Request, job_id: str):
    import asyncio
    from pathlib import Path
    from recalld.config import load_config

    job = load_job(job_id)
    cfg = load_config()
    job_dir = DEFAULT_SCRATCH_ROOT / job.id
    source = job_dir / job.original_filename
    asyncio.create_task(run_pipeline(job, source, cfg))
    return templates.TemplateResponse("processing.html", {"request": request, "job": job})


@router.post("/{job_id}/skip-diarise", response_class=HTMLResponse)
async def skip_diarise(request: Request, job_id: str):
    """Continue pipeline with unlabelled transcript (SPEAKER_00, SPEAKER_01 labels)."""
    import asyncio
    import json
    from pathlib import Path
    from recalld.config import load_config
    from recalld.pipeline.align import align
    from recalld.pipeline.transcribe import WordSegment
    from recalld.jobs import JobStage, JobStatus, save_job

    cfg = load_config()
    job = load_job(job_id)
    scratch = DEFAULT_SCRATCH_ROOT / job_id

    # Build a dummy diarisation: one turn covering everything
    words = [WordSegment(**w) for w in json.loads(Path(job.transcript_path).read_text())]
    if words:
        from recalld.pipeline.diarise import SpeakerTurn
        full_turn = [SpeakerTurn(start=words[0].start, end=words[-1].end, speaker="SPEAKER_00")]
    else:
        full_turn = []

    labelled = align(words, full_turn)
    aligned_path = scratch / "aligned.json"
    aligned_path.write_text(json.dumps([t.__dict__ for t in labelled]))
    job.aligned_path = str(aligned_path)
    job.current_stage = JobStage.postprocess
    job.status = JobStatus.running
    from recalld.jobs import save_job
    save_job(job)

    source = scratch / job.original_filename
    asyncio.create_task(run_pipeline(job, source, cfg))
    return templates.TemplateResponse("processing.html", {"request": request, "job": job})


@router.post("/{job_id}/write-transcript-only", response_class=HTMLResponse)
async def write_transcript_only(request: Request, job_id: str):
    """Write session note to vault with post_processing: failed, no summary."""
    import json
    from pathlib import Path
    from datetime import date
    from recalld.config import load_config
    from recalld.pipeline.align import LabelledTurn
    from recalld.pipeline.vault import VaultWriter, render_session_note
    from recalld.jobs import JobStatus, save_job

    cfg = load_config()
    job = load_job(job_id)
    labelled = [LabelledTurn(**t) for t in json.loads(Path(job.aligned_path).read_text())]
    cat = next((c for c in cfg.categories if c.id == job.category_id), None)

    if cat:
        writer = VaultWriter(cfg.obsidian_api_url, cfg.obsidian_api_key)
        session_date = date.today()
        filename = f"{session_date.isoformat()} {cat.name}.md"
        content = render_session_note(
            session_date=session_date,
            category=cat.name,
            speakers=[cat.speaker_a, cat.speaker_b],
            result=None,
            turns=labelled,
        )
        await writer.write_note(cat.vault_path, filename, content)
        job.status = JobStatus.complete
        save_job(job)
        bus.publish(job.id, {"stage": "vault", "status": "done",
                             "obsidian_uri": f"obsidian://open?path={quote_path(cat.vault_path + '/' + filename)}",
                             "summary": "", "focus_points": []})

    return templates.TemplateResponse("processing.html", {"request": request, "job": job})
