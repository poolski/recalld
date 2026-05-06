from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from recalld.app import templates
from recalld.events import bus
from recalld.jobs import DEFAULT_SCRATCH_ROOT, JobStage, JobStatus, can_restart_from_stage, delete_job, load_job, reset_job_for_rerun, save_job
from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.postprocess import PostProcessResult
from recalld.pipeline.vault import render_session_note_preview
from recalld.pipeline.runner import _normalize_note_title, run_pipeline
from recalld.runtime import spawn_pipeline_task

router = APIRouter(prefix="/jobs")


def _save_job(job) -> None:
    save_job(job, scratch_root=DEFAULT_SCRATCH_ROOT)


def _load_aligned_preview(job, limit: int = 5) -> str:
    if not job.aligned_path:
        return ""
    turns = [LabelledTurn(**t) for t in json.loads(Path(job.aligned_path).read_text())]
    preview = turns[:limit]
    return "\n".join(f"**{turn.speaker}:** {turn.text}" for turn in preview)


def _load_postprocess_state(job) -> dict:
    if not job.postprocess_path:
        return {}
    data = json.loads(Path(job.postprocess_path).read_text())
    return {
        "summary": data.get("summary", ""),
        "focus_points": data.get("focus_points", []),
        "strategy": data.get("strategy", ""),
        "topic_count": data.get("topic_count"),
    }


def _load_vault_preview(job, category) -> str:
    if not category or not job.postprocess_path or not job.aligned_path:
        return ""

    postprocess_state = _load_postprocess_state(job)
    if not postprocess_state:
        return ""

    turns = [LabelledTurn(**t) for t in json.loads(Path(job.aligned_path).read_text())]
    result = PostProcessResult(
        summary=postprocess_state.get("summary", ""),
        focus_points=postprocess_state.get("focus_points", []),
        raw_response="",
        strategy=postprocess_state.get("strategy", ""),
        topic_count=postprocess_state.get("topic_count") or 0,
    )
    return render_session_note_preview(
        session_date=job.created_at.date(),
        category=category.name,
        speakers=[category.speaker_a, category.speaker_b],
        result=result,
        turns=turns,
    )


def _vault_note_path(job, category) -> str | None:
    if not category:
        return None
    if job.note_target_path:
        return job.note_target_path
    session_date = job.created_at.date()
    filename = job.filename or f"{session_date.isoformat()} {category.name}.md"
    return f"{category.vault_path}/{filename}"


def _vault_uri(job, category, vault_name: str) -> str | None:
    note_path = _vault_note_path(job, category)
    if not note_path:
        return None
    from urllib.parse import quote

    return f"obsidian://open?vault={quote(vault_name, safe='')}&file={quote(note_path, safe='/')}"


def _swap_aligned_speakers(job) -> None:
    if not job.aligned_path:
        return
    turns = [LabelledTurn(**t) for t in json.loads(Path(job.aligned_path).read_text())]
    speakers = []
    for turn in turns:
        if turn.speaker not in speakers:
            speakers.append(turn.speaker)
        if len(speakers) == 2:
            break
    if len(speakers) != 2:
        return
    swap_map = {speakers[0]: speakers[1], speakers[1]: speakers[0]}
    for turn in turns:
        turn.speaker = swap_map.get(turn.speaker, turn.speaker)
    Path(job.aligned_path).write_text(json.dumps([turn.__dict__ for turn in turns]))


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


@router.get("/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str):
    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    return templates.TemplateResponse(request, "processing.html", {"job": job})


@router.get("/{job_id}/state", response_class=JSONResponse)
async def job_state(job_id: str):
    from recalld.config import load_config
    cfg = load_config()
    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    postprocess_state = _load_postprocess_state(job)

    obsidian_uri = None
    cat = next((c for c in cfg.categories if c.id == job.category_id), None)
    if cat and job.stage_statuses.get("vault") == "done":
        obsidian_uri = f"/jobs/{job.id}/open-in-obsidian"

    return JSONResponse({
        "id": job.id,
        "status": job.status.value,
        "current_stage": job.current_stage.value,
        "stage_statuses": job.stage_statuses,
        "preview": _load_aligned_preview(job),
        "filename": job.filename,
        "obsidian_uri": obsidian_uri,
        "error": job.error,
        "can_confirm_vault": job.stage_statuses.get("vault") == "awaiting_confirmation",
        "can_overwrite_vault_note": bool(job.vault_conflict_path),
        "can_append_vault_note": bool(job.vault_conflict_path),
        "vault_conflict_path": job.vault_conflict_path,
        "can_confirm_speakers": job.stage_statuses.get("align") == "awaiting_confirmation",
        "can_swap_speakers": job.stage_statuses.get("align") == "awaiting_confirmation",
        "vault_preview": _load_vault_preview(job, cat) if job.stage_statuses.get("vault") == "awaiting_confirmation" else "",
        **postprocess_state,
    })


@router.get("/{job_id}/events")
async def job_events(job_id: str):
    async def event_generator():
        import asyncio
        try:
            async for data in bus.subscribe(job_id):
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            # Expected when client disconnects or server shuts down.
            return

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{job_id}/open-in-obsidian")
async def open_in_obsidian(job_id: str):
    from recalld.config import load_config

    cfg = load_config()
    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    cat = next((c for c in cfg.categories if c.id == job.category_id), None)
    if not cat:
        return HTMLResponse("", status_code=404)

    note_path = _vault_note_path(job, cat)
    if not note_path:
        return HTMLResponse("", status_code=404)

    uri = _vault_uri(job, cat, cfg.vault_name)
    if not uri:
        return HTMLResponse("", status_code=404)

    return RedirectResponse(uri, status_code=302)


def _job_source_path(job_id: str, original_filename: str):
    job_dir = DEFAULT_SCRATCH_ROOT / job_id
    return job_dir / original_filename


async def _schedule_pipeline(request: Request, job_id: str, from_start: bool) -> HTMLResponse:
    from recalld.config import load_config

    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    reset_job_for_rerun(job, from_start=from_start)
    job.status = JobStatus.running
    _save_job(job)
    cfg = load_config()
    source = _job_source_path(job.id, job.original_filename)
    spawn_pipeline_task(run_pipeline(job, source, cfg))
    return templates.TemplateResponse(request, "processing.html", {"job": job})


async def _restart_from_stage(request: Request, job_id: str, stage: JobStage) -> HTMLResponse:
    from recalld.config import load_config

    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    if can_restart_from_stage(job, stage):
        reset_job_for_rerun(job, from_start=False, restart_stage=stage)
        if stage == JobStage.vault:
            job.status = JobStatus.pending
            job.stage_statuses["vault"] = "awaiting_confirmation"
            _save_job(job)
            bus.publish(job.id, {
                "stage": "vault",
                "status": "awaiting_confirmation",
                "can_confirm_vault": True,
                "filename": job.filename,
                "vault_preview": _load_vault_preview(
                    job,
                    next((c for c in load_config().categories if c.id == job.category_id), None),
                ),
            })
            return templates.TemplateResponse(request, "processing.html", {"job": job})
    job.status = JobStatus.running
    _save_job(job)
    cfg = load_config()
    source = _job_source_path(job.id, job.original_filename)
    spawn_pipeline_task(run_pipeline(job, source, cfg))
    return templates.TemplateResponse(request, "processing.html", {"job": job})


@router.post("/{job_id}/rerun-from-failed", response_class=HTMLResponse)
async def rerun_from_failed(request: Request, job_id: str):
    return await _schedule_pipeline(request, job_id, from_start=False)


@router.post("/{job_id}/rerun-from-start", response_class=HTMLResponse)
async def rerun_from_start(request: Request, job_id: str):
    return await _schedule_pipeline(request, job_id, from_start=True)


@router.post("/{job_id}/restart-from/{stage}", response_class=HTMLResponse)
async def restart_from_stage(request: Request, job_id: str, stage: JobStage):
    return await _restart_from_stage(request, job_id, stage)


@router.post("/{job_id}/confirm-speakers", response_class=HTMLResponse)
async def confirm_speakers(request: Request, job_id: str):
    from recalld.config import load_config
    from recalld.jobs import JobStage, JobStatus

    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    cfg = load_config()
    job.current_stage = JobStage.postprocess
    job.status = JobStatus.running
    job.stage_statuses["align"] = "done"
    _save_job(job)

    bus.publish(job.id, {"stage": "align", "status": "done"})

    source = _job_source_path(job.id, job.original_filename)
    spawn_pipeline_task(run_pipeline(job, source, cfg))
    return templates.TemplateResponse(request, "processing.html", {"job": job})


@router.post("/{job_id}/swap-speakers", response_class=HTMLResponse)
async def swap_speakers(request: Request, job_id: str):
    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    _swap_aligned_speakers(job)
    _save_job(job)
    bus.publish(job.id, {
        "stage": "align",
        "status": "awaiting_confirmation",
        "preview": _load_aligned_preview(job),
        "can_confirm_speakers": True,
    })
    return templates.TemplateResponse(request, "processing.html", {"job": job})


@router.post("/{job_id}/confirm-vault-write", response_class=HTMLResponse)
async def confirm_vault_write(
    request: Request,
    job_id: str,
    filename: str = Form(None),
    write_mode: str = Form(None),
):
    from recalld.config import load_config
    from recalld.jobs import JobStatus
    from recalld.pipeline.vault import VaultWriter

    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    cfg = load_config()
    cat = next((c for c in cfg.categories if c.id == job.category_id), None)
    if filename:
        sanitized = _normalize_note_title(filename, job.created_at.date())
        if sanitized:
            job.filename = sanitized
            job.note_target_path = f"{cat.vault_path}/{sanitized}" if cat else job.note_target_path
    if not cat:
        job.status = JobStatus.failed
        job.error = "Category not found"
        job.stage_statuses["vault"] = "failed"
        _save_job(job)
        return templates.TemplateResponse(request, "processing.html", {"job": job})

    note_name = job.filename or f"{job.created_at.date().isoformat()} {cat.name}.md"
    note_path = job.note_target_path or f"{cat.vault_path}/{note_name}"
    normalized_mode = (write_mode or "").strip().lower()
    existing_target = job.note_target_mode == "existing"

    if normalized_mode and normalized_mode not in {"overwrite", "append"}:
        return HTMLResponse("Invalid write mode", status_code=400)

    if existing_target:
        normalized_mode = "append"

    if normalized_mode == "append" and not existing_target:
        writer = VaultWriter(cfg.obsidian_api_url, cfg.obsidian_api_key)
        if not await writer.note_exists(note_path):
            normalized_mode = "overwrite"
    elif not normalized_mode:
        writer = VaultWriter(cfg.obsidian_api_url, cfg.obsidian_api_key)
        if await writer.note_exists(note_path):
            job.vault_conflict_path = note_path
            job.stage_statuses["vault"] = "awaiting_confirmation"
            job.status = JobStatus.pending
            _save_job(job)
            bus.publish(job.id, {
                "stage": "vault",
                "status": "awaiting_confirmation",
                "can_confirm_vault": True,
                "filename": note_name,
                "vault_conflict_path": note_path,
                "can_overwrite_vault_note": True,
                "can_append_vault_note": True,
            })
            return templates.TemplateResponse(request, "processing.html", {"job": job})
        normalized_mode = "overwrite"

    job.filename = note_name
    job.note_target_path = note_path
    job.vault_write_mode = normalized_mode
    job.vault_conflict_path = None
    job.status = JobStatus.running
    job.stage_statuses["vault"] = "pending"
    _save_job(job)

    source = _job_source_path(job.id, job.original_filename)
    spawn_pipeline_task(run_pipeline(job, source, cfg))
    return templates.TemplateResponse(request, "processing.html", {"job": job})


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
    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
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
    job.stage_statuses["diarise"] = "failed"
    job.stage_statuses["align"] = "done"
    job.current_stage = JobStage.postprocess
    job.status = JobStatus.running
    _save_job(job)

    bus.publish(job.id, {"stage": "align", "status": "done"})

    source = _job_source_path(job.id, job.original_filename)
    spawn_pipeline_task(run_pipeline(job, source, cfg))
    return templates.TemplateResponse(request, "processing.html", {"job": job})


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
    job = load_job(job_id, scratch_root=DEFAULT_SCRATCH_ROOT)
    labelled = [LabelledTurn(**t) for t in json.loads(Path(job.aligned_path).read_text())]
    cat = next((c for c in cfg.categories if c.id == job.category_id), None)

    if cat:
        writer = VaultWriter(cfg.obsidian_api_url, cfg.obsidian_api_key)
        session_date = job.created_at.date()
        filename = job.filename or f"{session_date.isoformat()} {cat.name}.md"
        note_path = job.note_target_path or f"{cat.vault_path}/{filename}"
        content = render_session_note(
            session_date=session_date,
            category=cat.name,
            speakers=[cat.speaker_a, cat.speaker_b],
            result=None,
            turns=labelled,
        )
        note_dir, note_name = note_path.rsplit("/", 1)
        await writer.write_note(note_dir, note_name, content)
        job.status = JobStatus.complete
        job.stage_statuses["vault"] = "done"
        _save_job(job)
        bus.publish(job.id, {"stage": "vault", "status": "done",
                             "obsidian_uri": f"/jobs/{job.id}/open-in-obsidian",
                             "summary": "", "focus_points": []})

    return templates.TemplateResponse(request, "processing.html", {"job": job})
