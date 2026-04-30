from __future__ import annotations

import asyncio
import re
from pathlib import Path

from recalld.config import Config, load_config
from recalld.events import bus
from recalld.jobs import Job, JobStage, JobStatus, save_job, DEFAULT_SCRATCH_ROOT
from recalld.llm.client import LLMClient
from recalld.llm.context import ensure_loaded_context_length, token_budget
from recalld.pipeline.align import align
from recalld.pipeline.diarise import diarise, DiariseError
from recalld.pipeline.ingest import ingest, IngestError
from recalld.pipeline.postprocess import postprocess
from recalld.pipeline.transcribe import transcribe
from recalld.pipeline.vault import VaultWriter, render_session_note, render_session_note_preview, render_focus_section
from datetime import date


def _emit(job: Job, stage: str, status: str, message: str = "", **extra) -> None:
    bus.publish(job.id, {"stage": stage, "status": status, "message": message, **extra})


def _emit_lmstudio_event(job: Job, event_type: str, data: dict) -> None:
    message = ""
    extra = {"lmstudio_event": event_type}

    if event_type == "prompt_processing.start":
        message = "LM Studio is processing the prompt."
    elif event_type == "prompt_processing.progress":
        progress = data.get("progress")
        if isinstance(progress, (int, float)):
            message = f"LM Studio prompt processing: {round(progress * 100)}%"
            extra["lmstudio_progress"] = progress
        else:
            message = "LM Studio prompt processing in progress."
    elif event_type == "prompt_processing.end":
        message = "LM Studio finished processing the prompt."
    elif event_type == "reasoning.start":
        message = "LM Studio started reasoning."
    elif event_type == "reasoning.end":
        message = "LM Studio finished reasoning."
    elif event_type == "message.start":
        message = "LM Studio started streaming the response."
    elif event_type == "message.end":
        message = "LM Studio finished streaming the response."
    elif event_type == "chat.start":
        message = "LM Studio chat started."
    elif event_type == "chat.end":
        message = "LM Studio chat completed."
    elif event_type == "error":
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message", "")
            if error.get("type"):
                extra["lmstudio_error_type"] = error["type"]
        else:
            message = "LM Studio reported an error."

    if message:
        _emit(job, "postprocess", "running", message, **extra)


def _set_stage_status(job: Job, stage: str, status: str) -> None:
    job.stage_statuses[stage] = status


def _save(job: Job) -> None:
    save_job(job, scratch_root=DEFAULT_SCRATCH_ROOT)


def _build_speaker_map(raw_turns, speaker_a: str, speaker_b: str) -> dict[str, str]:
    speakers: list[str] = []
    for turn in raw_turns:
        if turn.speaker not in speakers:
            speakers.append(turn.speaker)
        if len(speakers) == 2:
            break

    speaker_map: dict[str, str] = {}
    if speakers:
        speaker_map[speakers[0]] = speaker_a
    if len(speakers) > 1:
        speaker_map[speakers[1]] = speaker_b
    return speaker_map


def _normalize_note_title(title: str, session_date) -> str:
    cleaned = re.sub(r"\s+", " ", (title or "").strip())
    cleaned = cleaned.replace("/", "-").replace("\\", "-")
    if not cleaned:
        return ""
    if cleaned.lower().endswith(".md"):
        cleaned = cleaned[:-3].rstrip()
    if not cleaned.startswith(session_date.isoformat()):
        cleaned = f"{session_date.isoformat()} {cleaned}".strip()
    return f"{cleaned}.md"


async def _infer_note_title_with_llm(job: Job, cfg: Config, category_name: str, vault_path: str, labelled) -> str:
    session_date = job.created_at.date()
    transcript_excerpt = "\n".join(f"{t.speaker}: {t.text}" for t in labelled[:40])
    system = (
        "You create concise Obsidian note titles for conversation recordings. "
        "Return only one line in this exact format: YYYY-MM-DD Title. "
        "Do not include markdown, quotes, filename extension, slashes, or extra commentary."
    )
    user = (
        f"Session date: {session_date.isoformat()}\n"
        f"Category: {category_name}\n"
        f"Vault path: {vault_path}\n"
        f"Transcript excerpt:\n{transcript_excerpt}\n\n"
        "Generate a specific but concise title."
    )
    client = LLMClient(base_url=cfg.llm_base_url, model=cfg.llm_model)
    raw = await client.complete(system, user)
    return _normalize_note_title(raw, session_date)


async def run_pipeline(job: Job, source_path: Path, cfg: Config) -> None:
    scratch = DEFAULT_SCRATCH_ROOT / job.id
    job.status = JobStatus.running

    try:
        # --- Ingest ---
        if job.current_stage == JobStage.ingest:
            _set_stage_status(job, "ingest", "running")
            _save(job)
            _emit(job, "ingest", "running")
            try:
                wav = await asyncio.to_thread(ingest, source_path, scratch)
            except IngestError as e:
                job.status = JobStatus.failed
                job.error = str(e)
                _set_stage_status(job, "ingest", "failed")
                _save(job)
                _emit(job, "ingest", "failed", str(e))
                return
            job.wav_path = str(wav)
            _set_stage_status(job, "ingest", "done")
            job.current_stage = JobStage.transcribe
            _save(job)
            _emit(job, "ingest", "done")

        # --- Transcribe ---
        if job.current_stage == JobStage.transcribe:
            _set_stage_status(job, "transcribe", "running")
            _save(job)
            _emit(job, "transcribe", "running")
            import json
            try:
                words = await asyncio.to_thread(transcribe, Path(job.wav_path), cfg.whisper_model)
            except Exception as e:
                job.status = JobStatus.failed
                job.error = str(e)
                _set_stage_status(job, "transcribe", "failed")
                _save(job)
                _emit(job, "transcribe", "failed", str(e))
                return
            transcript_path = scratch / "transcript.json"
            transcript_path.write_text(json.dumps([w.__dict__ for w in words]))
            job.transcript_path = str(transcript_path)
            _set_stage_status(job, "transcribe", "done")
            job.current_stage = JobStage.diarise
            _save(job)
            _emit(job, "transcribe", "done")

        # --- Diarise ---
        if job.current_stage == JobStage.diarise:
            _set_stage_status(job, "diarise", "running")
            _save(job)
            _emit(job, "diarise", "running")
            import json
            try:
                turns = await asyncio.to_thread(diarise, Path(job.wav_path), cfg.huggingface_token)
            except DiariseError as e:
                # Offer to continue with unlabelled transcript
                job.status = JobStatus.failed
                job.error = str(e)
                _set_stage_status(job, "diarise", "failed")
                _save(job)
                _emit(job, "diarise", "failed", str(e), can_skip=True)
                return
            diar_path = scratch / "diarisation.json"
            diar_path.write_text(json.dumps([t.__dict__ for t in turns]))
            job.diarisation_path = str(diar_path)
            _set_stage_status(job, "diarise", "done")
            job.current_stage = JobStage.align
            _save(job)
            _emit(job, "diarise", "done")

        # --- Align ---
        if job.current_stage == JobStage.align:
            _set_stage_status(job, "align", "running")
            _save(job)
            import json
            from recalld.pipeline.transcribe import WordSegment
            from recalld.pipeline.diarise import SpeakerTurn

            words = [WordSegment(**w) for w in json.loads(Path(job.transcript_path).read_text())]
            raw_turns = [SpeakerTurn(**t) for t in json.loads(Path(job.diarisation_path).read_text())]

            cat = next((c for c in cfg.categories if c.id == job.category_id), None)
            speaker_map = None
            if cat:
                speaker_map = _build_speaker_map(raw_turns, cat.speaker_a, cat.speaker_b)
                job.speaker_00 = cat.speaker_a
                job.speaker_01 = cat.speaker_b

            labelled = align(words, raw_turns, speaker_map=speaker_map)
            aligned_path = scratch / "aligned.json"
            aligned_path.write_text(json.dumps([t.__dict__ for t in labelled]))
            job.aligned_path = str(aligned_path)
            _set_stage_status(job, "align", "awaiting_confirmation")
            job.status = JobStatus.pending
            _save(job)

            preview = labelled[:5]
            preview_text = "\n".join(f"**{t.speaker}:** {t.text}" for t in preview)
            _emit(
                job,
                "align",
                "awaiting_confirmation",
                preview=preview_text,
                can_confirm_speakers=True,
                can_swap_speakers=True,
            )
            return

        # --- Post-process ---
        if job.current_stage == JobStage.postprocess:
            import json
            from recalld.pipeline.align import LabelledTurn

            _set_stage_status(job, "postprocess", "running")
            _save(job)
            _emit(job, "postprocess", "running")
            labelled = [LabelledTurn(**t) for t in json.loads(Path(job.aligned_path).read_text())]
            cat = next((c for c in cfg.categories if c.id == job.category_id), None)
            speaker_a_name = cat.speaker_a if cat else "You"
            speaker_b_name = cat.speaker_b if cat else "Coach"

            ctx_len = await ensure_loaded_context_length(cfg.llm_base_url, cfg.llm_model)
            budget = token_budget(ctx_len, cfg.llm_context_headroom)

            try:
                result = await postprocess(
                    turns=labelled,
                    llm_base_url=cfg.llm_base_url,
                    llm_model=cfg.llm_model,
                    token_budget=budget,
                    progress_cb=lambda msg: _emit(job, "postprocess", "running", msg),
                    stream_cb=lambda text: _emit(job, "postprocess", "running", summary=text),
                    event_cb=lambda event_type, data: _emit_lmstudio_event(job, event_type, data),
                    speaker_a_name=speaker_a_name,
                    speaker_b_name=speaker_b_name,
                )
            except Exception as e:
                job.status = JobStatus.failed
                job.error = str(e)
                _set_stage_status(job, "postprocess", "failed")
                _save(job)
                _emit(job, "postprocess", "failed", str(e), can_write_transcript_only=True)
                return

            pp_path = scratch / "postprocess.json"
            pp_path.write_text(json.dumps({
                "summary": result.summary,
                "focus_points": result.focus_points,
                "strategy": result.strategy,
                "topic_count": result.topic_count,
            }))
            job.postprocess_path = str(pp_path)
            job.topic_count = result.topic_count
            job.chunk_strategy = result.strategy

            if not job.filename and cat:
                try:
                    inferred = await _infer_note_title_with_llm(job, cfg, cat.name, cat.vault_path, labelled)
                except Exception:
                    inferred = ""
                if inferred:
                    job.filename = inferred
                else:
                    session_date = job.created_at.date()
                    job.filename = f"{session_date.isoformat()} {cat.name}.md"

            _set_stage_status(job, "postprocess", "done")
            job.current_stage = JobStage.vault
            job.status = JobStatus.pending
            _set_stage_status(job, "vault", "awaiting_confirmation")
            _save(job)
            preview = render_session_note_preview(
                session_date=job.created_at.date(),
                category=cat.name if cat else "",
                speakers=[cat.speaker_a, cat.speaker_b] if cat else ["You", "Coach"],
                result=result,
                turns=labelled,
            ) if cat else ""
            _emit(
                job,
                "postprocess",
                "done",
                topic_count=result.topic_count,
                strategy=result.strategy,
                summary=result.summary,
                focus_points=result.focus_points,
            )
            _emit(job, "vault", "awaiting_confirmation", can_confirm_vault=True, filename=job.filename, vault_preview=preview)
            return

        # --- Vault write ---
        if job.current_stage == JobStage.vault:
            import json
            from recalld.pipeline.align import LabelledTurn
            from recalld.pipeline.postprocess import PostProcessResult

            if job.stage_statuses.get("vault") == "awaiting_confirmation":
                labelled = [LabelledTurn(**t) for t in json.loads(Path(job.aligned_path).read_text())]
                pp_data = json.loads(Path(job.postprocess_path).read_text()) if job.postprocess_path else None
                result = PostProcessResult(**pp_data, raw_response="") if pp_data else None
                cat = next((c for c in cfg.categories if c.id == job.category_id), None)
                preview = render_session_note_preview(
                    session_date=job.created_at.date(),
                    category=cat.name if cat else "",
                    speakers=[cat.speaker_a, cat.speaker_b] if cat else ["You", "Coach"],
                    result=result,
                    turns=labelled,
                ) if cat and result else ""
                _emit(job, "vault", "awaiting_confirmation", can_confirm_vault=True, filename=job.filename, vault_preview=preview)
                return

            _set_stage_status(job, "vault", "running")
            _save(job)
            _emit(job, "vault", "running")

            labelled = [LabelledTurn(**t) for t in json.loads(Path(job.aligned_path).read_text())]
            pp_data = json.loads(Path(job.postprocess_path).read_text()) if job.postprocess_path else None

            result = PostProcessResult(**pp_data, raw_response="") if pp_data else None
            cat = next((c for c in cfg.categories if c.id == job.category_id), None)

            if not cat:
                job.status = JobStatus.failed
                job.error = "Category not found"
                _set_stage_status(job, "vault", "failed")
                _save(job)
                _emit(job, "vault", "failed", "Category not found")
                return

            writer = VaultWriter(cfg.obsidian_api_url, cfg.obsidian_api_key)
            session_date = job.created_at.date()
            filename = job.filename or f"{session_date.isoformat()} {cat.name}.md"
            note_content = render_session_note(
                session_date=session_date,
                category=cat.name,
                speakers=[cat.speaker_a, cat.speaker_b],
                result=result,
                turns=labelled,
            )

            try:
                mode = (job.vault_write_mode or "overwrite").lower()
                if mode == "append":
                    await writer.append_note(cat.vault_path, filename, "\n\n" + note_content)
                else:
                    await writer.write_note(cat.vault_path, filename, note_content)
            except Exception as e:
                job.status = JobStatus.failed
                job.error = str(e)
                _set_stage_status(job, "vault", "failed")
                _save(job)
                _emit(job, "vault", "failed", str(e))
                return

            if cat.focus_note_path and result:
                focus_section = render_focus_section(session_date, result.focus_points)
                try:
                    exists = await writer.note_exists(cat.focus_note_path)
                    if not exists:
                        heading = f"# {cat.name} Focus\n"
                        await writer.write_note("", cat.focus_note_path, heading + focus_section)
                    else:
                        await writer.append_to_note(cat.focus_note_path, focus_section)
                except Exception:
                    pass  # Focus note failure does not affect session note

            job.status = JobStatus.complete
            _set_stage_status(job, "vault", "done")
            job.vault_write_mode = None
            job.vault_conflict_path = None
            _save(job)
            _emit(job, "vault", "done",
                  obsidian_uri=f"/jobs/{job.id}/open-in-obsidian",
                  summary=result.summary if result else "",
                  focus_points=result.focus_points if result else [])

    except Exception as e:
        job.status = JobStatus.failed
        job.error = str(e)
        _set_stage_status(job, job.current_stage.value, "failed")
        _save(job)
        _emit(job, job.current_stage.value, "failed", str(e))
