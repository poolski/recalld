from __future__ import annotations

import asyncio
from pathlib import Path

from recalld.config import Config, load_config
from recalld.events import bus
from recalld.jobs import Job, JobStage, JobStatus, save_job, DEFAULT_SCRATCH_ROOT
from recalld.llm.context import detect_context_length, token_budget
from recalld.pipeline.align import align
from recalld.pipeline.diarise import diarise, DiariseError
from recalld.pipeline.ingest import ingest, IngestError
from recalld.pipeline.postprocess import postprocess
from recalld.pipeline.transcribe import transcribe
from recalld.pipeline.vault import VaultWriter, render_session_note, render_focus_section
from datetime import date


def _emit(job: Job, stage: str, status: str, message: str = "", **extra) -> None:
    bus.publish(job.id, {"stage": stage, "status": status, "message": message, **extra})


async def run_pipeline(job: Job, source_path: Path, cfg: Config) -> None:
    scratch = DEFAULT_SCRATCH_ROOT / job.id
    job.status = JobStatus.running

    try:
        # --- Ingest ---
        if job.current_stage == JobStage.ingest:
            _emit(job, "ingest", "running")
            try:
                wav = await asyncio.to_thread(ingest, source_path, scratch)
            except IngestError as e:
                job.status = JobStatus.failed
                job.error = str(e)
                save_job(job)
                _emit(job, "ingest", "failed", str(e))
                return
            job.wav_path = str(wav)
            job.current_stage = JobStage.transcribe
            save_job(job)
            _emit(job, "ingest", "done")

        # --- Transcribe ---
        if job.current_stage == JobStage.transcribe:
            _emit(job, "transcribe", "running")
            import json
            try:
                words = await asyncio.to_thread(transcribe, Path(job.wav_path), cfg.whisper_model)
            except Exception as e:
                job.status = JobStatus.failed
                job.error = str(e)
                save_job(job)
                _emit(job, "transcribe", "failed", str(e))
                return
            transcript_path = scratch / "transcript.json"
            transcript_path.write_text(json.dumps([w.__dict__ for w in words]))
            job.transcript_path = str(transcript_path)
            job.current_stage = JobStage.diarise
            save_job(job)
            _emit(job, "transcribe", "done")

        # --- Diarise ---
        if job.current_stage == JobStage.diarise:
            _emit(job, "diarise", "running")
            import json
            try:
                turns = await asyncio.to_thread(diarise, Path(job.wav_path), cfg.huggingface_token)
            except DiariseError as e:
                # Offer to continue with unlabelled transcript
                job.status = JobStatus.failed
                job.error = str(e)
                save_job(job)
                _emit(job, "diarise", "failed", str(e), can_skip=True)
                return
            diar_path = scratch / "diarisation.json"
            diar_path.write_text(json.dumps([t.__dict__ for t in turns]))
            job.diarisation_path = str(diar_path)
            job.current_stage = JobStage.align
            save_job(job)
            _emit(job, "diarise", "done")

        # --- Align ---
        if job.current_stage == JobStage.align:
            import json
            from recalld.pipeline.transcribe import WordSegment
            from recalld.pipeline.diarise import SpeakerTurn

            words = [WordSegment(**w) for w in json.loads(Path(job.transcript_path).read_text())]
            raw_turns = [SpeakerTurn(**t) for t in json.loads(Path(job.diarisation_path).read_text())]

            cat = next((c for c in cfg.categories if c.id == job.category_id), None)
            speaker_map = None
            if cat:
                speaker_map = {"SPEAKER_00": cat.speaker_a, "SPEAKER_01": cat.speaker_b}
                job.speaker_00 = cat.speaker_a
                job.speaker_01 = cat.speaker_b

            labelled = align(words, raw_turns, speaker_map=speaker_map)
            aligned_path = scratch / "aligned.json"
            aligned_path.write_text(json.dumps([t.__dict__ for t in labelled]))
            job.aligned_path = str(aligned_path)
            job.current_stage = JobStage.postprocess
            save_job(job)

            preview = labelled[:5]
            preview_text = "\n".join(f"**{t.speaker}:** {t.text}" for t in preview)
            _emit(job, "align", "done", preview=preview_text, needs_speaker_names=(speaker_map is None))

        # --- Post-process ---
        if job.current_stage == JobStage.postprocess:
            import json
            from recalld.pipeline.align import LabelledTurn

            _emit(job, "postprocess", "running")
            labelled = [LabelledTurn(**t) for t in json.loads(Path(job.aligned_path).read_text())]

            ctx_len = await detect_context_length(cfg.llm_base_url, cfg.llm_model)
            budget = token_budget(ctx_len, cfg.llm_context_headroom)

            try:
                result = await postprocess(
                    turns=labelled,
                    llm_base_url=cfg.llm_base_url,
                    llm_model=cfg.llm_model,
                    token_budget=budget,
                    progress_cb=lambda msg: _emit(job, "postprocess", "running", msg),
                )
            except Exception as e:
                job.status = JobStatus.failed
                job.error = str(e)
                save_job(job)
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
            job.current_stage = JobStage.vault
            save_job(job)
            _emit(job, "postprocess", "done", topic_count=result.topic_count, strategy=result.strategy)

        # --- Vault write ---
        if job.current_stage == JobStage.vault:
            import json
            from recalld.pipeline.align import LabelledTurn
            from recalld.pipeline.postprocess import PostProcessResult

            _emit(job, "vault", "running")

            labelled = [LabelledTurn(**t) for t in json.loads(Path(job.aligned_path).read_text())]
            pp_data = json.loads(Path(job.postprocess_path).read_text()) if job.postprocess_path else None

            result = PostProcessResult(**pp_data, raw_response="") if pp_data else None
            cat = next((c for c in cfg.categories if c.id == job.category_id), None)

            if not cat:
                job.status = JobStatus.failed
                job.error = "Category not found"
                save_job(job)
                _emit(job, "vault", "failed", "Category not found")
                return

            writer = VaultWriter(cfg.obsidian_api_url, cfg.obsidian_api_key)
            session_date = date.today()
            filename = f"{session_date.isoformat()} {cat.name}.md"
            note_content = render_session_note(
                session_date=session_date,
                category=cat.name,
                speakers=[cat.speaker_a, cat.speaker_b],
                result=result,
                turns=labelled,
            )

            try:
                await writer.write_note(cat.vault_path, filename, note_content)
            except Exception as e:
                job.status = JobStatus.failed
                job.error = str(e)
                save_job(job)
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
            save_job(job)
            _emit(job, "vault", "done",
                  obsidian_uri=f"obsidian://open?path={quote_path(cat.vault_path + '/' + filename)}",
                  summary=result.summary if result else "",
                  focus_points=result.focus_points if result else [])

    except Exception as e:
        job.status = JobStatus.failed
        job.error = str(e)
        save_job(job)
        _emit(job, job.current_stage.value, "failed", str(e))


def quote_path(path: str) -> str:
    from urllib.parse import quote
    return quote(path)
