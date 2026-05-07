import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from recalld.config import Category, Config
from recalld.pipeline.diarise import DiariseError, SpeakerTurn
from recalld.pipeline.transcribe import WordSegment
from recalld.jobs import JobStage, JobStatus, create_job, load_job, save_job
from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.postprocess import PostProcessResult
from recalld.pipeline.runner import _infer_note_title_with_llm, run_pipeline


@pytest.mark.asyncio
async def test_pipeline_waits_for_vault_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    scratch = tmp_path / job.id
    aligned_path = scratch / "aligned.json"
    aligned_path.write_text(json.dumps([
        LabelledTurn(speaker="You", start=0.0, end=1.0, text="Hello").__dict__,
    ]))
    job.aligned_path = str(aligned_path)
    job.current_stage = JobStage.postprocess
    save_job(job, scratch_root=tmp_path)

    cfg = Config(
        llm_model="test-model",
        llm_reasoning="off",
        categories=[Category(id="cat-1", name="Planning", vault_path="Notes/Sessions")],
    )
    result = PostProcessResult(
        summary="Summary",
        focus_points=["Follow up"],
        raw_response="",
        strategy="single",
        topic_count=1,
    )

    with patch("recalld.pipeline.runner.ensure_loaded_context_length", AsyncMock(return_value=1000)), \
         patch("recalld.pipeline.runner.postprocess", AsyncMock(return_value=result)), \
         patch("recalld.pipeline.runner._infer_note_title_with_llm", AsyncMock(return_value="2026-04-29 Project Alpha Meeting.md")), \
         patch("recalld.pipeline.runner.bus.publish") as mock_publish, \
         patch("recalld.pipeline.runner.VaultWriter") as MockWriter:
        await run_pipeline(job, scratch / "session.m4a", cfg)

    MockWriter.assert_not_called()
    assert any(
        call.args[1].get("stage") == "vault" and call.args[1].get("vault_preview")
        for call in mock_publish.call_args_list
    )
    reloaded = load_job(job.id, scratch_root=tmp_path)
    assert reloaded.current_stage == JobStage.vault
    assert reloaded.status == JobStatus.pending
    assert reloaded.stage_statuses["postprocess"] == "done"
    assert reloaded.stage_statuses["vault"] == "awaiting_confirmation"
    assert reloaded.postprocess_path is not None
    assert reloaded.filename == "2026-04-29 Project Alpha Meeting.md"


@pytest.mark.asyncio
async def test_pipeline_appends_to_existing_vault_note_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    scratch = tmp_path / job.id
    aligned_path = scratch / "aligned.json"
    aligned_path.write_text(json.dumps([
        LabelledTurn(speaker="You", start=0.0, end=1.0, text="Hello").__dict__,
    ]))
    postprocess_path = scratch / "postprocess.json"
    postprocess_path.write_text(json.dumps({
        "summary": "Summary",
        "focus_points": ["Follow up"],
        "strategy": "single",
        "topic_count": 1,
    }))
    job.aligned_path = str(aligned_path)
    job.postprocess_path = str(postprocess_path)
    job.filename = "2026-04-29 Project Alpha Meeting.md"
    job.vault_write_mode = "append"
    job.current_stage = JobStage.vault
    job.stage_statuses["postprocess"] = "done"
    job.stage_statuses["vault"] = "pending"
    save_job(job, scratch_root=tmp_path)

    cfg = Config(
        llm_model="test-model",
        categories=[Category(id="cat-1", name="Planning", vault_path="Notes/Sessions")],
    )

    with patch("recalld.pipeline.runner.render_session_note", return_value="\n".join([
        "## Summary",
        "Updated summary",
        "",
        "## Focus",
        "- [ ] Follow up",
        "",
        "## New Topic",
        "Fresh material",
        "",
        "## Transcript",
        "Existing transcript",
    ])), patch("recalld.pipeline.runner.VaultWriter") as MockWriter:
        writer = MockWriter.return_value
        writer.read_note = AsyncMock(return_value="\n".join([
            "## Summary",
            "Existing summary",
            "",
            "## Focus",
            "- [ ] Existing item",
            "",
            "## Transcript",
            "Existing transcript",
        ]))
        writer.patch_heading = AsyncMock()
        writer.append_to_heading = AsyncMock()
        writer.append_to_note = AsyncMock()
        writer.write_note = AsyncMock()
        writer.note_exists = AsyncMock(return_value=True)
        await run_pipeline(job, scratch / "session.m4a", cfg)

    assert writer.read_note.await_count == 1
    assert writer.write_note.await_count == 1
    assert writer.patch_heading.await_count == 0
    assert writer.append_to_heading.await_count == 0
    writer.append_to_note.assert_not_awaited()
    reloaded = load_job(job.id, scratch_root=tmp_path)
    assert reloaded.status == JobStatus.complete
    assert reloaded.vault_write_mode is None


@pytest.mark.asyncio
async def test_pipeline_uses_existing_note_content_when_editing_target_note(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    scratch = tmp_path / job.id
    aligned_path = scratch / "aligned.json"
    aligned_path.write_text(json.dumps([
        LabelledTurn(speaker="You", start=0.0, end=1.0, text="Hello").__dict__,
    ]))
    job.aligned_path = str(aligned_path)
    job.filename = "2026-05-05 Planning.md"
    job.note_target_mode = "existing"
    job.note_target_path = "Notes/Sessions/2026-05-05 Planning.md"
    job.current_stage = JobStage.postprocess
    save_job(job, scratch_root=tmp_path)

    cfg = Config(
        llm_model="test-model",
        categories=[Category(id="cat-1", name="Planning", vault_path="Notes/Sessions")],
    )
    result = PostProcessResult(
        summary="Summary",
        focus_points=["Follow up"],
        raw_response="",
        strategy="single",
        topic_count=1,
    )

    with patch("recalld.pipeline.runner.ensure_loaded_context_length", AsyncMock(return_value=1000)), \
         patch("recalld.pipeline.runner.postprocess", AsyncMock(return_value=result)) as mock_postprocess, \
         patch("recalld.pipeline.runner.VaultWriter") as MockWriter:
        writer = MockWriter.return_value
        writer.read_note = AsyncMock(return_value="## Summary\n- terse outline")
        writer.write_note = AsyncMock()
        writer.append_to_note = AsyncMock()
        writer.note_exists = AsyncMock(return_value=True)
        await run_pipeline(job, scratch / "session.m4a", cfg)

    assert mock_postprocess.call_args.kwargs["existing_note_content"] == "## Summary\n- terse outline"
    writer.read_note.assert_awaited_once_with("Notes/Sessions/2026-05-05 Planning.md")


@pytest.mark.asyncio
async def test_note_title_inference_uses_runtime_prompt_loader(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    labelled = [
        LabelledTurn(speaker="You", start=0.0, end=1.0, text="We should name this meeting"),
        LabelledTurn(speaker="Facilitator", start=1.0, end=2.0, text="What title fits best?"),
    ]

    prompt = SimpleNamespace(
        text="title-system",
        prompt="title-prompt",
        metadata={"prompt_name": "recalld/note-title", "prompt_source": "langfuse"},
    )
    calls: list[tuple[str, dict]] = []

    def fake_resolve(name, fallback, **variables):
        return prompt

    async def fake_complete(system, user, prompt=None, metadata=None):
        calls.append({"system": system, "prompt": prompt, "metadata": metadata})
        return "2026-05-06 Project Alpha Meeting"

    with patch("recalld.pipeline.runner.resolve_text_prompt", side_effect=fake_resolve), \
         patch("recalld.pipeline.runner.LLMClient") as MockClient:
        instance = MockClient.return_value
        instance.complete = AsyncMock(side_effect=fake_complete)
        result = await _infer_note_title_with_llm(job, Config(llm_model="test-model"), "Planning", "Notes/Sessions", labelled)

    assert result == f"{job.created_at.date().isoformat()} Project Alpha Meeting.md"
    assert calls[0]["prompt"] == "title-prompt"
    assert calls[0]["metadata"]["prompt_name"] == "recalld/note-title"


@pytest.mark.asyncio
async def test_pipeline_traces_root_and_postprocess_stage(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    scratch = tmp_path / job.id
    aligned_path = scratch / "aligned.json"
    aligned_path.write_text(json.dumps([
        LabelledTurn(speaker="You", start=0.0, end=1.0, text="Hello").__dict__,
    ]))
    job.aligned_path = str(aligned_path)
    job.current_stage = JobStage.postprocess
    save_job(job, scratch_root=tmp_path)

    cfg = Config(
        llm_model="test-model",
        categories=[Category(id="cat-1", name="Planning", vault_path="Notes/Sessions")],
    )
    result = PostProcessResult(
        summary="Summary",
        focus_points=["Follow up"],
        raw_response="",
        strategy="single",
        topic_count=1,
    )

    calls: list[tuple[str, dict]] = []

    class FakeObservation:
        def __init__(self, name: str, kwargs: dict):
            self.name = name
            self.kwargs = kwargs

        def __enter__(self):
            calls.append(("enter", {"name": self.name, **self.kwargs}))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", {"name": self.name, "exc_type": exc_type}))
            return False

        def update(self, **kwargs):
            calls.append(("update", {"name": self.name, **kwargs}))

    class FakeClient:
        def start_as_current_observation(self, **kwargs):
            calls.append(("start", kwargs))
            return FakeObservation(kwargs["name"], kwargs)

    monkeypatch.setattr("recalld.tracing._get_client", lambda: FakeClient())
    monkeypatch.setattr("recalld.pipeline.runner.ensure_loaded_context_length", AsyncMock(return_value=1000))
    monkeypatch.setattr("recalld.pipeline.runner.postprocess", AsyncMock(return_value=result))
    monkeypatch.setattr("recalld.pipeline.runner._infer_note_title_with_llm", AsyncMock(return_value="2026-04-29 Project Alpha Meeting.md"))
    with patch("recalld.pipeline.runner.VaultWriter") as MockWriter:
        writer = MockWriter.return_value
        writer.read_note = AsyncMock(return_value="")
        writer.write_note = AsyncMock()
        writer.append_to_note = AsyncMock()
        writer.note_exists = AsyncMock(return_value=False)
        await run_pipeline(job, scratch / "session.m4a", cfg)

    stage_names = [entry[1]["name"] for entry in calls if entry[0] == "start"]
    assert stage_names[0] == "conversation-processing"
    assert "generate-session-summary" in stage_names


@pytest.mark.asyncio
async def test_pipeline_waits_for_speaker_confirmation_after_align(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    scratch = tmp_path / job.id
    transcript_path = scratch / "transcript.json"
    diarisation_path = scratch / "diarisation.json"

    words = [
        WordSegment(start=0.0, end=0.5, word="Hey,"),
        WordSegment(start=0.5, end=1.0, word="did"),
        WordSegment(start=1.0, end=1.5, word="you"),
        WordSegment(start=1.5, end=2.0, word="finish"),
        WordSegment(start=2.0, end=2.5, word="the"),
        WordSegment(start=2.5, end=3.0, word="report?"),
        WordSegment(start=3.0, end=3.5, word="Not"),
        WordSegment(start=3.5, end=4.0, word="yet,"),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=3.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=3.0, end=5.0, speaker="SPEAKER_01"),
    ]
    transcript_path.write_text(json.dumps([w.__dict__ for w in words]))
    diarisation_path.write_text(json.dumps([t.__dict__ for t in turns]))

    job.transcript_path = str(transcript_path)
    job.diarisation_path = str(diarisation_path)
    job.current_stage = JobStage.align
    save_job(job, scratch_root=tmp_path)

    cfg = Config(
        llm_model="test-model",
        categories=[Category(
            id="cat-1",
            name="Planning",
            vault_path="Notes/Sessions",
            speaker_a="Speaker A",
            speaker_b="Speaker B",
        )],
    )

    with patch("recalld.pipeline.runner.postprocess", AsyncMock()) as mock_postprocess:
        await run_pipeline(job, scratch / "session.m4a", cfg)

    mock_postprocess.assert_not_called()
    reloaded = load_job(job.id, scratch_root=tmp_path)
    assert reloaded.current_stage == JobStage.align
    assert reloaded.status == JobStatus.pending
    assert reloaded.stage_statuses["align"] == "awaiting_confirmation"
    assert reloaded.aligned_path is not None


@pytest.mark.asyncio
async def test_pipeline_maps_diariser_speakers_in_encounter_order(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    scratch = tmp_path / job.id
    transcript_path = scratch / "transcript.json"
    diarisation_path = scratch / "diarisation.json"

    words = [
        WordSegment(start=0.0, end=0.5, word="Hello"),
        WordSegment(start=0.5, end=1.0, word="there"),
        WordSegment(start=1.0, end=1.5, word="Hi"),
        WordSegment(start=1.5, end=2.0, word="back"),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_01"),
        SpeakerTurn(start=1.0, end=2.0, speaker="SPEAKER_02"),
    ]
    transcript_path.write_text(json.dumps([w.__dict__ for w in words]))
    diarisation_path.write_text(json.dumps([t.__dict__ for t in turns]))

    job.transcript_path = str(transcript_path)
    job.diarisation_path = str(diarisation_path)
    job.current_stage = JobStage.align
    save_job(job, scratch_root=tmp_path)

    cfg = Config(
        llm_model="test-model",
        categories=[Category(
            id="cat-1",
            name="Planning",
            vault_path="Notes/Sessions",
            speaker_a="You",
            speaker_b="Facilitator",
        )],
    )

    with patch("recalld.pipeline.runner.postprocess", AsyncMock()):
        await run_pipeline(job, scratch / "session.m4a", cfg)

    aligned = json.loads((scratch / "aligned.json").read_text())
    assert aligned[0]["speaker"] == "You"
    assert aligned[1]["speaker"] == "Facilitator"


@pytest.mark.asyncio
async def test_pipeline_collapses_extra_diariser_speakers_into_two_named_speakers(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    scratch = tmp_path / job.id
    transcript_path = scratch / "transcript.json"
    diarisation_path = scratch / "diarisation.json"

    words = [
        WordSegment(start=0.0, end=0.5, word="Hello"),
        WordSegment(start=0.5, end=1.0, word="there"),
        WordSegment(start=1.0, end=1.5, word="How"),
        WordSegment(start=1.5, end=2.0, word="are"),
        WordSegment(start=2.0, end=2.5, word="you"),
    ]
    turns = [
        SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_01"),
        SpeakerTurn(start=1.0, end=2.0, speaker="SPEAKER_02"),
        SpeakerTurn(start=2.0, end=3.0, speaker="SPEAKER_03"),
    ]
    transcript_path.write_text(json.dumps([w.__dict__ for w in words]))
    diarisation_path.write_text(json.dumps([t.__dict__ for t in turns]))

    job.transcript_path = str(transcript_path)
    job.diarisation_path = str(diarisation_path)
    job.current_stage = JobStage.align
    save_job(job, scratch_root=tmp_path)

    cfg = Config(
        llm_model="test-model",
        categories=[Category(
            id="cat-1",
            name="Planning",
            vault_path="Notes/Sessions",
            speaker_a="You",
            speaker_b="Facilitator",
        )],
    )

    with patch("recalld.pipeline.runner.postprocess", AsyncMock()):
        await run_pipeline(job, scratch / "session.m4a", cfg)

    aligned = json.loads((scratch / "aligned.json").read_text())
    assert {turn["speaker"] for turn in aligned} <= {"You", "Facilitator"}
    assert "SPEAKER_03" not in {turn["speaker"] for turn in aligned}


@pytest.mark.asyncio
async def test_pipeline_falls_back_to_default_name_when_inference_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    scratch = tmp_path / job.id
    aligned_path = scratch / "aligned.json"
    aligned_path.write_text(json.dumps([
        LabelledTurn(speaker="You", start=0.0, end=1.0, text="Hello").__dict__,
    ]))
    job.aligned_path = str(aligned_path)
    job.current_stage = JobStage.postprocess
    save_job(job, scratch_root=tmp_path)

    cfg = Config(
        llm_model="test-model",
        categories=[Category(id="cat-1", name="Planning", vault_path="Notes/Sessions")],
    )
    result = PostProcessResult(
        summary="Summary",
        focus_points=["Follow up"],
        raw_response="",
        strategy="single",
        topic_count=1,
    )

    with patch("recalld.pipeline.runner.ensure_loaded_context_length", AsyncMock(return_value=1000)), \
         patch("recalld.pipeline.runner.postprocess", AsyncMock(return_value=result)), \
         patch("recalld.pipeline.runner._infer_note_title_with_llm", AsyncMock(side_effect=Exception("boom"))), \
         patch("recalld.pipeline.runner.VaultWriter"):
        await run_pipeline(job, scratch / "session.m4a", cfg)

    reloaded = load_job(job.id, scratch_root=tmp_path)
    assert reloaded.filename.endswith("Planning.md")


@pytest.mark.asyncio
async def test_pipeline_emits_diarise_progress_messages(tmp_path, monkeypatch):
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    job.wav_path = str(tmp_path / job.id / "session.wav")
    job.current_stage = JobStage.diarise
    save_job(job, scratch_root=tmp_path)

    cfg = Config(
        llm_model="test-model",
        categories=[Category(id="cat-1", name="Planning", vault_path="Notes/Sessions")],
    )

    def fake_diarise(wav_path, token, progress_cb=None):
        assert callable(progress_cb)
        progress_cb("Loading diarisation model.")
        progress_cb("Planned 3 diarisation chunk(s) for 5.0s of audio in 0.0s.")
        raise DiariseError("boom")

    with patch("recalld.pipeline.runner.diarise", side_effect=fake_diarise), \
         patch("recalld.pipeline.runner.bus.publish") as mock_publish:
        await run_pipeline(job, tmp_path / job.id / "session.m4a", cfg)

    published_messages = [call.args[1].get("message", "") for call in mock_publish.call_args_list]
    assert "Loading diarisation model." in published_messages
    assert any(msg.startswith("Planned 3 diarisation chunk(s)") for msg in published_messages)
    assert any(call.args[1].get("status") == "failed" and call.args[1].get("stage") == "diarise" for call in mock_publish.call_args_list)


@pytest.mark.asyncio
async def test_pipeline_marks_job_pending_on_cancellation(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setattr("recalld.pipeline.runner.DEFAULT_SCRATCH_ROOT", tmp_path)

    job = create_job(category_id="cat-1", original_filename="session.m4a", scratch_root=tmp_path)
    job.status = JobStatus.running
    job.current_stage = JobStage.ingest
    save_job(job, scratch_root=tmp_path)

    cfg = Config(llm_model="test-model", categories=[])

    async def _raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    with patch("recalld.pipeline.runner.asyncio.to_thread", side_effect=_raise_cancelled):
        await run_pipeline(job, tmp_path / job.id / "session.m4a", cfg)

    reloaded = load_job(job.id, scratch_root=tmp_path)
    assert reloaded.status == JobStatus.pending
