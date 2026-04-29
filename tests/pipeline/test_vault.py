import pytest
from datetime import date
from recalld.pipeline.vault import render_session_note, render_session_note_preview, render_focus_section, VaultWriter
from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.postprocess import PostProcessResult
from unittest.mock import AsyncMock, patch
import httpx
import respx


def _result(summary="A good session.", focus=None) -> PostProcessResult:
    return PostProcessResult(
        summary=summary,
        focus_points=focus or ["Do thing one", "Do thing two"],
        raw_response="",
        strategy="single",
        topic_count=1,
    )


def _turns() -> list[LabelledTurn]:
    return [
        LabelledTurn(speaker="You", start=0, end=5, text="I struggle"),
        LabelledTurn(speaker="Coach", start=5, end=10, text="Let's explore"),
    ]


def test_render_session_note_contains_summary():
    note = render_session_note(
        session_date=date(2025, 4, 28),
        category="ADHD Coaching",
        speakers=["You", "Coach"],
        result=_result(),
        turns=_turns(),
    )
    assert "A good session." in note
    assert "date: 2025-04-28" in note
    assert "category: ADHD Coaching" in note


def test_render_session_note_focus_checkboxes():
    note = render_session_note(
        session_date=date(2025, 4, 28),
        category="ADHD Coaching",
        speakers=["You", "Coach"],
        result=_result(),
        turns=_turns(),
    )
    assert "- [ ] Do thing one" in note
    assert "- [ ] Do thing two" in note


def test_render_session_note_collapsed_transcript():
    note = render_session_note(
        session_date=date(2025, 4, 28),
        category="ADHD Coaching",
        speakers=["You", "Coach"],
        result=_result(),
        turns=_turns(),
    )
    assert "> [!note]-" in note
    assert "> **You:**" in note
    assert "> **Coach:**" in note


def test_render_session_note_failed_postprocess():
    note = render_session_note(
        session_date=date(2025, 4, 28),
        category="ADHD Coaching",
        speakers=["You", "Coach"],
        result=None,
        turns=_turns(),
    )
    assert "post_processing: failed" in note


def test_render_session_note_preview_strips_frontmatter_and_truncates():
    note = render_session_note_preview(
        session_date=date(2025, 4, 28),
        category="ADHD Coaching",
        speakers=["You", "Coach"],
        result=_result(summary="A" * 2000),
        turns=_turns(),
        max_chars=200,
    )
    assert "date:" not in note
    assert note.endswith("...")
    assert "## Summary" in note


def test_render_focus_section():
    section = render_focus_section(
        session_date=date(2025, 4, 28),
        focus_points=["Do thing one"],
    )
    assert "## 2025-04-28" in section
    assert "- [ ] Do thing one" in section


@respx.mock
@pytest.mark.asyncio
async def test_vault_writer_creates_note():
    respx.post("https://127.0.0.1:27124/vault/Life/Sessions/2025-04-28%20ADHD%20Coaching.md").mock(
        return_value=httpx.Response(200)
    )
    writer = VaultWriter(api_url="https://127.0.0.1:27124", api_key="token")
    await writer.write_note(
        vault_path="Life/Sessions",
        filename="2025-04-28 ADHD Coaching.md",
        content="# Note",
    )
