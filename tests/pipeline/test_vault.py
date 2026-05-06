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
        LabelledTurn(speaker="Facilitator", start=5, end=10, text="Let's explore"),
    ]


def test_render_session_note_contains_summary():
    note = render_session_note(
        session_date=date(2025, 4, 28),
        category="Weekly Planning",
        speakers=["You", "Facilitator"],
        result=_result(),
        turns=_turns(),
    )
    assert "A good session." in note
    assert "date: 2025-04-28" in note
    assert "category: Weekly Planning" in note


def test_render_session_note_focus_checkboxes():
    note = render_session_note(
        session_date=date(2025, 4, 28),
        category="Weekly Planning",
        speakers=["You", "Facilitator"],
        result=_result(),
        turns=_turns(),
    )
    assert "- [ ] Do thing one" in note
    assert "- [ ] Do thing two" in note


def test_render_session_note_collapsed_transcript():
    note = render_session_note(
        session_date=date(2025, 4, 28),
        category="Weekly Planning",
        speakers=["You", "Facilitator"],
        result=_result(),
        turns=_turns(),
    )
    assert "> [!note]-" in note
    assert "> **You:**" in note
    assert "> **Facilitator:**" in note


def test_render_session_note_failed_postprocess():
    note = render_session_note(
        session_date=date(2025, 4, 28),
        category="Weekly Planning",
        speakers=["You", "Facilitator"],
        result=None,
        turns=_turns(),
    )
    assert "post_processing: failed" in note


def test_render_session_note_preview_strips_frontmatter_and_truncates():
    note = render_session_note_preview(
        session_date=date(2025, 4, 28),
        category="Weekly Planning",
        speakers=["You", "Facilitator"],
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
    respx.put("https://127.0.0.1:27124/vault/Notes/Sessions/2025-04-28%20Weekly%20Planning.md").mock(
        return_value=httpx.Response(200)
    )
    writer = VaultWriter(api_url="https://127.0.0.1:27124", api_key="token")
    await writer.write_note(
        vault_path="Notes/Sessions",
        filename="2025-04-28 Weekly Planning.md",
        content="# Note",
    )


@respx.mock
@pytest.mark.asyncio
async def test_vault_writer_opens_note_via_rest_api():
    route = respx.post("https://127.0.0.1:27124/open/Notes/Sessions/2025-04-28%20Weekly%20Planning.md").mock(
        return_value=httpx.Response(200)
    )
    writer = VaultWriter(api_url="https://127.0.0.1:27124", api_key="token")
    await writer.open_note("Notes/Sessions/2025-04-28 Weekly Planning.md")

    assert route.called


@respx.mock
@pytest.mark.asyncio
async def test_vault_writer_omits_auth_header_when_api_key_blank():
    route = respx.put("https://127.0.0.1:27124/vault/Notes/Sessions/2025-04-28%20Weekly%20Planning.md").mock(
        return_value=httpx.Response(200)
    )
    writer = VaultWriter(api_url="https://127.0.0.1:27124", api_key="")
    await writer.write_note(
        vault_path="Notes/Sessions",
        filename="2025-04-28 Weekly Planning.md",
        content="# Note",
    )

    assert route.called
    request = route.calls[0].request
    assert "authorization" not in request.headers


@respx.mock
@pytest.mark.asyncio
async def test_vault_writer_reads_note_content():
    route = respx.get("https://127.0.0.1:27124/vault/Notes/Sessions/2025-04-28%20Weekly%20Planning.md").mock(
        return_value=httpx.Response(200, text="# Draft\n- shorthand")
    )
    writer = VaultWriter(api_url="https://127.0.0.1:27124", api_key="token")

    content = await writer.read_note("Notes/Sessions/2025-04-28 Weekly Planning.md")

    assert route.called
    assert content == "# Draft\n- shorthand"


@respx.mock
@pytest.mark.asyncio
async def test_vault_writer_patches_heading_content():
    route = respx.put("https://127.0.0.1:27124/vault/Notes/Sessions/2025-04-28%20Weekly%20Planning.md/heading/Summary").mock(
        return_value=httpx.Response(200)
    )
    writer = VaultWriter(api_url="https://127.0.0.1:27124", api_key="token")

    await writer.patch_heading(
        vault_path="Notes/Sessions/2025-04-28 Weekly Planning.md",
        heading="Summary",
        content="Updated summary text",
    )

    assert route.called
    request = route.calls[0].request
    assert request.content == b"Updated summary text"


@respx.mock
@pytest.mark.asyncio
async def test_vault_writer_appends_to_heading_content():
    route = respx.post("https://127.0.0.1:27124/vault/Notes/Sessions/2025-04-28%20Weekly%20Planning.md/heading/Summary").mock(
        return_value=httpx.Response(200)
    )
    writer = VaultWriter(api_url="https://127.0.0.1:27124", api_key="token")

    await writer.append_to_heading(
        vault_path="Notes/Sessions/2025-04-28 Weekly Planning.md",
        heading="Summary",
        content="Appended summary text",
    )

    assert route.called
    request = route.calls[0].request
    assert request.content == b"Appended summary text"


@respx.mock
@pytest.mark.asyncio
async def test_vault_writer_patches_mixed_heading_levels_through_runner_section_split():
    from recalld.pipeline.runner import _split_rendered_note_sections

    note = "\n".join(
        [
            "---",
            "date: 2025-04-28",
            "---",
            "",
            "# Summary",
            "Top level summary text",
            "",
            "## Details",
            "Nested details text",
            "",
            "### Notes",
            "Deeper notes text",
        ]
    )

    sections = _split_rendered_note_sections(note)

    assert sections == [
        ("Summary", "Top level summary text"),
        ("Details", "Nested details text"),
        ("Notes", "Deeper notes text"),
    ]


@respx.mock
@pytest.mark.asyncio
async def test_vault_writer_lists_directory_contents():
    route = respx.get("https://127.0.0.1:27124/vault/Notes/Sessions/").mock(
        return_value=httpx.Response(200, json={"files": ["2026-05-05 Session.md", "archive/"]})
    )
    writer = VaultWriter(api_url="https://127.0.0.1:27124", api_key="token")

    files = await writer.list_directory("Notes/Sessions")

    assert route.called
    assert files == ["2026-05-05 Session.md", "archive/"]
