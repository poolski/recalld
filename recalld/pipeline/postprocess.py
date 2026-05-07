from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from recalld.llm.context import estimate_tokens
from recalld.llm.chunking import chunk_transcript, ChunkStrategy
from recalld.llm.client import LLMClient, complete_with_prompt, stream_with_prompt
from recalld.llm.prompts import resolve_text_prompt
from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.themes import ThemeSuggestion


SYSTEM_PROMPT_TEMPLATE = """\
You are a session notes assistant. Given a transcript, produce:
1. A summary under the heading "## Summary".
   - Refer to {speaker_a_name} as "you" and {speaker_b_name} by name.
   - Prioritize concrete facts and discussion details over narrative filler.
   - Cover the discussion in detail, proportional to transcript depth and duration.
   - For a blank slate, organize the summary into a few thematic sections rather than a single chronological block.
   - Group related ideas together even when they are not chronologically adjacent.
   - Extract specific topics discussed, concrete suggestions made, decisions reached, and open questions.
   - Include only what is stated in the transcript; do not infer facts or inject opinions.
   - Follow the provided style profile closely for wording, cadence, and register.
   - The style profile controls phrasing only; it must not change or add facts.
   - Use a direct, pragmatic, no-fluff style.
   - If style guidance conflicts with transcript fidelity, transcript fidelity wins.
   - If existing note content is provided, preserve its headings, sections, and in-progress thoughts.
   - Continue existing sections where relevant instead of flattening or replacing the document structure.
   - Preserve the existing headings, but add new headings when the transcript introduces distinct themes or when a split would improve clarity.
   - Preserve the overall markdown structure, including heading order and section boundaries.
   - You may expand or reword text within a section inline, but do not flatten or reorder the note.
   - If the transcript introduces a distinct topic or subtopic, add a new heading for it when helpful.
   - Existing headings should remain, but they do not need to stay verbatim if the transcript adds detail.
   - Preserve links, embeds, and link targets in existing note content.
   - If surrounding text is rewritten inline, keep links and embeds in a context that still makes sense.
   - Add other relevant discussion points under their own headings when the existing note does not already cover them.
   - Expand numbered and bulleted lists when the transcript supports more detail; do not preserve lists verbatim just because they already exist.
   - Separate distinct themes into their own paragraphs.
   - Paragraphs must be separated by blank lines.
   - Do not use any other headings or formatting.
2. A short list of focus points or action items under "## Focus" using markdown checkboxes (- [ ] item)
   - Include only actions or follow-ups grounded in the transcript.
   - Use an instructive, actionable tone for focus points, without being overly formal or verbose.
Write clearly. Do not add extra headings, advice, or commentary outside transcript-grounded summarization."""

MAP_SYSTEM_PROMPT = """\
You are summarising one section of a longer session transcript.
Extract concrete details: topics discussed, suggestions made, decisions, and open follow-ups.
Use only transcript-grounded facts, with no opinions or invented details.
Use a direct, pragmatic, no-fluff style.
Align phrasing with the provided style profile when present.
Style controls wording only; do not add or alter facts.
Format: plain prose, no headings."""

REDUCE_SYSTEM_PROMPT_TEMPLATE = """\
You are combining partial summaries of a coaching session transcript into final notes.
Produce:
1. A summary under "## Summary".
   - Refer to {speaker_a_name} as "you" and {speaker_b_name} by name.
   - Prioritize concrete facts and discussion details over narrative filler.
   - Preserve detailed factual coverage from the partial summaries.
   - For a blank slate, organize the summary into a few thematic sections rather than a single chronological block.
   - Group related ideas together even when they are not chronologically adjacent.
   - Extract specific topics, concrete suggestions, decisions, and open follow-ups.
   - Include only what is evidenced in the summaries; do not add opinions or inferred facts.
   - Use a direct, pragmatic, no-fluff style.
   - Follow the provided style profile closely for wording, cadence, and register.
   - The style profile controls phrasing only; it must not change or add facts.
   - If style guidance conflicts with factual fidelity, factual fidelity wins.
   - If existing note content is provided, preserve its headings, sections, and in-progress thoughts.
   - Continue existing sections where relevant instead of flattening or replacing the document structure.
   - Preserve the existing headings, but add new headings when the transcript introduces distinct themes or when a split would improve clarity.
   - Preserve the overall markdown structure, including heading order and section boundaries.
   - You may expand or reword text within a section inline, but do not flatten or reorder the note.
   - If the transcript introduces a distinct topic or subtopic, add a new heading for it when helpful.
   - Existing headings should remain, but they do not need to stay verbatim if the transcript adds detail.
   - Preserve links, embeds, and link targets in existing note content.
   - If surrounding text is rewritten inline, keep links and embeds in a context that still makes sense.
   - Add other relevant discussion points under their own headings when the existing note does not already cover them.
   - Expand numbered and bulleted lists when the transcript supports more detail; do not preserve lists verbatim just because they already exist.
   - Separate distinct themes into their own paragraphs.
   - Paragraphs MUST be separated by blank lines.
   - Do not use any other headings or formatting.
2. A focused list of action items under "## Focus" using markdown checkboxes (- [ ] item), grounded only in the summaries."""

STYLE_ANALYSIS_SYSTEM_PROMPT = """\
You extract writing style characteristics from a transcript sample between two people, {speaker_a_name} and {speaker_b_name}.
Use only one speaker's turns from the sample; do not blend both speakers into a single style profile.
Return 3-5 short bullet points describing voice/style only:
- directness and tone
- sentence length and pacing
- vocabulary/register
- preference for concrete language vs abstract language
Do not include transcript facts, names, topics, or action items.
Keep bullets concise and implementation-friendly."""

DEFAULT_STYLE_PROFILE = """\
- Use direct, plain language.
- Prefer concrete wording over abstract framing.
- Keep sentences concise but complete.
- Avoid filler, hype, and editorial commentary."""

STYLE_ANALYSIS_PROMPT_NAME = "recalld/postprocess-style-analysis"
SUMMARY_SINGLE_PROMPT_NAME = "recalld/postprocess-summary-single"
SUMMARY_REDUCE_PROMPT_NAME = "recalld/postprocess-summary-reduce"
SUMMARY_MAP_PROMPT_NAME = "recalld/postprocess-summary-map"


@dataclass
class PostProcessResult:
    summary: str
    focus_points: list[str]
    raw_response: str
    strategy: str
    topic_count: int


def parse_focus_points(markdown: str) -> list[str]:
    """Extract focus point text from '- [ ] ...' lines."""
    pattern = re.compile(r"^- \[ \] (.+)$", re.MULTILINE)
    return [m.group(1).strip() for m in pattern.finditer(markdown)]


def parse_summary(markdown: str) -> str:
    """Extract content after ## Summary, excluding the focus section if present."""
    marker = "## Summary"
    if marker in markdown:
        parts = markdown.split(marker, 1)
        content = parts[1].lstrip()
        focus_heading = re.search(r"^##\s+Focus\b", content, re.MULTILINE)
        if focus_heading:
            return content[:focus_heading.start()].strip()
        return content.strip()
    return markdown.strip()


def _turns_to_text(turns: list[LabelledTurn]) -> str:
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


def _apply_prompt_variables(text: str, *, speaker_a_name: str, speaker_b_name: str) -> str:
    return (
        text.replace("{speaker_a_name}", speaker_a_name)
        .replace("{speaker_b_name}", speaker_b_name)
    )


def _single_system_prompt(speaker_a_name: str, speaker_b_name: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        speaker_a_name=speaker_a_name, speaker_b_name=speaker_b_name
    )


def _reduce_system_prompt(speaker_a_name: str, speaker_b_name: str) -> str:
    return REDUCE_SYSTEM_PROMPT_TEMPLATE.format(
        speaker_a_name=speaker_a_name, speaker_b_name=speaker_b_name
    )


def _extract_markdown_headings(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    ]


def _find_followup_heading(text: str) -> str | None:
    followup_markers = (
        "focus",
        "for next time",
        "next steps",
        "next step",
        "next time",
        "action items",
        "follow-up",
        "follow up",
        "to do",
        "todo",
    )
    for heading in _extract_markdown_headings(text):
        normalized = heading.lower()
        if any(marker in normalized for marker in followup_markers):
            return heading
    return None


def _style_sample_turns(turns: list[LabelledTurn], speaker_a_name: str) -> list[LabelledTurn]:
    preferred_label = speaker_a_name.strip().lower()
    preferred_turns = [t for t in turns if t.speaker.strip().lower() == preferred_label]
    if preferred_turns:
        return preferred_turns
    if not turns:
        return []

    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for index, turn in enumerate(turns):
        label = turn.speaker.strip().lower()
        counts[label] = counts.get(label, 0) + 1
        if label not in first_seen:
            first_seen[label] = index

    chosen_label = max(counts, key=lambda label: (counts[label], -first_seen[label]))
    return [t for t in turns if t.speaker.strip().lower() == chosen_label]


def _sample_style_window(
    turns: list[LabelledTurn], speaker_a_name: str, seconds: float = 60.0
) -> str:
    source = _style_sample_turns(turns, speaker_a_name)
    if not source:
        return ""
    start = source[0].start
    selected: list[LabelledTurn] = []
    for t in source:
        if t.end - start > seconds:
            break
        selected.append(t)
    return "\n".join(f"{t.speaker}: {t.text}" for t in selected[:80])


def _sanitize_style_profile(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return DEFAULT_STYLE_PROFILE
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    bullet_lines = []
    for line in lines:
        if line.startswith("- "):
            bullet_lines.append(line)
        else:
            bullet_lines.append(f"- {line.lstrip('-* ')}")
        if len(bullet_lines) >= 5:
            break
    return "\n".join(bullet_lines) if bullet_lines else DEFAULT_STYLE_PROFILE


async def _build_style_profile(
    client: LLMClient,
    turns: list[LabelledTurn],
    speaker_a_name: str,
    speaker_b_name: str,
    prompt_label: str | None = None,
) -> str:
    sample = _sample_style_window(turns, speaker_a_name=speaker_a_name, seconds=60.0)
    if not sample or estimate_tokens(sample) < 30:
        return DEFAULT_STYLE_PROFILE
    prompt = resolve_text_prompt(
        STYLE_ANALYSIS_PROMPT_NAME,
        STYLE_ANALYSIS_SYSTEM_PROMPT,
        prompt_label=prompt_label,
        speaker_a_name=speaker_a_name,
        speaker_b_name=speaker_b_name,
    )
    user = (
        "Transcript style sample (about one minute):\n"
        f"{sample}\n\n"
        "Return only style bullets."
    )
    try:
        raw = await complete_with_prompt(
            client,
            _apply_prompt_variables(
                prompt.text,
                speaker_a_name=speaker_a_name,
                speaker_b_name=speaker_b_name,
            ),
            user,
            prompt=prompt.prompt,
            metadata=prompt.metadata,
        )
    except Exception:
        return DEFAULT_STYLE_PROFILE
    return _sanitize_style_profile(raw)


def _effective_transcript_budget(token_budget: int, *prompts: str) -> int:
    """Reserve room for prompts and generation so chunking is conservative."""
    prompt_tokens = max((estimate_tokens(prompt) for prompt in prompts), default=0)
    safety_reserve = 512
    return max(1, token_budget - prompt_tokens - safety_reserve)


def _note_scaffold_block(existing_note_content: str) -> str:
    scaffold = (existing_note_content or "").strip()
    if not scaffold:
        return ""
    followup_heading = _find_followup_heading(scaffold)
    followup_block = ""
    if followup_heading:
        followup_block = (
            "\n\nExisting follow-up-style heading detected:\n"
            f"## {followup_heading}\n"
            "Expand it with more detail and transcript-backed action items instead of "
            "creating a separate ## Focus heading."
        )
    return (
        "\n\nExisting note content to expand:\n"
        f"{scaffold}\n\n"
        "Expand shorthand sections with transcript-grounded detail.\n"
        "Preserve the note's overview and retain relevant existing headings.\n"
        "Preserve existing headings, sections, and in-progress thoughts.\n"
        "Preserve the overall markdown structure, including heading order and section boundaries.\n"
        "You may expand or reword text within a section inline, but do not flatten or reorder the note.\n"
        "Existing links, embeds, and link targets are authoritative. Do not remove or rewrite them.\n"
        "If you expand a sentence that already contains a link or embed, keep the exact link or embed text intact.\n"
        "Preserve links, embeds, and link targets in existing note content.\n"
        "If surrounding text is rewritten inline, keep links and embeds in a context that still makes sense.\n"
        "Continue existing sections where relevant instead of flattening the document.\n"
        "Add transcript-backed details and new relevant subjects where helpful."
        f"{followup_block}"
    )


def _theme_guidance_block(theme_guidance: list[ThemeSuggestion | dict | str] | None) -> str:
    if not theme_guidance:
        return ""

    lines: list[str] = []
    for index, theme in enumerate(theme_guidance, start=1):
        if isinstance(theme, ThemeSuggestion):
            title = theme.title.strip()
            notes = theme.notes.strip()
            enabled = theme.enabled
        elif isinstance(theme, dict):
            title = str(theme.get("title", "")).strip()
            notes = str(theme.get("notes", "")).strip()
            enabled = bool(theme.get("enabled", True))
        else:
            title = str(theme).strip()
            notes = ""
            enabled = True

        if not title:
            continue
        if not enabled:
            continue
        detail = f" — {notes}" if notes else ""
        lines.append(f"{len(lines) + 1}. {title}{detail}")

    if not lines:
        return ""

    return (
        "\n\nConfirmed theme guidance:\n"
        + "\n".join(lines)
        + "\n"
        "Confirmed theme guidance is the primary organizational guidance for the summary.\n"
        "Use enabled themes as the organizing backbone for the summary when they fit the transcript.\n"
        "Theme guidance must not remove, rewrite, or restyle existing links, embeds, or link targets from the note scaffold.\n"
    )


async def postprocess(
    turns: list[LabelledTurn],
    llm_base_url: str,
    llm_model: str,
    token_budget: int,
    progress_cb: Optional[Callable[[str], None]] = None,
    stream_cb: Optional[Callable[[str], None]] = None,
    event_cb: Optional[Callable[[str, dict], None]] = None,
    speaker_a_name: str = "You",
    speaker_b_name: str = "Speaker 2",
    existing_note_content: str = "",
    theme_guidance: list[ThemeSuggestion | dict | str] | None = None,
    prompt_label: str | None = None,
) -> PostProcessResult:
    client = LLMClient(base_url=llm_base_url, model=llm_model)
    if progress_cb:
        progress_cb("Detecting style from transcript sample.")
    style_profile = await _build_style_profile(
        client,
        turns=turns,
        speaker_a_name=speaker_a_name,
        speaker_b_name=speaker_b_name,
        prompt_label=prompt_label,
    )
    style_block = f"\n\nStyle profile (from transcript sample):\n{style_profile}\n"
    theme_block = _theme_guidance_block(theme_guidance)
    scaffold_block = _note_scaffold_block(existing_note_content)
    single_prompt_ref = resolve_text_prompt(
        SUMMARY_SINGLE_PROMPT_NAME,
        SYSTEM_PROMPT_TEMPLATE,
        prompt_label=prompt_label,
        speaker_a_name=speaker_a_name,
        speaker_b_name=speaker_b_name,
    )
    reduce_prompt_ref = resolve_text_prompt(
        SUMMARY_REDUCE_PROMPT_NAME,
        REDUCE_SYSTEM_PROMPT_TEMPLATE,
        prompt_label=prompt_label,
        speaker_a_name=speaker_a_name,
        speaker_b_name=speaker_b_name,
    )
    map_prompt_ref = resolve_text_prompt(
        SUMMARY_MAP_PROMPT_NAME,
        MAP_SYSTEM_PROMPT,
        prompt_label=prompt_label,
    )
    single_prompt = (
        _apply_prompt_variables(
            single_prompt_ref.text,
            speaker_a_name=speaker_a_name,
            speaker_b_name=speaker_b_name,
        )
        + style_block
        + theme_block
        + scaffold_block
    )
    reduce_prompt = (
        _apply_prompt_variables(
            reduce_prompt_ref.text,
            speaker_a_name=speaker_a_name,
            speaker_b_name=speaker_b_name,
        )
        + style_block
        + theme_block
        + scaffold_block
    )
    if progress_cb:
        progress_cb("Calculating context budget for summarization.")
    effective_budget = _effective_transcript_budget(
        token_budget, single_prompt, reduce_prompt
    )
    if progress_cb:
        progress_cb("Selecting summarization strategy.")
    strategy = chunk_transcript(turns, token_budget=effective_budget)

    if strategy.strategy == "single":
        if progress_cb:
            progress_cb(
                "Chunking strategy: single-pass summary (no transcript chunk splitting)."
        )
        transcript_text = _turns_to_text(turns)
        raw = ""
        async for token in stream_with_prompt(
            client,
            single_prompt,
            transcript_text,
            event_cb=event_cb,
            prompt=single_prompt_ref.prompt,
            metadata=single_prompt_ref.metadata,
        ):
            raw += token
            if stream_cb:
                stream_cb(parse_summary(raw))

        return PostProcessResult(
            summary=parse_summary(raw),
            focus_points=parse_focus_points(raw),
            raw_response=raw,
            strategy="single",
            topic_count=strategy.topic_count,
        )

    # Map phase
    if progress_cb:
        progress_cb(
            f"Chunking strategy: map-reduce across {len(strategy.chunks)} chunks "
            f"(topic count: {strategy.topic_count})."
        )

    partial_summaries = []
    for i, chunk in enumerate(strategy.chunks):
        chunk_text = _turns_to_text(chunk)
        if progress_cb:
            progress_cb(f"Summarising chunk {i + 1}/{len(strategy.chunks)}.")
        partial = await complete_with_prompt(
            client,
            map_prompt_ref.text,
            chunk_text,
            prompt=map_prompt_ref.prompt,
            metadata=map_prompt_ref.metadata,
        )
        partial_summaries.append(partial)
        if progress_cb:
            progress_cb(f"Completed chunk {i + 1}/{len(strategy.chunks)}.")

    # Reduce phase
    if progress_cb:
        progress_cb("Reducing chunk summaries into final transcript summary.")
    combined = "\n\n---\n\n".join(partial_summaries)
    raw = ""
    async for token in stream_with_prompt(
        client,
        reduce_prompt,
        combined,
        event_cb=event_cb,
        prompt=reduce_prompt_ref.prompt,
        metadata=reduce_prompt_ref.metadata,
    ):
        raw += token
        if stream_cb:
            stream_cb(parse_summary(raw))

    return PostProcessResult(
        summary=parse_summary(raw),
        focus_points=parse_focus_points(raw),
        raw_response=raw,
        strategy="map_reduce",
        topic_count=strategy.topic_count,
    )
