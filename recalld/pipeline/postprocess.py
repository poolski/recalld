from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from recalld.llm.context import estimate_tokens
from recalld.llm.chunking import chunk_transcript, ChunkStrategy
from recalld.llm.client import LLMClient
from recalld.pipeline.align import LabelledTurn


SYSTEM_PROMPT_TEMPLATE = """\
You are a session notes assistant. Given a transcript, produce:
1. A summary under the heading "## Summary".
   - Refer to {speaker_a_name} as "you" and {speaker_b_name} by name.
   - Prioritize concrete facts and discussion details over narrative filler.
   - Cover the discussion in detail, proportional to transcript depth and duration.
   - Extract specific topics discussed, concrete suggestions made, decisions reached, and open questions.
   - Include only what is stated in the transcript; do not infer facts or inject opinions.
   - Follow the provided style profile closely for wording, cadence, and register.
   - The style profile controls phrasing only; it must not change or add facts.
   - If style guidance conflicts with transcript fidelity, transcript fidelity wins.
   - Separate distinct themes into their own paragraphs.
   - Paragraphs must be separated by blank lines.
   - Do not use any other headings or formatting.
2. A short list of focus points or action items under "## Focus" using markdown checkboxes (- [ ] item)
   - Include only actions or follow-ups grounded in the transcript.
Write clearly. Do not add extra headings, advice, or commentary outside transcript-grounded summarization."""

MAP_SYSTEM_PROMPT = """\
You are summarising one section of a longer session transcript.
Extract concrete details: topics discussed, suggestions made, decisions, and open follow-ups.
Use only transcript-grounded facts, with no opinions or invented details.
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
   - Extract specific topics, concrete suggestions, decisions, and open follow-ups.
   - Include only what is evidenced in the summaries; do not add opinions or inferred facts.
   - Follow the provided style profile closely for wording, cadence, and register.
   - The style profile controls phrasing only; it must not change or add facts.
   - If style guidance conflicts with factual fidelity, factual fidelity wins.
   - Separate distinct themes into their own paragraphs.
   - Paragraphs MUST be separated by blank lines.
   - Do not use any other headings or formatting.
2. A focused list of action items under "## Focus" using markdown checkboxes (- [ ] item), grounded only in the summaries."""

STYLE_ANALYSIS_SYSTEM_PROMPT = """\
You extract writing style characteristics from a transcript sample.
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
    """Extract content after ## Summary, excluding following headings if present."""
    marker = "## Summary"
    if marker in markdown:
        parts = markdown.split(marker, 1)
        content = parts[1].lstrip()
        # Look for next heading to stop at
        next_heading = content.find("\n##")
        if next_heading != -1:
            return content[:next_heading].strip()
        return content.strip()
    return markdown.strip()


def _turns_to_text(turns: list[LabelledTurn]) -> str:
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


def _single_system_prompt(speaker_a_name: str, speaker_b_name: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(speaker_a_name=speaker_a_name, speaker_b_name=speaker_b_name)


def _reduce_system_prompt(speaker_a_name: str, speaker_b_name: str) -> str:
    return REDUCE_SYSTEM_PROMPT_TEMPLATE.format(speaker_a_name=speaker_a_name, speaker_b_name=speaker_b_name)


def _sample_style_window(turns: list[LabelledTurn], speaker_a_name: str, seconds: float = 60.0) -> str:
    speaker_turns = [t for t in turns if t.speaker.strip().lower() == speaker_a_name.strip().lower()]
    source = speaker_turns if speaker_turns else turns
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


async def _build_style_profile(client: LLMClient, turns: list[LabelledTurn], speaker_a_name: str) -> str:
    sample = _sample_style_window(turns, speaker_a_name=speaker_a_name, seconds=60.0)
    if not sample:
        return DEFAULT_STYLE_PROFILE
    user = (
        "Transcript style sample (about one minute):\n"
        f"{sample}\n\n"
        "Return only style bullets."
    )
    try:
        raw = await client.complete(STYLE_ANALYSIS_SYSTEM_PROMPT, user)
    except Exception:
        return DEFAULT_STYLE_PROFILE
    return _sanitize_style_profile(raw)


def _effective_transcript_budget(token_budget: int, *prompts: str) -> int:
    """Reserve room for prompts and generation so chunking is conservative."""
    prompt_tokens = max((estimate_tokens(prompt) for prompt in prompts), default=0)
    safety_reserve = 512
    return max(1, token_budget - prompt_tokens - safety_reserve)


async def postprocess(
    turns: list[LabelledTurn],
    llm_base_url: str,
    llm_model: str,
    token_budget: int,
    progress_cb: Optional[Callable[[str], None]] = None,
    stream_cb: Optional[Callable[[str], None]] = None,
    event_cb: Optional[Callable[[str, dict], None]] = None,
    speaker_a_name: str = "You",
    speaker_b_name: str = "Coach",
) -> PostProcessResult:
    client = LLMClient(base_url=llm_base_url, model=llm_model)
    if progress_cb:
        progress_cb("Detecting style from transcript sample.")
    style_profile = await _build_style_profile(client, turns=turns, speaker_a_name=speaker_a_name)
    style_block = f"\n\nStyle profile (from transcript sample):\n{style_profile}\n"
    single_prompt = _single_system_prompt(speaker_a_name, speaker_b_name) + style_block
    reduce_prompt = _reduce_system_prompt(speaker_a_name, speaker_b_name) + style_block
    if progress_cb:
        progress_cb("Calculating context budget for summarization.")
    effective_budget = _effective_transcript_budget(token_budget, single_prompt, reduce_prompt)
    if progress_cb:
        progress_cb("Selecting summarization strategy.")
    strategy = chunk_transcript(turns, token_budget=effective_budget)

    if strategy.strategy == "single":
        if progress_cb:
            progress_cb("Chunking strategy: single-pass summary (no transcript chunk splitting).")
        transcript_text = _turns_to_text(turns)
        raw = ""
        async for token in client.stream(single_prompt, transcript_text, event_cb=event_cb):
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
        partial = await client.complete(MAP_SYSTEM_PROMPT, chunk_text)
        partial_summaries.append(partial)
        if progress_cb:
            progress_cb(f"Completed chunk {i + 1}/{len(strategy.chunks)}.")

    # Reduce phase
    if progress_cb:
        progress_cb("Reducing chunk summaries into final transcript summary.")
    combined = "\n\n---\n\n".join(partial_summaries)
    raw = ""
    async for token in client.stream(reduce_prompt, combined, event_cb=event_cb):
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
