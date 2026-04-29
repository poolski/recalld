from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from recalld.llm.chunking import chunk_transcript, ChunkStrategy
from recalld.llm.client import LLMClient
from recalld.pipeline.align import LabelledTurn


SYSTEM_PROMPT_TEMPLATE = """\
You are a session notes assistant. Given a transcript, produce:
1. A summary under the heading "## Summary".
   - Refer to {speaker_a_name} as "you" and {speaker_b_name} by name.
   - Use 3-5 concise paragraphs to keep the text readable.
   - Paragraphs should be at most 3 sentences long.
   - Avoid long, dense blocks of text.
   - Separate distinct themes into their own paragraphs.
   - Do not use any other headings or formatting.
2. A short list of focus points or action items under "## Focus" using markdown checkboxes (- [ ] item)
Write clearly and concisely. Do not add extra headings or commentary."""

MAP_SYSTEM_PROMPT = """\
You are summarising one section of a longer session transcript.
Produce a brief, concise summary of the key themes and any action items mentioned.
Keep it short and avoid dense blocks of text.
Format: plain prose, no headings."""

REDUCE_SYSTEM_PROMPT_TEMPLATE = """\
You are combining partial summaries of a coaching session transcript into final notes.
Produce:
1. A summary under "## Summary".
   - Refer to {speaker_a_name} as "you" and {speaker_b_name} by name.
   - Use 3-5 concise paragraphs to keep the text readable.
   - Separate distinct themes into their own paragraphs.
   - Do not use any other headings or formatting.
2. A focused list of action items under "## Focus" using markdown checkboxes (- [ ] item)"""


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


async def postprocess(
    turns: list[LabelledTurn],
    llm_base_url: str,
    llm_model: str,
    token_budget: int,
    progress_cb: Optional[Callable[[str], None]] = None,
    stream_cb: Optional[Callable[[str], None]] = None,
    speaker_a_name: str = "You",
    speaker_b_name: str = "Coach",
) -> PostProcessResult:
    client = LLMClient(base_url=llm_base_url, model=llm_model)
    strategy = chunk_transcript(turns, token_budget=token_budget)
    single_prompt = _single_system_prompt(speaker_a_name, speaker_b_name)
    reduce_prompt = _reduce_system_prompt(speaker_a_name, speaker_b_name)

    if strategy.strategy == "single":
        transcript_text = _turns_to_text(turns)
        raw = ""
        async for token in client.stream(single_prompt, transcript_text):
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
        progress_cb(f"Transcript is too large for one summary pass; splitting into {len(strategy.chunks)} chunks.")

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
    combined = "\n\n---\n\n".join(partial_summaries)
    raw = ""
    async for token in client.stream(reduce_prompt, combined):
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
