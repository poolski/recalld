from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from recalld.llm.chunking import chunk_transcript, ChunkStrategy
from recalld.llm.client import LLMClient
from recalld.pipeline.align import LabelledTurn


SYSTEM_PROMPT = """\
You are a session notes assistant. Given a transcript, produce:
1. A 2-3 paragraph summary under the heading "## Summary"
2. A short list of focus points or action items under "## Focus" using markdown checkboxes (- [ ] item)
Write clearly and concisely. Do not add extra headings or commentary."""

MAP_SYSTEM_PROMPT = """\
You are summarising one section of a longer session transcript.
Produce a brief summary of the key themes and any action items mentioned.
Format: plain prose, no headings."""

REDUCE_SYSTEM_PROMPT = """\
You are combining partial summaries of a coaching session transcript into final notes.
Produce:
1. A 2-3 paragraph summary under "## Summary"
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
    """Extract content between ## Summary and the next ## heading."""
    match = re.search(r"## Summary\s*\n(.*?)(?=\n##|\Z)", markdown, re.DOTALL)
    if match:
        return match.group(1).strip()
    return markdown.strip()


def _turns_to_text(turns: list[LabelledTurn]) -> str:
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


async def postprocess(
    turns: list[LabelledTurn],
    llm_base_url: str,
    llm_model: str,
    token_budget: int,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> PostProcessResult:
    client = LLMClient(base_url=llm_base_url, model=llm_model)
    strategy = chunk_transcript(turns, token_budget=token_budget)

    if strategy.strategy == "single":
        transcript_text = _turns_to_text(turns)
        raw = await client.complete(SYSTEM_PROMPT, transcript_text)
        return PostProcessResult(
            summary=parse_summary(raw),
            focus_points=parse_focus_points(raw),
            raw_response=raw,
            strategy="single",
            topic_count=1,
        )

    # Map phase
    if progress_cb:
        progress_cb(f"Detected {strategy.topic_count} topics, summarising each…")

    partial_summaries = []
    for i, chunk in enumerate(strategy.chunks):
        chunk_text = _turns_to_text(chunk)
        partial = await client.complete(MAP_SYSTEM_PROMPT, chunk_text)
        partial_summaries.append(partial)
        if progress_cb:
            progress_cb(f"Summarised topic {i + 1}/{strategy.topic_count}")

    # Reduce phase
    combined = "\n\n---\n\n".join(partial_summaries)
    raw = await client.complete(REDUCE_SYSTEM_PROMPT, combined)

    return PostProcessResult(
        summary=parse_summary(raw),
        focus_points=parse_focus_points(raw),
        raw_response=raw,
        strategy="map_reduce",
        topic_count=strategy.topic_count,
    )
