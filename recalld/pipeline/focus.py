from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recalld.llm.client import LLMClient, complete_with_prompt
from recalld.llm.context import estimate_tokens
from recalld.llm.prompts import resolve_text_prompt
from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.postprocess import parse_focus_points
from recalld.llm.chunking import chunk_transcript


FOCUS_INSTRUCTIONS_PROMPT_NAME = "recalld/focus-instructions"
FOCUS_INSTRUCTIONS_SYSTEM_PROMPT = """\
You extract the most important follow-up items and action points from a transcript.
Return 3-5 short markdown bullet points only, using - [ ] item.
Each point must be grounded in the transcript and phrased as a concrete action or follow-up.
Do not include summary prose, headings, or commentary.
Use a direct, pragmatic, no-fluff style.
"""


@dataclass(frozen=True)
class FocusResult:
    focus_points: list[str]
    raw_response: str
    strategy: str
    topic_count: int


def _turns_to_text(turns: list[LabelledTurn]) -> str:
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


def _dedupe_focus_points(points: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        normalized = " ".join(point.lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(point.strip())
    return deduped


def _effective_transcript_budget(token_budget: int, *prompts: str) -> int:
    prompt_tokens = max((estimate_tokens(prompt) for prompt in prompts), default=0)
    safety_reserve = 256
    return max(1, token_budget - prompt_tokens - safety_reserve)


async def generate_focus_points(
    *,
    turns: list[LabelledTurn],
    llm_base_url: str,
    llm_model: str,
    token_budget: int,
    speaker_a_name: str = "You",
    speaker_b_name: str = "Speaker 2",
    prompt_label: str | None = None,
) -> FocusResult:
    client = LLMClient(base_url=llm_base_url, model=llm_model)
    prompt = resolve_text_prompt(
        FOCUS_INSTRUCTIONS_PROMPT_NAME,
        FOCUS_INSTRUCTIONS_SYSTEM_PROMPT,
        prompt_label=prompt_label,
        speaker_a_name=speaker_a_name,
        speaker_b_name=speaker_b_name,
    )
    effective_budget = _effective_transcript_budget(token_budget, prompt.text)
    strategy = chunk_transcript(turns, token_budget=effective_budget)
    if not turns:
        return FocusResult(focus_points=[], raw_response="", strategy=strategy.strategy, topic_count=strategy.topic_count)

    if strategy.strategy == "single":
        raw = await complete_with_prompt(
            client,
            prompt.text,
            _turns_to_text(turns),
            prompt=prompt.prompt,
            metadata=prompt.metadata,
        )
        focus_points = _dedupe_focus_points(parse_focus_points(raw))
        return FocusResult(
            focus_points=focus_points,
            raw_response=raw,
            strategy="single",
            topic_count=strategy.topic_count,
        )

    raw_parts: list[str] = []
    points: list[str] = []
    for chunk in strategy.chunks:
        partial = await complete_with_prompt(
            client,
            prompt.text,
            _turns_to_text(chunk),
            prompt=prompt.prompt,
            metadata=prompt.metadata,
        )
        raw_parts.append(partial)
        points.extend(parse_focus_points(partial))

    focus_points = _dedupe_focus_points(points)
    raw = "\n".join(f"- [ ] {point}" for point in focus_points)
    return FocusResult(
        focus_points=focus_points,
        raw_response=raw or "\n\n".join(raw_parts),
        strategy="map_reduce",
        topic_count=strategy.topic_count,
    )
