from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from recalld.llm.client import LLMClient, complete_with_prompt
from recalld.llm.chunking import ChunkStrategy, chunk_transcript
from recalld.llm.prompts import resolve_text_prompt
from recalld.pipeline.align import LabelledTurn


SYSTEM_PROMPT = """\
You extract a small set of likely discussion themes or headings from a conversation transcript.
Return only valid JSON, with no markdown, commentary, or surrounding prose.

Preferred shape:
{
  "themes": [
    {"id": "theme-1", "title": "Concise heading", "notes": "Short rationale or cue", "enabled": true}
  ]
}

Rules:
- Produce 3 to 7 themes when the transcript supports that many.
- Titles must be concise and heading-like.
- Notes should be short and transcript-grounded.
- Use only facts, topics, and wording supported by the transcript.
- Do not include action items unless they are clearly central themes.
- Leave themes enabled by default.
"""

MAP_SYSTEM_PROMPT = """\
You extract candidate discussion themes from one section of a longer transcript.
Return only valid JSON using the same shape as the main theme extractor.
Keep the list short and transcript-grounded.
"""

THEMES_SINGLE_PROMPT_NAME = "recalld/themes-single"
THEMES_MAP_PROMPT_NAME = "recalld/themes-map"


@dataclass(eq=True)
class ThemeSuggestion:
    id: str
    title: str
    notes: str = ""
    enabled: bool = True
    order: int = 0
    source: str = "transcript"


@dataclass(eq=True)
class ThemeProposalResult:
    themes: list[ThemeSuggestion]
    raw_response: str
    strategy: str
    topic_count: int


def _turns_to_text(turns: list[LabelledTurn]) -> str:
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns)


def _coerce_theme_payload(payload) -> list[ThemeSuggestion]:
    if isinstance(payload, dict):
        payload = payload.get("themes", [])

    if not isinstance(payload, list):
        return []

    themes: list[ThemeSuggestion] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        notes = str(item.get("notes", "")).strip()
        enabled = item.get("enabled", True)
        source = str(item.get("source", "transcript")).strip() or "transcript"
        theme_id = str(item.get("id", "")).strip() or f"theme-{index}"
        order = item.get("order", index)
        try:
            order = int(order)
        except (TypeError, ValueError):
            order = index
        themes.append(
            ThemeSuggestion(
                id=theme_id,
                title=title,
                notes=notes,
                enabled=bool(enabled),
                order=order,
                source=source,
            )
        )

    themes.sort(key=lambda theme: (theme.order, theme.title.lower(), theme.id))
    for index, theme in enumerate(themes, start=1):
        theme.order = index
    return themes


def _extract_json(raw: str):
    cleaned = (raw or "").strip()
    if not cleaned:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("[")
    start_obj = cleaned.find("{")
    if start == -1 or (0 <= start_obj < start):
        start = start_obj
    if start == -1:
        return None
    end = max(cleaned.rfind("]"), cleaned.rfind("}"))
    if end == -1 or end < start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def _dedupe(themes: list[ThemeSuggestion]) -> list[ThemeSuggestion]:
    seen: set[str] = set()
    deduped: list[ThemeSuggestion] = []
    for theme in themes:
        normalized = re.sub(r"\s+", " ", theme.title.strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(theme)
    for index, theme in enumerate(deduped, start=1):
        theme.order = index
    return deduped


async def propose_themes(
    turns: list[LabelledTurn],
    llm_base_url: str,
    llm_model: str,
    token_budget: int,
    prompt_label: str | None = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ThemeProposalResult:
    client = LLMClient(base_url=llm_base_url, model=llm_model)
    if progress_cb:
        progress_cb("Selecting candidate themes from transcript.")

    strategy = chunk_transcript(turns, token_budget=token_budget)
    if not turns:
        return ThemeProposalResult(themes=[], raw_response="", strategy="single", topic_count=0)

    single_prompt = resolve_text_prompt(
        THEMES_SINGLE_PROMPT_NAME,
        SYSTEM_PROMPT,
        prompt_label=prompt_label,
    )
    map_prompt = resolve_text_prompt(
        THEMES_MAP_PROMPT_NAME,
        MAP_SYSTEM_PROMPT,
        prompt_label=prompt_label,
    )

    if strategy.strategy == "single":
        transcript_text = _turns_to_text(turns)
        raw = await complete_with_prompt(
            client,
            single_prompt.text,
            transcript_text,
            prompt=single_prompt.prompt,
            metadata=single_prompt.metadata,
        )
        themes = _coerce_theme_payload(_extract_json(raw))
        return ThemeProposalResult(
            themes=themes,
            raw_response=raw,
            strategy="single",
            topic_count=len(themes),
        )

    partials: list[str] = []
    all_themes: list[ThemeSuggestion] = []
    for index, chunk in enumerate(strategy.chunks, start=1):
        if progress_cb:
            progress_cb(f"Extracting themes from chunk {index}/{len(strategy.chunks)}.")
        raw = await complete_with_prompt(
            client,
            map_prompt.text,
            _turns_to_text(chunk),
            prompt=map_prompt.prompt,
            metadata=map_prompt.metadata,
        )
        partials.append(raw)
        all_themes.extend(_coerce_theme_payload(_extract_json(raw)))

    merged = _dedupe(all_themes)
    return ThemeProposalResult(
        themes=merged,
        raw_response="\n\n---\n\n".join(partials),
        strategy="map_reduce",
        topic_count=len(merged),
    )
