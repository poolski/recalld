from __future__ import annotations

from datetime import date
from typing import Optional
from urllib.parse import quote

import httpx

from recalld.pipeline.align import LabelledTurn
from recalld.pipeline.postprocess import PostProcessResult


def render_session_note(
    session_date: date,
    category: str,
    speakers: list[str],
    result: Optional[PostProcessResult],
    turns: list[LabelledTurn],
) -> str:
    post_processing_status = "failed" if result is None else "ok"
    speakers_yaml = "[" + ", ".join(speakers) + "]"
    date_str = session_date.isoformat()

    transcript_lines = "\n".join(f"> **{t.speaker}:** {t.text}" for t in turns)

    if result is None:
        body = "_Post-processing failed. Transcript preserved below._\n"
        focus_section = ""
    else:
        focus_items = "\n".join(f"- [ ] {p}" for p in result.focus_points)
        body = f"{result.summary}\n\n[Full transcript ↓](#transcript)\n"
        focus_section = f"\n## Focus\n\n{focus_items}\n"

    return f"""---
date: {date_str}
category: {category}
speakers: {speakers_yaml}
post_processing: {post_processing_status}
---

## Summary

{body}{focus_section}
## Transcript

> [!note]- Full transcript
{transcript_lines}
"""


def render_focus_section(session_date: date, focus_points: list[str]) -> str:
    items = "\n".join(f"- [ ] {p}" for p in focus_points)
    return f"\n## {session_date.isoformat()}\n\n{items}\n"


class VaultWriteError(Exception):
    pass


class VaultWriter:
    def __init__(self, api_url: str, api_key: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def write_note(self, vault_path: str, filename: str, content: str) -> None:
        encoded = quote(f"{vault_path}/{filename}")
        url = f"{self.api_url}/vault/{encoded}"
        async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
            resp = await client.post(url, content=content.encode(), headers={
                **self._headers(),
                "Content-Type": "text/markdown",
            })
            if resp.status_code >= 400:
                raise VaultWriteError(f"Obsidian API error {resp.status_code}: {resp.text}")

    async def append_to_note(self, vault_path: str, content: str) -> None:
        encoded = quote(vault_path)
        url = f"{self.api_url}/vault/{encoded}"
        async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
            resp = await client.patch(url, content=content.encode(), headers={
                **self._headers(),
                "Content-Type": "text/markdown",
            })
            if resp.status_code >= 400:
                raise VaultWriteError(f"Obsidian API error {resp.status_code}: {resp.text}")

    async def note_exists(self, vault_path: str) -> bool:
        encoded = quote(vault_path)
        url = f"{self.api_url}/vault/{encoded}"
        async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
            resp = await client.get(url, headers=self._headers())
            return resp.status_code == 200
