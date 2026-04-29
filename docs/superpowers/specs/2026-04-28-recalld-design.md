# recalld — Design Spec

**Date:** 2026-04-28  
**Status:** Approved

---

## Overview

`recalld` is a fully local tool for processing audio/video recordings into enriched, structured notes in an Obsidian vault. It transcribes recordings, identifies speakers, detects topic structure, summarises sessions, and extracts focus points — all on-device, with no cloud dependencies.

The initial use case is ADHD coaching sessions recorded via Zoom, but the tool is general-purpose: any recording category can be configured with its own vault path and speaker names.

---

## 1. Architecture

A single local Python service:

- **FastAPI backend** — pipeline orchestration, Obsidian vault writes via Local REST API
- **HTMX + vanilla JS frontend** — served by the same FastAPI process, no build step required
- **Config** — `~/.config/recalld/config.json`
- **Scratch space** — `~/.local/share/recalld/jobs/` for in-progress and completed work

One command starts everything: `uv run recalld` (or `python -m recalld`). No Docker, no separate processes, no build pipeline.

All required directories are created on first run — no manual setup needed.

---

## 2. Pipeline

Each uploaded recording is processed through the following stages in sequence. Stage outputs are persisted to scratch space as each completes, enabling crash recovery and resume from the last incomplete stage.

1. **Ingest** — accept dropped/uploaded file; extract audio to `.wav` via `ffmpeg` if needed (handles `.m4a`, `.mp4`, etc.)
2. **Transcribe** — `whisper.cpp` with Metal backend via `pywhispercpp`; produces word-level timestamped segments
3. **Diarise** — `pyannote.audio` on MPS (Apple Silicon); produces speaker-turn timestamps. The parser accepts both direct `Annotation` output and `DiarizeOutput` wrappers from current pyannote releases, preferring exclusive diarisation when available.
4. **Align** — merge Whisper word segments with pyannote speaker turns to produce a labelled transcript; show a truncated preview (~5 exchanges) in the UI with speaker names applied (or prompt for assignment if not yet saved for this category)
5. **Post-process** — local LLM (OpenAI-compatible endpoint) generates summary + focus points from the confirmed labelled transcript
6. **Vault confirmation** — the UI shows the generated summary and focus points and waits for explicit user confirmation before any vault write occurs
7. **Write to vault** — after confirmation, Obsidian Local REST API creates the session note and appends to the focus note if configured

Each stage reports progress to the UI in real time via SSE. The processing page also rehydrates from the latest persisted job state on load so fast early-stage completions are not missed if the browser attaches after SSE events were emitted.

### Persistence

Each stage writes its output to `~/.local/share/recalld/jobs/<job-id>/` as it completes. Job state (current stage, per-stage status checkpoints, file paths, metadata, category) is written to `job.json` after each stage transition. On startup, `recalld` scans for incomplete jobs and surfaces them in the UI with a "Resume" button. Resuming re-runs from the last incomplete stage only, while previously completed stages are rendered as completed in the UI.

---

## 3. Recording Categories

Configured in the UI, stored in config. Each category has:

- **Name** — e.g. "ADHD Coaching", "Work Meeting"
- **Vault path** — where session notes are created, e.g. `Life/Mental Health/ADHD Coaching/Sessions`
- **Focus note path** — optional; path to a running focus/action note, e.g. `Life/Mental Health/ADHD Coaching/Coaching Focus.md`
- **Speaker names** — two names, persisted per category; prompted in the UI on first use for that category. **v1 constraint: two speakers only.** Multi-speaker support (e.g. work meetings) is out of scope for this version.

Selected at upload time via a dropdown. Defaults to last used category (`last_used_category` in config, updated on each job submission). New categories can be added from the upload screen or the settings page.

---

## 4. Vault Output

### Session note

Created at `<vault path>/YYYY-MM-DD <category name>.md`:

```markdown
---
date: 2025-04-28
category: ADHD Coaching
speakers: [You, Coach]
---

## Summary

<2-3 paragraph summary of the session>

[Full transcript ↓](#transcript)

## Focus

- [ ] Start mornings with 10 minutes of planning before opening email
- [ ] Revisit the "good enough" conversation next session

## Transcript

> [!note]- Full transcript
> **You:** So I've been struggling with...
> **Coach:** That makes sense. What I'm noticing is...
```

- The `## Transcript` callout uses Obsidian's native collapsed callout syntax (`[!note]-`) — collapsed by default, no plugin required
- The `[Full transcript ↓](#transcript)` anchor link at the end of the summary provides quick navigation

### Focus note

If a focus note path is configured for the category, a new dated section is appended:

```markdown
## 2025-04-28

- [ ] Start mornings with 10 minutes of planning before opening email
- [ ] Revisit the "good enough" conversation next session
```

If the focus note does not yet exist, it is created with a top-level heading before the first dated section.

Focus points use standard markdown checkbox syntax (`- [ ]`), compatible with the Obsidian Tasks plugin if installed.

### Incomplete processing

If post-processing fails, the session note can still be written on demand via a transcript-only fallback, marked with `post_processing: failed` in frontmatter. The focus note is not updated in that path.

Session note and focus note are written independently — a failure writing the focus note does not affect the session note.

---

## 5. LLM Context Handling

Long recordings can exceed the context window of smaller local models. `recalld` handles this automatically:

### Context length detection

On startup (and when LLM settings change), `recalld` queries the LLM endpoint for the loaded model's context length via `/v1/models`. Token budget = 80% of detected context length (configurable headroom), leaving room for system prompt and response. If the endpoint is unreachable, falls back to a 6,000 token default. Detected context length is shown in the settings page.

### Topic-based chunking

Before sending to the LLM, `recalld` uses `sentence-transformers` (`all-MiniLM-L6-v2`, local, ~90MB) to detect semantic topic boundaries between speaker turns. When embedding similarity between adjacent turns drops below a threshold, that marks a topic boundary. Each topic segment becomes a natural chunk.

- If the full transcript fits within the token budget: sent in full
- If not: topic-based map-reduce — each chunk summarised independently, combined into final summary + focus points in a second LLM pass
- If a single topic chunk exceeds the budget: split further at the nearest speaker turn boundary
- Token count estimated via heuristic (1 token ≈ 0.75 words)
- Strategy and chunk count surfaced in the UI ("Detected 4 topics, summarising…")

---

## 6. UI

A single-page interface served by FastAPI, built with HTMX and minimal vanilla JS for drag-and-drop. No frontend build step.

### Status bar (persistent, top of every page)

Shows health of all system dependencies, polling every 30 seconds:

| Indicator               | States                                            |
| ----------------------- | ------------------------------------------------- |
| Obsidian Local REST API | Reachable / Unreachable                           |
| LLM endpoint            | Ready (model name + context length) / Unavailable |
| `ffmpeg`                | Found / Missing                                   |
| pyannote model          | Downloaded / Not yet downloaded                   |

Clicking a failed indicator shows a brief explanation and fix hint (e.g. "Start LM Studio and load a model").

### Upload screen

- Large drag-and-drop zone (also supports click-to-browse)
- Category selector dropdown, defaulting to last used, with "Add new category" option
- Recent and incomplete jobs listed below with "Resume" button

### Processing screen

Replaces upload screen when a job starts:

- Stage-by-stage progress: Ingesting → Transcribing → Diarising → Aligning → Post-processing → Vault confirmation → Writing to vault
- Each stage: spinner while running, checkmark when done, explicit status preserved across refresh/resume, and recovery action when failed
- After Align: truncated preview (~5 exchanges) with speaker names applied; "Who is who?" prompt shown if names not yet saved for this category
- During Post-processing: detected topic count and chunk strategy shown if applicable
- Before vault write: generated summary and focus points are shown in-browser and the user must press a confirmation button before any note is written

### Results screen

- Summary and focus points displayed in browser before and after vault write
- Link to open session note in Obsidian via `obsidian://` URI
- Transcript-only vault write fallback offered if post-processing fails

### Debug log panel

Collapsible panel at the bottom of processing and results screens:

- Streams structured log output in real time via SSE — stage transitions, LLM prompts/responses, token counts, pyannote confidence scores, errors
- "Copy to clipboard" and "Save to file" buttons
- Log level configurable in settings (Info / Debug / Verbose)

### Settings page

- Manage categories (name, vault path, focus note path, speaker names)
- Obsidian Local REST API URL + key
- LLM endpoint URL, model name, context headroom percentage
- Detected context length display
- Whisper model size selector
- Log level selector

---

## 7. Configuration

Stored at `~/.config/recalld/config.json`. Created on first run with universal defaults only.

```json
{
  "obsidian_api_url": "https://127.0.0.1:27124",
  "obsidian_api_key": "",
  "llm_base_url": "http://localhost:1234/v1",
  "llm_model": "",
  "llm_context_headroom": 0.8,
  "log_level": "info",
  "last_used_category": null,
  "categories": []
}
```

- `last_used_category` updated automatically on each job submission
- `categories` starts empty — first-run UI prompts creation of at least one category before uploading
- All fields editable via settings UI; no manual JSON editing required
- File is human-readable for backup or version control purposes

---

## 8. Error Handling & Failure Recovery

- **Resume from checkpoints** — completed stage outputs and stage statuses are persisted so resumed jobs continue from the last incomplete stage and still display earlier stages as completed
- **Crash recovery** — on startup, incomplete jobs detected via `job.json` and surfaced with a "Resume" button
- **LLM unavailable at post-processing time** — offer two options: wait and retry, or write transcript-only session note to vault (marked `post_processing: failed` in frontmatter)
- **Vault write confirmation** — successful post-processing does not write immediately; the user confirms before the vault stage runs
- **Vault write failure** — session note content remains reproducible from scratch outputs; the vault stage can be retried without re-running earlier stages
- **pyannote failure** — diarisation parsing is tolerant of current pyannote output variants; if diarisation still fails, the UI offers continuing with an unlabelled transcript (speakers as `SPEAKER_00`, `SPEAKER_01`) so transcript is not lost
- **Partial vault writes** — session note and focus note written and retried independently
- **Scratch space cleanup** — completed jobs remain in scratch space and can be cleaned up later according to retention policy

---

## 9. Technology Stack

### Runtime dependencies

| Package                 | Purpose                                          |
| ----------------------- | ------------------------------------------------ |
| `fastapi`               | Web framework + SSE                              |
| HTMX                    | Frontend interactivity, no build step            |
| `pywhispercpp`          | Python bindings for whisper.cpp (Metal backend)  |
| `pyannote.audio`        | Speaker diarisation (MPS backend)                |
| `sentence-transformers` | Topic boundary detection                         |
| `httpx`                 | Async HTTP client (Obsidian API + LLM calls)     |
| `pydantic`              | Config and job state validation                  |
| `ffmpeg`                | Audio extraction (system dependency, not Python) |

### Models

| Model                | Size   | Notes                                                                    |
| -------------------- | ------ | ------------------------------------------------------------------------ |
| Whisper `small`      | ~500MB | Default; configurable up to `medium` if accuracy needs improving         |
| pyannote diarisation | ~1GB   | Requires HuggingFace token + licence acceptance; downloaded on first run |
| `all-MiniLM-L6-v2`   | ~90MB  | Downloaded automatically via `sentence-transformers`                     |

### Recommended LLM

**Qwen2.5 7B Q4** (~4.5GB) — strong structured markdown output, fits comfortably on M3 Pro with other apps running, fast inference on Metal. Any OpenAI-compatible model works; this is a recommendation not a requirement.

### Package management

`uv` — single `uv run recalld` to start the service.

---

## Open Questions

None — all design decisions resolved during brainstorming session.
