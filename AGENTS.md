# Agent Guidance

This repository contains `recalld`, a local FastAPI app for processing conversation audio and writing notes to Obsidian.

## Before You Edit

- Read the relevant source files first; do not assume the current behavior from the README alone.
- Prefer small, focused changes over broad rewrites.
- Preserve user changes in unrelated files.
- Use `apply_patch` for manual file edits.

## Project Shape

- Python 3.11+
- App entrypoint: `recalld/__main__.py`
- Web app: `recalld/app.py`
- Config: `recalld/config.py`
- Job state: `recalld/jobs.py`
- Pipeline code: `recalld/pipeline/`
- HTTP routes: `recalld/routers/`
- Templates and static assets: `recalld/templates/`, `recalld/static/`

## Install and Run

- Install dependencies with `uv sync --all-groups`
- Start the app with `make run` or `uv run recalld`
- Open the app at `http://127.0.0.1:8765`

## Testing

- Run the main test suite with `make test` or `uv run pytest`
- Use `make lint` for formatting and lint checks
- Keep default tests independent from local services such as LM Studio
- The audio fixture test is opt-in via `RECALLD_RUN_AUDIO_FIXTURES=1`

## Configuration and Runtime Notes

- The config file lives at `~/.config/recalld/config.json`
- The app creates its scratch workspace automatically
- Obsidian REST API, an LLM server, `ffmpeg`, and optionally a Hugging Face token are needed for the full workflow
- Be careful with defaults in documentation; keep them aligned with the code

## Coding Standards

- Keep Python code idiomatic and readable
- Match the existing FastAPI and Jinja patterns
- Avoid adding unnecessary abstractions
- Update tests when behavior changes
- Do not modify tests just to make failures disappear

## When In Doubt

- Check the relevant tests first.
- If a change affects runtime behavior, add or update tests before or alongside the code.
- If you find conflicting instructions, follow the repo-specific instructions in the current conversation and the codebase over generic habits.
