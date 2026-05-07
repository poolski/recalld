# Project Setup

This guide gets a new user from a clean machine to a running `recalld` instance.

## Requirements

Install these first:

- Python 3.11 or newer
- `uv`
- `ffmpeg`

For the full workflow you will also want:

- Obsidian with the Local REST API plugin enabled
- An LM Studio server, or another OpenAI-compatible local LLM server
- A Hugging Face token if you want the diarisation pipeline to use `pyannote.audio`

## Install

1. Clone the repository.
2. Open the project directory.
3. Install dependencies:

```bash
uv sync --all-groups
```

## Run the App

Start the app with:

```bash
make run
```

Then open:

```text
http://127.0.0.1:8765
```

You can also run the app directly with:

```bash
uv run recalld
```

## First Run

The app creates its config and scratch directories automatically:

- Config: `~/.config/recalld/config.json`
- Scratch jobs: `~/.local/share/recalld`

On first launch, open the Settings page and configure:

- Vault name
- Obsidian API URL
- Obsidian API key
- LLM base URL
- LLM model
- Hugging Face token
- Whisper model
- Scratch retention days

The default values are:

- Obsidian API URL: `https://127.0.0.1:27124`
- LLM base URL: `http://localhost:1234/v1`

## Categories

You need at least one category before the app can write notes.

A category defines:

- The Obsidian vault path for the note
- The display names for Speaker A and Speaker B
- An optional focus note path to append to

## Use the App

1. Start Obsidian and make sure the Local REST API plugin is running.
2. Start your LLM server and load a model.
3. Start `recalld`.
4. Open the app in your browser.
5. Add a category if needed.
6. Upload an audio file.
7. Review the processing stages.
8. Confirm speaker matching or vault writing when prompted.

If everything is configured correctly, the app can open the generated note in Obsidian.

## Useful Commands

```bash
make test
make lint
make run
```
