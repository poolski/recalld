# recalld

`recalld` is a local web app for turning conversation audio into Obsidian notes.

It helps you:

- transcribe audio
- figure out who spoke when
- match speakers to the names you choose
- generate a summary and key points
- write the finished note to an Obsidian vault

## Requirements

- [Python 3.11+](https://www.python.org/downloads/)
- [`uv`](https://docs.astral.sh/uv/)
- [`ffmpeg`](https://ffmpeg.org/)

For the full workflow, you also need:

- [Obsidian](https://obsidian.md/) with the [Local REST API plugin](https://github.com/coddingtonbear/obsidian-local-rest-api) enabled
- an LLM server such as [LM Studio](https://lmstudio.ai/)
- a [Hugging Face](https://huggingface.co/) token for [`pyannote.audio`](https://github.com/pyannote/pyannote-audio)

## Install

Install the system dependencies first, then set up the project.

1. Install Python 3.11 or newer:
   - macOS with Homebrew: `brew install python`
   - Debian/Ubuntu: `sudo apt install python3`
   - Fedora: `sudo dnf install python3`
   - Arch Linux: `sudo pacman -S python`
   - Otherwise, install it from [python.org](https://www.python.org/downloads/)
2. Install `ffmpeg`:
   - macOS with Homebrew: `brew install ffmpeg`
   - Debian/Ubuntu: `sudo apt install ffmpeg`
   - Fedora: `sudo dnf install ffmpeg`
   - Arch Linux: `sudo pacman -S ffmpeg`
   - Windows: use Chocolatey, Winget, or another package manager
3. Install `uv`:
   - macOS with Homebrew: `brew install uv`
   - macOS and Linux: follow the instructions at [docs.astral.sh/uv](https://docs.astral.sh/uv/)
   - Windows: use the same `uv` installation guide
4. Get the code and open the project folder.
5. Install the project dependencies:

```bash
uv sync --all-groups
```

## Run

Start the app with:

```bash
make run
```

Then open <http://127.0.0.1:8765> in your browser.

## Configure

Open the Settings page and set:

- Vault name
- Obsidian API URL
- Obsidian API key
- LLM base URL
- LLM model
- Hugging Face token
- Whisper model
- Scratch retention days

The built-in defaults are:

- Obsidian API URL: `https://127.0.0.1:27124`
- LLM base URL: `http://localhost:1234/v1`

You also need at least one category. A category tells the app:

- which Obsidian vault path to write to
- what to call the two speakers in the final note
- optionally, which focus note to append to

## Use

1. Start Obsidian and make sure the Local REST API plugin is running.
2. Start your LLM server and load a model.
3. Run `recalld`.
4. Open the app in your browser.
5. Add a category if you do not have one yet.
6. Upload an audio file.
7. Review the processing stages.
8. Confirm speaker matching or vault writing when prompted.

If everything is configured, the app can open the generated note in Obsidian.

## Test

Run:

```bash
make test
```

Other useful commands:

```bash
make install
make run
make lint
```

## Notes

- Config is stored in `~/.config/recalld/config.json`.
- Scratch workspace is stored in `~/.local/share/recalld`.
- The scratch workspace is created automatically on first run.
- If the config file becomes invalid, delete it and restart the app.

## Documentation

- [Docs hub](docs/README.md)
- [Project setup](docs/setup.md)
- [Langfuse setup](docs/langfuse.md)
- [Experiments](docs/experiments.md)
- [Troubleshooting](docs/troubleshooting.md)
