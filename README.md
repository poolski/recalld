# recalld

`recalld` is a local web app that turns a conversation audio recording into an Obsidian note. It transcribes the audio, identifies who is speaking, matches speakers to names you choose, generates a summary and action items, then writes the finished note directly to your vault.

Everything runs on your own machine. No data leaves your computer.

---

## Requirements

| Requirement | Notes |
|---|---|
| [Python 3.11+](https://www.python.org/downloads/) | Check with `python3 --version` |
| [uv](https://docs.astral.sh/uv/) | Installs dependencies and runs recalld |
| [ffmpeg](https://ffmpeg.org/) | Converts audio before transcription |
| [LM Studio](https://lmstudio.ai/) | Runs the local AI model. Download a 4–8 B parameter model. `make setup` installs a `transcript-summariser` preset with recommended generation settings — run it with LM Studio open. |
| [Obsidian](https://obsidian.md/) + [Local REST API plugin](https://github.com/coddingtonbear/obsidian-local-rest-api) | Where notes are saved. Install the plugin from the Obsidian community plugins browser and enable it. |
| [Hugging Face](https://huggingface.co/) account and token | Required for speaker identification. Accept the [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) terms, then generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Without a token, recalld transcribes but cannot tell speakers apart. |

---

## Install

### 1. Install system tools

**macOS (Homebrew)**

```bash
brew install python ffmpeg uv
```

**Debian / Ubuntu**

```bash
sudo apt install python3 ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Fedora**

```bash
sudo dnf install python3 ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Get the code

```bash
git clone https://github.com/your-org/recalld.git
cd recalld
```

### 3. Run setup

```bash
make setup
```

This installs Python dependencies, loads the default AI prompts into Langfuse (if you have it running), and copies a `transcript-summariser` preset into LM Studio with recommended generation settings. If Langfuse or LM Studio is not running, those steps are skipped gracefully — you can re-run `make setup` later.

---

## Configure

Start recalld, then open **Settings** in the browser to fill in:

| Setting                    | What it is                                                                                                              |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Obsidian API URL**       | The URL shown in the Local REST API plugin settings. Default: `https://127.0.0.1:27124`                                 |
| **Obsidian API key**       | The key shown in the Local REST API plugin settings. Copy it from there.                                                |
| **LLM base URL**           | The URL of the LM Studio local server. Default: `http://localhost:1234/v1`                                              |
| **LLM model**              | The model ID from LM Studio — copy it exactly from the LM Studio interface.                                             |
| **Whisper model**          | Controls transcription quality. `small` is a good starting point; use `large` for better accuracy at the cost of speed. |
| **Hugging Face token**     | Your token from huggingface.co. Required for speaker identification.                                                    |
| **Scratch retention days** | How long to keep old job files before they are deleted automatically.                                                   |

### Categories

A **category** is a profile for a recurring type of meeting — for example, "1:1 with Alice" or "Team standup".

| Field | What it does |
|---|---|
| Vault path | Folder inside your Obsidian vault where notes are saved |
| Speaker names | What to call each person in the note |
| Focus note | Optional note to append action items to |

You need at least one category before you can process a recording. Create one in the **Categories** section of the app.

---

## Use

1. Open **Obsidian** and confirm the Local REST API plugin is active (green indicator in its settings).
2. Open **LM Studio** and start the local server (Developer tab → Start server). recalld loads the model automatically.
3. Start recalld:
   ```bash
   make run
   ```
4. Open <http://127.0.0.1:8765> in your browser.
5. Click **New job**, choose a category, and upload an audio file.
6. recalld processes the recording in stages — transcription, speaker identification, theme extraction, and note generation. Each stage runs automatically.
7. When speaker identification finishes, recalld pauses and asks you to confirm which voice belongs to which name. Make your selections and continue.
8. When the note is ready, recalld writes it to your Obsidian vault. If Obsidian is open, the note opens automatically.

---

## Other commands

| Command | What it does |
|---|---|
| `make test` | Run the test suite |
| `make lint` | Check code style |
| `make fmt` | Auto-fix formatting |
| `make setup` | Re-run first-time setup (safe to run again) |

---

## Where files are stored

| What                | Where                            |
| ------------------- | -------------------------------- |
| Config              | `~/.config/recalld/config.json`  |
| Prompt cache        | `~/.config/recalld/prompts.json` |
| Jobs and recordings | `~/.local/share/recalld/jobs/`   |

If the config file becomes corrupted, delete it and restart — recalld will recreate it with defaults.

---

## Documentation

- [Full setup guide](docs/setup.md)
- [Langfuse prompt management](docs/langfuse.md)
- [Running experiments](docs/experiments.md)
- [Troubleshooting](docs/troubleshooting.md)
