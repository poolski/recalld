# Setup

This guide takes you from a clean machine to processing your first recording.

## What You Will Need

### Required

**Python 3.11 or newer**
Check your version with `python3 --version`. If you need to install or upgrade Python, use [python.org](https://www.python.org/downloads/) or your system package manager.

**uv**
`uv` is a fast Python package manager used to run recalld. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**ffmpeg**
Used to convert audio files before transcription. Install with Homebrew on macOS:

```bash
brew install ffmpeg
```

**LM Studio**
recalld uses a local AI model to generate notes. LM Studio lets you download and run these models on your own machine — nothing is sent to the cloud. Download it from [lmstudio.ai](https://lmstudio.ai).

**Obsidian**
recalld writes notes directly into an Obsidian vault. Download it from [obsidian.md](https://obsidian.md).

### Optional

**Hugging Face account and token**
Needed only if you want automatic speaker identification (telling the AI who said what). Create a free account at [huggingface.co](https://huggingface.co) and generate a token in your account settings.

---

## Install

```bash
git clone <repo-url>
cd recalld
make setup
```

`make setup` installs Python dependencies, copies a `transcript-summariser` preset into LM Studio with recommended generation settings, and loads the default AI prompts into Langfuse if you have it running. If LM Studio or Langfuse is not running, those steps are skipped — you can re-run `make setup` at any time.

---

## Configure Your Tools

### LM Studio

1. Open LM Studio.
2. Go to the **Discover** tab and search for a model. A good starting point is **Gemma 3** (4B or 12B depending on your machine).
3. Download the model.
4. Go to **Developer** → **Local Server** and start the server. The default address is `http://localhost:1234`.
5. Run `make setup` (with LM Studio open) to install the `transcript-summariser` preset with recommended generation settings. If you have already run setup, check **Presets** in LM Studio to confirm the preset is there.

### Obsidian Local REST API

recalld writes notes through a plugin called Local REST API, which lets apps talk to Obsidian on your machine.

1. Open Obsidian and go to **Settings** → **Community plugins** → **Browse**.
2. Search for **Local REST API** and install it.
3. Enable the plugin and open its settings. Copy the **API key** — you will need it shortly.
4. The plugin listens on `https://127.0.0.1:27124` by default.

---

## Start recalld

```bash
make run
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser.

---

## Configure recalld

On first launch, open the **Settings** page (gear icon) and fill in:

| Setting            | What to enter                                                       |
| ------------------ | ------------------------------------------------------------------- |
| Vault name         | The name of your Obsidian vault (shown at the top-left in Obsidian) |
| Obsidian API URL   | `https://127.0.0.1:27124` (the default)                             |
| Obsidian API key   | The key you copied from the Local REST API plugin                   |
| LLM base URL       | `http://localhost:1234/v1` (the LM Studio default)                  |
| LLM model          | The model ID shown in LM Studio — e.g. `google/gemma-3-4b-it`       |
| Whisper model      | Start with `small`; use `medium` or `large` for better accuracy     |
| Hugging Face token | Your HF token, if you want speaker identification                   |

Save the settings.

---

## Create a Category

A **category** tells recalld where to save notes and who the speakers are. You need at least one before you can process a recording.

Go to the **Categories** page and click **Add category**. Fill in:

- **Name** — a label for this type of conversation, e.g. "Work meetings" or "Coaching sessions"
- **Vault path** — the folder inside your Obsidian vault where notes should be saved, e.g. `/Work/Meetings`
  - This is **not** the path to your vault on disk. `recalld` interacts with the Obsidian REST API.
- **Speaker A name** — your name (recalld refers to you as "you" in notes)
- **Speaker B name** — the other person's name or role, e.g. "Dave" or "Client"

---

## Process Your First Recording

1. Make sure Obsidian is open and the Local REST API plugin is running.
2. Make sure LM Studio is open and the local server is started (Developer tab → Start server). recalld loads the model automatically — you do not need to load it manually.
3. Make sure recalld is running.
4. Go to the **Upload** page, choose an audio file (`.m4a`, `.mp3`, `.wav`, and others are supported), select your category, and click **Upload**.

recalld works through seven stages. Three of them pause and wait for your input before continuing.

| Stage           | UI label           | What happens                                                                                                                                                                                  |
| --------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ingest**      | Ingest             | Converts your audio to the format Whisper needs. Requires ffmpeg.                                                                                                                             |
| **Transcribe**  | Transcribe         | Runs speech-to-text locally using Whisper. No internet required.                                                                                                                              |
| **Diarise**     | Detect speakers    | Segments the transcript by speaker. Requires a Hugging Face token. If you skipped the token, click **Continue without diarisation** — the note will treat all speech as one speaker.          |
| **Align**       | Confirm speakers ⏸ | **Pauses.** Shows which voice is which and lets you assign names. Confirm to continue.                                                                                                        |
| **Themes**      | Propose themes ⏸   | **Pauses.** recalld proposes section headings for the note. Review, edit if needed, then confirm. This step requires LM Studio. If it fails, you can skip it and generate a themes-free note. |
| **Postprocess** | Summarise          | Generates the note using the themes and transcript. Requires LM Studio.                                                                                                                       |
| **Vault**       | Send to Obsidian ⏸ | **Pauses.** Shows the note title and a preview. If a note with that name already exists, you can overwrite or append. Confirm to write to Obsidian.                                           |

When all stages are complete, recalld shows a link to open the note directly in Obsidian.
