# Troubleshooting

## recalld Will Not Start

| Check | Command | Fix |
|---|---|---|
| Python version | `python3 --version` | Must be 3.11 or newer. Install from [python.org](https://www.python.org/downloads/) or your package manager. |
| Dependencies | `uv sync --all-groups` | Re-run if any packages are missing. |
| ffmpeg | `ffmpeg -version` | Install ffmpeg — see [Setup](setup.md). |
| Config file | — | If `~/.config/recalld/config.json` is corrupted, delete it and restart. recalld recreates it with defaults. |

---

## The AI Is Not Generating Notes

| Check | Fix |
|---|---|
| LM Studio server running | Open LM Studio → Developer tab. Confirm the server is started (green indicator). |
| Model available | recalld loads the model automatically, but LM Studio must be open and the server running. The model must be downloaded under **My Models**. |
| Model ID correct | The model ID in recalld's Settings must exactly match what LM Studio shows. Copy it directly to avoid typos. |
| `transcript-summariser` preset exists | Go to **Presets** in LM Studio and confirm the preset is there. Re-run `make setup` with LM Studio open if it is missing. |

---

## Transcription Is Slow or Inaccurate

The **Whisper model** setting controls transcription quality and speed. Larger models are more accurate but slower:

| Model | Speed | Accuracy |
|---|---|---|
| `tiny` | Very fast | Low |
| `small` | Fast | Decent |
| `medium` | Moderate | Good |
| `large` | Slow | Best |

Change the model in recalld's Settings. The model is downloaded automatically on first use.

---

## Speaker Names Are Wrong or Missing

| Problem | Fix |
|---|---|
| No speaker identification at all | Add your Hugging Face token in Settings and reprocess. Without a token recalld transcribes but cannot tell speakers apart. |
| Wrong names assigned | Open the job page and reprocess from the **Confirm speakers** stage. |

---

## Notes Are Not Appearing in Obsidian

| Check | Fix |
|---|---|
| Obsidian is open | The Local REST API plugin only works while Obsidian is running. |
| Plugin is enabled | Obsidian → **Settings** → **Community plugins** → confirm Local REST API is on. |
| API key correct | The key in recalld's Settings must match the key shown in the plugin settings. Copy it again to be sure. |
| Vault path correct | The path in your category must be a folder that exists inside your vault. It is case-sensitive. |

---

## Langfuse Scores Are Not Appearing

This affects experiments only — normal note generation does not depend on scores.

| Check | Fix |
|---|---|
| Evaluator rule exists and is enabled | Langfuse → **Settings** → **Evaluators**. |
| Rule is attached to the right dataset | Evaluator rules are filtered by dataset. See [Langfuse](langfuse.md) for how to update the filter. |
| Scores just need a moment | Langfuse runs evaluators asynchronously. Wait 30–60 seconds and refresh. |
| You are viewing a trace, not a session | Scores are on the session. Navigate to **Tracing** → **Sessions** to see them. |

---

## An Experiment Dataset Is Missing

Run the experiment at least once for that job to create the dataset automatically. If the experiment exits early because a required output is missing (for example, the job has no themes yet), complete that stage of the recording in the recalld UI first, then rerun the experiment.

---

## Something Else Is Wrong

| Where to look | What you will find |
|---|---|
| recalld terminal output | Error messages and stack traces |
| Langfuse at [http://localhost:3000](http://localhost:3000) → Tracing | Exactly what the AI received and returned for each step |
| `~/.local/share/recalld/jobs/<job-id>/` | All intermediate files for that recording |
