# Troubleshooting

## recalld Will Not Start

**Check Python version**
```bash
python3 --version
```
recalld requires Python 3.11 or newer.

**Check dependencies are installed**
```bash
uv sync --all-groups
```

**Check ffmpeg is available**
```bash
ffmpeg -version
```
If this fails, install ffmpeg (see [Setup](setup.md)).

**Check the config file**
recalld stores its settings at `~/.config/recalld/config.json`. If this file is corrupted or contains invalid JSON, recalld will refuse to start with an error message. Delete the file and restart — recalld will recreate it with defaults.

---

## The AI Is Not Generating Notes

**Is LM Studio running?**
Open LM Studio and confirm the local server is running (green indicator in the Developer tab). The server must be active for recalld to reach it.

**Is a model loaded?**
In LM Studio, go to **My Models** and confirm a model is loaded (not just downloaded). recalld will explicitly load the model you have configured in Settings, but LM Studio must be open and the server must be running.

**Is the model ID correct?**
The model ID in recalld's Settings must exactly match the identifier shown in LM Studio. Copy it directly from LM Studio to avoid typos.

**Does the `transcript-summariser` preset exist?**
recalld uses an LM Studio preset named `transcript-summariser` for every AI request. In LM Studio, go to **Presets** and confirm this preset exists. Create it if it is missing.

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

**Diarisation is not configured**
Speaker identification requires a Hugging Face token. Add your token in Settings and reprocess the recording. Without a token, recalld will still transcribe, but it cannot distinguish who said what.

**Speakers were assigned the wrong names**
recalld pauses during processing and asks you to confirm which speaker is which. If you assigned the names incorrectly, you can reprocess the recording from the job page.

---

## Notes Are Not Appearing in Obsidian

**Is Obsidian open?**
The Local REST API plugin only works while Obsidian is running. Open Obsidian before starting a recalld job.

**Is the Local REST API plugin enabled?**
In Obsidian, go to **Settings** → **Community plugins** and confirm Local REST API is toggled on.

**Is the API key correct?**
The key in recalld's Settings must match the key shown in the Local REST API plugin settings in Obsidian. Copy it again to be sure.

**Is the vault path correct?**
The vault path in your recalld category must be a folder that exists inside your Obsidian vault. Check that the folder exists and that the path matches exactly (it is case-sensitive).

---

## Langfuse Scores Are Not Appearing

This affects experiments only — normal note generation does not depend on scores.

**Check evaluator rules are configured**
Open Langfuse → **Settings** → **Evaluators**. The relevant evaluator rule must exist and be enabled.

**Check the rule is attached to the right dataset**
Evaluator rules are filtered by dataset. If the rule is not attached to the dataset produced by your experiment, no scores will appear. See [Langfuse](langfuse.md) for how to update the filter.

**Wait a moment**
Langfuse runs evaluators asynchronously. If a run just finished, wait 30–60 seconds and refresh the session page. The scores appear after the background job completes.

**You are looking at a trace, not a session**
Scores are attached to the session, not the individual trace. In Langfuse, navigate to **Tracing** → **Sessions** to see the aggregated scores for a run.

---

## An Experiment Dataset Is Missing

Run the experiment at least once for that job to create the dataset automatically. If the experiment exits early because a required output is missing (for example, the job has no themes yet), complete that stage of the recording in the recalld UI first, then rerun the experiment.

---

## Something Else Is Wrong

Useful places to look:

- The recalld terminal output — error messages and stack traces appear here
- Langfuse at [http://localhost:3000](http://localhost:3000) — traces show exactly what the AI received and returned
- The job folder at `~/.local/share/recalld/jobs/<job-id>/` — contains all the intermediate files for that recording
