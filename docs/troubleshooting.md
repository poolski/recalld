# Troubleshooting

This page collects the most common setup and experiment issues.

## The App Will Not Start

Check:

- Python 3.11+ is installed
- `uv sync --all-groups` completed successfully
- `ffmpeg` is on your `PATH`
- `~/.config/recalld/config.json` is readable and valid JSON

If the config file is broken, delete it and restart `recalld` so it can be recreated.

## No LLM Responses

Check:

- The LLM server is running
- The configured base URL is correct
- The configured model is loaded
- The LM Studio preset `@local:transcript-summariser` exists

If you are sweeping multiple models, make sure each model is available in LM Studio before starting the experiment.

## Langfuse Scores Are Missing

Check:

- The relevant evaluator rule exists in Langfuse
- The rule is attached to the correct dataset ID
- The experiment used the correct job and prompt label
- You are looking at a session, not only the trace detail page

For summary, style, and focus experiments, the session should eventually show both the reference alignment score and the evaluator score.

If the session only shows `reference_alignment`, the usual causes are:

- The evaluator rule was not configured
- The dataset ID was not added to the rule filter
- Langfuse was slow to attach the evaluator score

## Dataset Not Found

If Langfuse says a dataset does not exist, run the experiment once for that job so the dataset can be seeded automatically.

If the experiment needs an output file that the job does not have yet, run the missing pipeline stage first.

## Model Context Length Looks Wrong

When running model sweeps, `recalld` asks LM Studio to load the model at the requested context length.

If the model still reports the wrong length:

- Confirm the model supports the requested length
- Confirm LM Studio actually reloaded the model
- Try unloading the model in LM Studio and rerunning the sweep

## Need More Help

Useful places to inspect:

- `docs/langfuse.md`
- `docs/experiments.md`
- `scripts/langfuse`
- `recalld/experiments/`
