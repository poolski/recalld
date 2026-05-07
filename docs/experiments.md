# Running Experiments

This guide is for comparing prompts and models on your own saved job data.

## What Experiments Do

Each experiment:

- Loads a saved job from the scratch workspace
- Seeds a Langfuse dataset for that job if needed
- Runs one or more prompt labels
- Loads one or more LM Studio models
- Writes scores and traces back to Langfuse

The supported experiment types are:

- `themes`
- `summary`
- `style`
- `focus`

## Choose a Job

Pick a job that already has the pipeline output for the step you want to test.

Typical rules of thumb:

- `themes` needs a job with `themes.json`
- `summary` needs `aligned.json` and `postprocess.json`
- `style` needs `aligned.json` and `postprocess.json`
- `focus` needs `aligned.json` and `postprocess.json`

The scratch job directory is usually:

```text
~/.local/share/recalld/jobs/<job-id>
```

## Run a Baseline

To run a single prompt label:

```bash
uv run python scripts/langfuse experiment summary \
  --job-id <job-id> \
  --prompt-label production
```

Repeat the same idea for:

```bash
uv run python scripts/langfuse experiment style \
  --job-id <job-id> \
  --prompt-label production

uv run python scripts/langfuse experiment focus \
  --job-id <job-id> \
  --prompt-label production

uv run python scripts/langfuse experiment themes \
  --job-id <job-id> \
  --prompt-label production
```

## Compare Production vs Candidate

To compare two prompt labels, repeat `--prompt-label`:

```bash
uv run python scripts/langfuse experiment summary \
  --job-id <job-id> \
  --prompt-label production \
  --prompt-label candidate
```

The same pattern works for `style`, `focus`, and `themes`.

## Multi-Model Sweeps

You can compare multiple local models in one run:

```bash
uv run python scripts/langfuse experiment focus \
  --job-id <job-id> \
  --llm-model google/gemma-4-e4b \
  --llm-model google/gemma-4-e2b \
  --llm-model qwen/qwen3-4b
```

When you pass `--llm-model`, the CLI defaults to a context length of `32768` unless you override `--context-length`.

## Model Loading

The experiment runner:

- Loads the requested LM Studio model
- Unloads the previous model before switching when needed
- Uses the LM Studio preset `@local:transcript-summariser` for every query

That keeps the experiment path aligned with the production pipeline.

## Score Types

The score you see depends on the step:

- `themes` uses `reference_alignment`
- `summary` uses `reference_alignment` plus Langfuse evaluator output
- `style` uses Langfuse evaluator output
- `focus` uses Langfuse evaluator output

If a session page only shows `reference_alignment`, the most common causes are:

- The evaluator rule is missing
- The evaluator rule is filtered to the wrong dataset
- The run was too fast and Langfuse had not finished attaching evaluator scores yet

## Inspect Results

Useful commands after a run:

```bash
uv run python scripts/langfuse dataset runs recalld/jobs/<job-id>/summary
uv run python scripts/langfuse dataset runs recalld/jobs/<job-id>/style
uv run python scripts/langfuse dataset runs recalld/jobs/<job-id>/focus
uv run python scripts/langfuse trace get <trace-id>
```

The Langfuse UI should show the trace, session, and score history grouped together once the run completes.
