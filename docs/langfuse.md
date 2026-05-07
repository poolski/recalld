# Langfuse Setup

`recalld` uses Langfuse for prompt management, datasets, evaluator rules, traces, and experiments.

## Start Langfuse

Use the self-hosted Langfuse instance at:

```text
http://localhost:3000
```

Make sure Langfuse has working project credentials before trying to run experiments from `recalld`.

## Required Prompts

Check that the production label exists for the prompt library:

```bash
uv run python scripts/langfuse prompt list --label production
```

Required prompts:

- `recalld/themes-single`
- `recalld/themes-map`
- `recalld/postprocess-style-analysis`
- `recalld/postprocess-summary-single`
- `recalld/postprocess-summary-reduce`
- `recalld/postprocess-summary-map`
- `recalld/style-instructions`
- `recalld/theme-guidance-instructions`
- `recalld/focus-instructions`

If any are missing, create or edit them in the Langfuse UI and apply the `production` label.

## Evaluator Rules

Automated scoring needs evaluator rules configured in the Langfuse UI under Settings -> Evaluators.

Create these rules:

- `summary-evaluator`
- `style-evaluator`
- `focus-evaluator`

These are LLM-based evaluators, so they must be created interactively in the UI.

Once they exist, restrict them to the datasets you want them to score:

```bash
uv run python scripts/langfuse eval-rule list
uv run python scripts/langfuse eval-rule update-filter <rule-id> --filter-dataset <dataset-id>
```

### Dataset Mapping

- `summary-evaluator` -> summary datasets
- `style-evaluator` -> style datasets
- `focus-evaluator` -> focus datasets

Themes currently use deterministic `reference_alignment` only.

## Datasets

Datasets are seeded automatically when you run an experiment for a job.

The experiment needs the relevant saved output first:

- `themes` needs the themes output
- `summary` needs aligned and postprocess output
- `style` needs aligned and postprocess output
- `focus` needs aligned and postprocess output

If the required output file is missing, run the pipeline stage first or use a saved job that already has the right artifact.

## CLI Commands

Use the `scripts/langfuse` helper for day-to-day operations:

```bash
uv run python scripts/langfuse prompt get recalld/themes-single --label production
uv run python scripts/langfuse prompt clone-label production candidate --name recalld/themes-single --name recalld/themes-map
uv run python scripts/langfuse evaluator list
uv run python scripts/langfuse eval-rule list
uv run python scripts/langfuse dataset get recalld/jobs/<job-id>/summary
uv run python scripts/langfuse trace get <trace-id>
```

## Tracing Environment

Experiment runs use the `experiments` Langfuse tracing environment.

That keeps experiment traces separate from normal app usage.
