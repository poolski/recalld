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

## Prompt Tuning

recalld uses [Langfuse](https://langfuse.com/) (self-hosted, `http://localhost:3000`) for prompt management and experiment tracking.

### First-run setup

Before experiments can run, Langfuse needs prompts, datasets, and evaluator rules in place. Check whether each is already set up before assuming it exists. The detailed operator guide lives in [docs/langfuse.md](docs/langfuse.md).

**Prompts** — verify the production label exists for all prompts in the table below:

```bash
uv run python scripts/langfuse prompt list --label production
```

If any are missing, create them in the Langfuse UI (`http://localhost:3000`) and assign the `production` label. The app will not call the LLM without a resolvable production prompt.

**Datasets** — datasets are created and seeded automatically the first time an experiment runs for a given job, using that job's saved output files as the reference. The job must have completed the relevant pipeline step first (e.g. themes must have run before a themes dataset can be seeded). If the required output file is missing, ask the user to point you at a saved job that has it, or run the pipeline step manually to produce it.

**Evaluator rules** — automated LLM scoring requires evaluator rules configured in the Langfuse UI. These cannot be created through the CLI or API alone because they require an LLM configuration that is set up interactively. **Tell the user** if evaluator rules are not yet in place and ask them to set up the following in the Langfuse UI (Settings → Evaluators):

| Rule name           | Evaluates                                          | Fires on            |
| ------------------- | -------------------------------------------------- | ------------------- |
| `summary-evaluator` | Overall summary quality                             | Summary datasets    |
| `style-evaluator`   | Whether the writing style matches the style prompt | Style datasets      |
| `focus-evaluator`    | Whether focus points are captured correctly        | Focus datasets      |

Once created, restrict each rule to its relevant datasets using:

```bash
uv run python scripts/langfuse eval-rule update-filter <rule-id> \
  --filter-dataset <dataset-id> ...
```

### Tool preference

Use tools in this order. Stop at the first one that works for your task:

1. **`scripts/langfuse` CLI** — the preferred tool for all Langfuse operations. Run with `uv run python scripts/langfuse <command>`. It covers prompts, datasets, evaluators, evaluation rules, traces, and experiments.
2. **Extend `scripts/langfuse`** — if the CLI is missing a command you need, **add it** rather than reaching for a one-off curl or Python call. The script is a single self-contained file with a consistent `argparse` pattern; adding a subcommand takes a few minutes and leaves a reusable, documented tool behind. Prefer this over any workaround.
3. **Langfuse MCP tools** (`mcp__langfuse__*`) — use for operations that are genuinely interactive (e.g. composing prompt text from scratch), where a CLI subcommand would not add value.
4. **Direct API calls** (`curl http://localhost:3000/api/public/...`) — only for one-off exploratory calls during debugging. Never leave a workflow that relies on raw curl in the codebase or in documentation.

### Prompts

recalld uses these Langfuse prompts:

| Prompt name                           | Pipeline step                                    |
| ------------------------------------- | ------------------------------------------------ |
| `recalld/themes-single`               | Theme extraction (short transcript)              |
| `recalld/themes-map`                  | Theme extraction (map phase, long transcript)    |
| `recalld/postprocess-style-analysis`  | Style analysis sub-prompt                        |
| `recalld/postprocess-summary-single`  | Summary generation (short transcript)            |
| `recalld/postprocess-summary-reduce`  | Summary generation (reduce phase)                |
| `recalld/postprocess-summary-map`     | Summary generation (map phase)                   |
| `recalld/style-instructions`          | Style sub-prompt (composed into summary prompts) |
| `recalld/theme-guidance-instructions` | Theme guidance sub-prompt                        |
| `recalld/focus-instructions`          | Focus points sub-prompt                          |

Sub-prompts (`recalld/style-instructions`, etc.) are composed into parent prompts using `@@@langfusePrompt:name=...|label=production@@@` syntax. Editing a sub-prompt's `production` label updates all parent prompts that reference it.

### Workflow: iterate on a prompt

1. **Inspect the current production prompt:**

   ```bash
   uv run python scripts/langfuse prompt get recalld/themes-single --label production
   ```

2. **Create a new version with the `candidate` label** — edit via Langfuse UI or MCP, then promote:

   ```bash
   # Clone production → candidate to create the starting point
   uv run python scripts/langfuse prompt clone-label production candidate \
     --name recalld/themes-single --name recalld/themes-map

   # After editing in the UI, confirm the version is correct
   uv run python scripts/langfuse prompt get recalld/themes-single --label candidate
   ```

3. **Run an experiment comparing labels:**

   ```bash
   # Themes experiment
   uv run python scripts/langfuse experiment themes \
     --job-id <job-id> \
     --prompt-label production \
     --prompt-label candidate \
     --run-tag <model-shortname>

   # Summary experiment
   uv run python scripts/langfuse experiment summary \
     --job-id <job-id> \
     --prompt-label production \
     --prompt-label candidate

   # Style sub-prompt experiment
   uv run python scripts/langfuse experiment style \
     --job-id <job-id> \
     --prompt-label production \
     --prompt-label candidate

   # Focus experiment
   uv run python scripts/langfuse experiment focus \
     --job-id <job-id> \
     --prompt-label production \
     --prompt-label candidate
   ```

   Each experiment run creates a dataset item for the job (if not already present), runs both labels, and logs evaluator scores to Langfuse. Experiments run in the `experiments` Langfuse tracing environment.

4. **Check scores in Langfuse UI** at `http://localhost:3000`.

   Key scores:
   - `reference_alignment` — automated score comparing output structure to stored reference (title similarity + count similarity for themes; summary similarity + focus overlap for summary).
   - `summary-evaluator`, `style-evaluator`, `focus-evaluator` — LLM-based evaluators that fire automatically on their matching datasets. They do **not** fire against themes datasets.

5. **Promote the candidate if it wins:**
   ```bash
   uv run python scripts/langfuse prompt update-labels \
     recalld/themes-single <version-number> production
   ```

### Shared themes dataset

The `recalld/themes` dataset contains items from multiple jobs, enabling cross-job themes evaluation. When adding a new job to it:

```bash
uv run python scripts/langfuse experiment themes \
  --job-id <new-job-id> \
  --prompt-label production \
  --dataset-name recalld/themes
```

### Dataset filter maintenance

The LLM evaluator rules (`summary-evaluator`, `style-evaluator`, `focus-evaluator`) only fire against a specific allowlist of dataset IDs. When you add a new job and create its summary, style, or focus dataset, add the new dataset IDs to the evaluator filters:

```bash
uv run python scripts/langfuse eval-rule update-filter <rule-id> \
  --filter-dataset <id1> --filter-dataset <id2> --filter-dataset <id3> --filter-dataset <id4>
```

Get rule IDs with:

```bash
uv run python scripts/langfuse eval-rule list
```

Get dataset IDs with:

```bash
uv run python scripts/langfuse dataset get recalld/jobs/<job-id>/summary
```

### Per-job vs shared datasets

- **Per-job datasets** (`recalld/jobs/<id>/themes`, `.../summary`, `.../style`, `.../focus`) — created automatically by experiment commands; used for single-job comparisons.
- **Shared dataset** (`recalld/themes`) — used for multi-job themes evaluation with `--dataset-name recalld/themes`.

When setting up a new job, run experiments once per type to create the per-job datasets, then add the summary, style, and focus dataset IDs to the evaluator rule filters.

### Experiment defaults

When you pass one or more `--llm-model` values to an experiment command, the CLI defaults to loading each model with a context length of `32768` unless you override `--context-length`.

Each LLM call uses the LM Studio preset `@local:transcript-summariser`.
