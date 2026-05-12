# Running Experiments

An experiment lets you run a past recording through a different version of an AI prompt and compare the results — for example, to see whether a rewritten summary prompt produces better notes than the current one.

You need [Langfuse set up](langfuse.md) before running experiments.

---

## Concepts

| Term | What it means |
|---|---|
| **Job** | Every recording creates a job — a folder at `~/.local/share/recalld/jobs/` containing the transcript, speaker data, themes, and generated note. |
| **Prompt label** | A tag on a prompt version. `production` is what recalld uses for real recordings; `candidate` is a version you are testing. |
| **Dataset** | A Langfuse dataset created for a job when you run an experiment. It stores the inputs so you can run multiple prompt versions against the same recording and compare results. |
| **Score** | Automatically computed after each run. Measures how closely the candidate output matches the production output and how well it meets quality criteria. |

---

## Choose a Recording to Test With

Find a job ID from a recording you have already processed. You can see all your jobs in the recalld UI, or list them on the command line:

```bash
ls ~/.local/share/recalld/jobs/
```

Each folder name is a job ID. Choose one that already has the output you want to test:

| Experiment type | What the job needs       |
| --------------- | ------------------------ |
| `themes`        | A completed themes step  |
| `summary`       | A completed summary step |
| `style`         | A completed summary step |
| `focus`         | A completed summary step |

If a job is missing a step, open the recording in the recalld UI and complete that stage first.

---

## Run a Baseline

Run the production prompt against your chosen job to get a baseline score:

```bash
uv run python scripts/langfuse experiment summary \
  --job-id <job-id> \
  --prompt-label production
```

Replace `summary` with `themes`, `style`, or `focus` depending on what you want to test.

---

## Compare Production Against a Candidate

First, create a candidate version of the prompt you want to test. In Langfuse, go to **Prompts**, open the prompt, edit it, and save the new version with the `candidate` label.

Then run the experiment with both labels:

```bash
uv run python scripts/langfuse experiment summary \
  --job-id <job-id> \
  --prompt-label production \
  --prompt-label candidate
```

recalld runs both prompt versions against the same recording inputs and writes scores for each to Langfuse. Open [http://localhost:3000](http://localhost:3000) and go to **Datasets** to compare the runs side by side.

---

## Compare Multiple Models

You can also test how different AI models perform on the same prompt:

```bash
uv run python scripts/langfuse experiment focus \
  --job-id <job-id> \
  --llm-model google/gemma-4-e4b \
  --llm-model google/gemma-4-e2b \
  --llm-model qwen/qwen3-4b
```

The models must be available in LM Studio. recalld loads each model in turn, runs the experiment, then switches to the next. When `--llm-model` is used, the context window defaults to 32,768 tokens unless you pass `--context-length` to override it.

---

## Understanding Scores

After a run, two types of scores appear in Langfuse:

**Reference alignment** — measures how closely the candidate output matches the production output. A score of `1.0` means identical; `0.0` means nothing overlaps. This score is purely structural — it does not judge quality.

**Evaluator score** — an AI judge reads the output and scores it on quality criteria such as factual accuracy, coverage, and style. This score appears after a short delay while Langfuse runs the evaluator rule in the background.

| Experiment | Scores produced                 |
| ---------- | ------------------------------- |
| `themes`   | reference alignment only        |
| `summary`  | reference alignment + evaluator |
| `style`    | evaluator only                  |
| `focus`    | reference alignment + evaluator |

If evaluator scores are not appearing, see [Troubleshooting](troubleshooting.md).

---

## Inspect Results

```bash
# List all runs for a dataset
uv run python scripts/langfuse dataset runs recalld/jobs/<job-id>/summary

# Fetch the details of a specific trace
uv run python scripts/langfuse trace get <trace-id>
```

You can also browse everything in the Langfuse UI at [http://localhost:3000](http://localhost:3000) — go to **Datasets** to see run comparisons, or **Tracing** to read individual outputs.
