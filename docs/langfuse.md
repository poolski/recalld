# Langfuse

Langfuse is a local tool that manages the AI prompts recalld uses to generate notes. You run it yourself on your own machine — it does not send any data to the cloud.

**You do not need Langfuse to use recalld.** recalld ships with a set of built-in prompts that work out of the box. Langfuse becomes useful when you want to:

- Edit the prompts that control how notes are written
- See exactly what the AI was asked and what it produced for each recording
- Run experiments to compare different prompts against your own recordings

---

## First-Time Setup

### 1. Install Docker

Langfuse runs inside Docker. Install **Docker Desktop** from [docker.com](https://www.docker.com/products/docker-desktop/) and make sure it is running before continuing.

### 2. Start Langfuse

```bash
mkdir ~/langfuse && cd ~/langfuse
curl -LO https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml
docker compose up -d
```

Wait about 30 seconds for the database to initialise, then open [http://localhost:3000](http://localhost:3000).

### 3. Create an Account and Project

1. Click **Sign up** and create a local account with any email and password. This account exists only on your machine.
2. When prompted, create an **Organisation** — the name does not matter, it is just a container.
3. Inside the organisation, create a **Project**. Name it `recalld`.

### 4. Get Your API Keys

API keys let recalld connect to your Langfuse instance.

1. Inside the project, click the **Settings** icon (bottom-left) → **API Keys**.
2. Click **Create new API key**.
3. Langfuse shows both keys once — copy them before closing the dialog:
   - **Secret key** — starts with `sk-lf-`
   - **Public key** — starts with `pk-lf-`

### 5. Configure Your Environment

In the recalld project directory, copy the example file:

```bash
cp .envrc.example .envrc
```

Open `.envrc` and fill in the three main values:

```bash
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_BASE_URL="http://localhost:3000"
```

The fourth variable, `LANGFUSE_MCP_AUTH`, is only needed if you use an AI coding tool (like Claude Code) to interact with Langfuse directly. To generate it, run:

```bash
echo -n "pk-lf-...:sk-lf-..." | base64
```

Then set it as:

```bash
export LANGFUSE_MCP_AUTH="Basic <the output from above>"
```

Load the file to apply the variables in your current terminal:

```bash
source .envrc
```

If you have [direnv](https://direnv.net) installed, this happens automatically whenever you open the recalld directory.

### 6. Sync Prompts

Once your environment is configured, pull the latest prompts from Langfuse into a local cache. This ensures recalld can still generate notes even if Langfuse is not running:

```bash
uv run python scripts/langfuse prompt sync
```

recalld also runs this sync automatically each time it starts.

---

## Prompts

The prompts are the instructions given to the AI for each part of the note-writing process. Each prompt has a **label** — a tag that marks which version is currently in use:

- `production` — the version recalld uses when processing recordings
- `candidate` — a version you are testing before promoting it to production

To see all current production prompts:

```bash
uv run python scripts/langfuse prompt list --label production
```

To read the text of a specific prompt:

```bash
uv run python scripts/langfuse prompt get recalld/themes-single --label production
```

To edit a prompt, open Langfuse at [http://localhost:3000](http://localhost:3000), go to **Prompts**, click on a prompt, and create a new version. Apply the `production` label to make it active immediately, or `candidate` to test it with experiments first.

### Prompt Names

| Prompt                                | What it controls                                                |
| ------------------------------------- | --------------------------------------------------------------- |
| `recalld/themes-single`               | Extracting section headings from a short transcript             |
| `recalld/themes-map`                  | Extracting section headings from one chunk of a long transcript |
| `recalld/theme-guidance-instructions` | How theme headings are used to organise the summary             |
| `recalld/postprocess-summary-single`  | Writing the final note from a short transcript                  |
| `recalld/postprocess-summary-map`     | Summarising one chunk of a long transcript                      |
| `recalld/postprocess-summary-reduce`  | Combining chunk summaries into a final note                     |
| `recalld/postprocess-style-analysis`  | Extracting your writing style from the transcript               |
| `recalld/focus-instructions`          | Writing the action items section of the note                    |
| `recalld/note-title`                  | Generating the note title                                       |

---

## Traces

Every time recalld processes a recording, it logs a **trace** in Langfuse — a record of what the AI was asked and what it produced at each step.

To view traces, open Langfuse at [http://localhost:3000](http://localhost:3000) and go to **Tracing**.

To fetch a trace from the command line:

```bash
uv run python scripts/langfuse trace get <trace-id>
```

---

## Evaluator Rules

Evaluator rules automatically score the AI output after an experiment run. They are used for experiments only — you do not need them for normal note generation.

To set up evaluators, open Langfuse → **Settings** → **Evaluators** and create rules named:

- `summary-evaluator` — scores summary quality
- `style-evaluator` — scores style profile quality
- `focus-evaluator` — scores action items quality

After creating a rule, restrict it to the datasets you want it to score:

```bash
uv run python scripts/langfuse eval-rule list
uv run python scripts/langfuse eval-rule update-filter <rule-id> --filter-dataset <dataset-id>
```

---

## Useful Commands

```bash
# Sync production prompts to local cache
uv run python scripts/langfuse prompt sync

# Copy the production version of a prompt to a candidate slot for testing
uv run python scripts/langfuse prompt clone-label production candidate \
  --name recalld/themes-single \
  --name recalld/themes-map

# List evaluator rules
uv run python scripts/langfuse eval-rule list

# List runs for a dataset
uv run python scripts/langfuse dataset runs recalld/jobs/<job-id>/summary
```
