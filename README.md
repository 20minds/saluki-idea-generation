# Saluki

Saluki is an AI co-scientist style idea-generation tool. Give it a research
question and it will generate hypotheses, critique them, run tournament-style
ranking, evolve stronger variants, and print the active ideas plus any final
overview.

```sh
uv run saluki --prompt "What is an Ig Nobel worthy research topic?"
```

```
#1  [h0_0]  Elo: 1287

Scores — novelty: 0.95  rigor: 0.85  testability: 0.70  impact: 0.40  simplicity: 0.30

Mechanism:  The auditory frequency of a person's sneeze is determined by the specific resonant frequency of their nasal cavity, which varies based on the individual's facial bone structure.

Prediction: Individuals with higher nasal cavity resonance will produce sneezes with a higher pitch (Hz) when exposed to the same irritant.

Experiment: Use high-fidelity acoustic sensors to record the frequency spectrum of sneezes from a diverse group of subjects while measuring their nasal cavity volume via 3D nasal endoscopy.
```

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An OpenAI-compatible API key and model
- Optional: `EXA_API_KEY` or `TAVILY_API_KEY` for web search during idea research
  and deep reflection. Exa is used when both are configured. Without either key,
  Saluki still runs, but search tools return no results.

## Install

Clone the repository:

```sh
git clone https://github.com/20minds/saluki-idea-generation.git
cd saluki-idea-generation
```

Create the environment and install dependencies:

```sh
uv sync
```

Create a local `.env` file from the example:

```sh
cp .env.example .env
```

For a local OpenAI-compatible server, also set:

```sh
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
```

You can put that in `.env`, pass it with `--base-url`, or export it in your
shell.

## Run

Use the installed console script through `uv`:

```sh
uv run saluki --prompt "What is an Ig Nobel worthy research topic?"
```

Or use the repository wrapper:

```sh
./saluki --prompt "What is an Ig Nobel worthy research topic?"
```

Useful options:

```sh
uv run saluki \
  --prompt "What are promising biomarkers for early sepsis detection?" \
  --model gpt-5.5 \
  --rounds 100 \
  --max-ideas 100 \
  --max-matches-per-idea 4 \
  --verbose
```

Common flags:

- `--model`, `-m`: OpenAI model name.
- `--rounds`, `-r`: generation/evolution rounds.
- `--base-url`: OpenAI-compatible API base URL.
- `--api-key`: API key for this run; overrides `OPENAI_API_KEY`.
- `--decision-mode`: frame ideas as decisions with reversal conditions.
- `--negative-space`: ask ideas to state what they will not do.
- `--no-deep-reflect`: disable Deep Agents reflection.
- `--max-tasks`: stop after a bounded number of queued tasks.
- `--max-ideas`: target number of generated ideas.
- `--max-matches-per-idea`: target tournament matches per idea.
- `--no-parallel`: execute task-queue work one task at a time.
- `--no-task-queue`: use the deprecated LangGraph pipeline.

## Pause And Resume

Saluki persists task-queue state under `~/.saluki/states` by default.

Pause after a productive cycle:

```sh
uv run saluki --prompt "New research question" --pause
```

Resume the most recent persisted run:

```sh
uv run saluki --prompt "ignored while resuming" --continue
```

Use a custom persistence directory:

```sh
uv run saluki --prompt "New research question" --persist-dir ./runs --pause
uv run saluki --prompt "ignored while resuming" --persist-dir ./runs --continue
```

When resuming, the stored question is used; the new `--prompt` is ignored.

## Development

Run the tests:

```sh
uv run python -m pytest
```

Run type checking:

```sh
./scripts/ty_check.sh
```

Install pre-commit hooks:

```sh
pre-commit install
```

The hook set runs:

- Ruff linting with autofix
- Black formatting
- Ty type checking
- SonarQube scan with quality-gate waiting

The SonarQube hook expects a local or reachable SonarQube server and these
environment variables:

```sh
SONAR_HOST_URL=http://localhost:9000
SONAR_TOKEN=...
```

Optional Sonar overrides:

- `SONAR_PROJECT_KEY` (defaults to `saluki-idea-generation`)
- `SONAR_PROJECT_NAME`
- `SONAR_SCANNER_HOST_URL`
- `SONAR_QUALITYGATE_TIMEOUT`

The scanner reads project metadata from
[`config/sonar-project.properties`](./config/sonar-project.properties).
