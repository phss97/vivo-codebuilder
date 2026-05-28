# codebuilder

A CrewAI **Flow** that turns a project brief into working code. It plans the work with a human-in-the-loop (HITL) gate, builds the code in an isolated per-job workspace, and reviews every artifact through deterministic checks plus a domain architecture gate (dispatched on `plan.domain`; e.g. `rpa`).

```
brief.json ──▶ ingest ──▶ plan ──▶ [HITL approve/amend/reject]
                                         │
                                         ▼
                          build (writer ↔ reviewer loop) ──▶ finalize (QA + history)
```

Two modes:

- **`new_project`** — agents scaffold a fresh project under `workspaces/<session_id>/output/` and `git init` it.
- **`patch_existing`** — the Git attachment in the brief is cloned into `workspaces/<session_id>/inputs/repo`, agents edit it in place, and a diff is captured on finalize.

## Requirements

- Python `>=3.10, <3.14`
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An OpenAI API key

## Setup

```bash
# Install deps
uv sync

# Configure environment
cp .env.example .env
# then edit .env and fill in OPENAI_API_KEY
```

Optional environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for the default OpenAI models. |
| `ANTHROPIC_API_KEY` | — | Required only when changing `agents.yaml` to an `anthropic/...` model. |
| `CODEBUILDER_WORKSPACE_ROOT` | `./workspaces` | Where each job's `inputs/`/`output/` lives. |
| `CODEBUILDER_HISTORY_DB` | `./data/codebuilder_history.db` | Per-project history SQLite log. |
| `CODEBUILDER_APPROVAL_WEBHOOK` | *(unset)* | POST target for HITL plan approvals. Falls back to a console prompt when unset. |
| Agent model settings | `agents.yaml` | Planner/writer/reviewer models and reasoning/planning settings live in each crew's YAML. Edit YAML instead of using Python env override plumbing. |
| `CODEBUILDER_GUARDRAIL_LLM` | `openai/gpt-5.4-mini` | Override the model used by the `@human_feedback` guardrail when classifying user replies. |
| `CODEBUILDER_MAX_SUBTASK_RETRIES` | `1` | Per-file writer retry count after deterministic review failure. |
| `CODEBUILDER_MAX_FINAL_QA_REPAIRS` | `1` | Whole-workspace repair attempts after final QA failure. |
| `CODEBUILDER_PROGRESS_WEBHOOK` | *(unset)* | Optional best-effort progress callback after subtasks and final QA. |
| `CODEBUILDER_PROGRESS_WEBHOOK_SECRET` | *(unset)* | Optional shared secret sent as `X-Codebuilder-Progress-Secret`. |

## Running a job

Inputs are passed as top-level keys to `CodebuilderFlow().kickoff(inputs={...})`. The flow expects:

| Input | Type | Notes |
|---|---|---|
| `session_id` | str | Caller-controlled UI/session id. Names the workspace and webhook correlation key. Do not pass `id`; `state.id` is the CrewAI flow id/resume token. |
| `project_name` | str | Display name; also keys per-project history when no git attachment is present. |
| `brief` | str | Free-text description of the project. |
| `goals` | list[str] | High-level goals. |
| `tech_stack` | list[str] | Languages / libraries. |
| `attachments` | list[Attachment] | Each entry is `git` (cloned), `zip` / `pdf` / `image` (base64 or path). |

### Entrypoints

```bash
# Start a new job with the hardcoded test inputs from src/codebuilder/main.py::kickoff()
uv run kickoff

# Render the flow graph
uv run plot          # writes codebuilder_flow.html
```

For programmatic kickoff (HTTP handlers, codebuilder-web, AMP), call `codebuilder.main.kickoff()` after editing the inputs in that function, or call `CodebuilderFlow().kickoff(inputs={...})` directly with your own dict.

### HITL approval

After the planner runs, the flow pauses and either:

- POSTs the plan to `$CODEBUILDER_APPROVAL_WEBHOOK` and returns — your webhook later calls `codebuilder.main.resume(job_id, feedback)` (e.g. `from codebuilder.main import resume; resume("…", "approved")`) once a human responds, **or**
- (no webhook configured) prompts on the console for `approved` / `amend: …` / `rejected: …`.

`amend` loops back through the planner with the prior plan + the amendment and gates again.

## Project layout

```
src/codebuilder/
├── main.py                 # CodebuilderFlow: ingest → plan → build → finalize
├── runtime_qa.py           # Deterministic review, final QA, and domain architecture gate registry
├── schemas.py              # Plan, SubTask, CodeArtifact, ReviewResult, QAReport, …
├── history.py              # Per-project SQLite history (observability only)
├── feedback_provider.py    # WebhookFeedbackProvider + ConsoleProvider fallback
├── crews/
│   ├── planner_crew/       # FileRead + DirectoryRead; produces a Plan
│   ├── writer_crew/        # Workspace tools only; produces a CodeArtifact
│   └── reviewer_crew/      # Lint/test + read/list; fallback review + QA task
├── tools/
│   ├── workspace_tool.py   # Sandboxed read/write/list within a job workspace
│   ├── lint_runner_tool.py # ruff check + pytest -q
│   ├── git_tool.py         # clone / init+commit / diff
│   └── attachment_tool.py  # Materialise brief attachments into inputs/
└── skills/                 # CrewAI skills: rpa (canonical RPA standard) + code-review-gate (domain-agnostic)
```

Cross-run context comes from a `project_history` SQLite table — a summary of past runs (mode, files touched, reviewer issues, QA notes) is fed to the planner on every new run against the same project.

## Dev commands

```bash
uv run ruff check src
uv run pytest -q
uv add <pkg>                 # prefer this over hand-editing pyproject
```

## Notes

- Workspaces, history DB, and the local `.env` are gitignored — see `.gitignore`.
- All file I/O from agents is routed through `Workspace*Tool`, which enforces that relative paths cannot escape the job workspace. Never give agents a raw `FileReadTool` pointed at a real filesystem path.
- Final QA runs deterministic `ruff` + `pytest` across the whole workspace. Skipped lint/tests are failures. If QA fails, the writer gets one repair pass by default, QA reruns, and artifacts are still returned with the final QA status.
- New-project jobs whose plan declares a `domain` (e.g. `rpa`) also run that domain's architecture gate before completion. For `rpa`, missing orchestrator/producer/consumer, Clean Architecture layers, `.env.example`/CCM config, tests, or traceability marks the job failed even if lint/tests pass. Plans without a registered `domain` finalize on lint/test alone.
- Crew outputs are validated through pydantic schemas with guardrails (e.g. the planner's `Plan` must have 1–15 subtasks with non-empty `file_path` and `test_criteria`; deterministic review rejects placeholder/TODO-only files).
