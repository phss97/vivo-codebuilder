# codebuilder

A CrewAI **Flow** that turns a project brief into working code. It plans the work with a human-in-the-loop (HITL) gate, builds the code in an isolated per-job workspace, and reviews every artifact through a writer↔reviewer loop before a final QA pass.

```
brief.json ──▶ ingest ──▶ plan ──▶ [HITL approve/amend/reject]
                                         │
                                         ▼
                          build (writer ↔ reviewer loop) ──▶ finalize (QA + history)
```

Two modes:

- **`new_project`** — agents scaffold a fresh project under `workspaces/<job_id>/output/` and `git init` it.
- **`patch_existing`** — the Git attachment in the brief is cloned into `workspaces/<job_id>/inputs/repo`, agents edit it in place, and a diff is captured on finalize.

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
| `OPENAI_API_KEY` | — | Required. Used by all agents. |
| `MODEL` | `gpt-5.4` | Override the default model. |
| `CODEBUILDER_WORKSPACE_ROOT` | `./workspaces` | Where each job's `inputs/`/`output/` lives. |
| `CODEBUILDER_HISTORY_DB` | `./data/codebuilder_history.db` | Per-project history SQLite log. |
| `CODEBUILDER_APPROVAL_WEBHOOK` | *(unset)* | POST target for HITL plan approvals. Falls back to a console prompt when unset. |
| `CODEBUILDER_WRITER_LLM` | `openai/gpt-5.4` | Override the writer model without editing YAML. |
| `CODEBUILDER_WRITER_REASONING` | `true` | Set to `false` for a faster but less careful writer. |
| `CODEBUILDER_MAX_SUBTASK_RETRIES` | `1` | Per-file writer retry count after deterministic review failure. |
| `CODEBUILDER_MAX_FINAL_QA_REPAIRS` | `1` | Whole-workspace repair attempts after final QA failure. |
| `CODEBUILDER_PROGRESS_WEBHOOK` | *(unset)* | Optional best-effort progress callback after subtasks and final QA. |
| `CODEBUILDER_PROGRESS_WEBHOOK_SECRET` | *(unset)* | Optional shared secret sent as `X-Codebuilder-Progress-Secret`. |

## Running a job

The brief is a JSON payload describing the project. See `brief.json` locally or the example below:

```json
{
  "project_name": "invoice_folder_sorter",
  "brief": "Build a Python RPA script that watches an 'inbox' folder for PDF invoices…",
  "goals": ["Single-file stdlib-first script", "Pytest unit tests", "Structured logging"],
  "tech_stack": ["python", "pypdf", "pytest", "ruff"],
  "attachments": []
}
```

`attachments[]` entries can be `git` URLs (cloned), `zip` / `pdf` / `image` (base64 or path).

### Entrypoints

```bash
# Start a new job (accepts a JSON string or a path to a JSON file)
uv run kickoff brief.json
uv run kickoff '{"project_name": "...", "brief": "...", "goals": [], "tech_stack": [], "attachments": []}'

# Resume a paused flow after HITL review
uv run resume <job_id> "approved"
uv run resume <job_id> "amend: swap pypdf for pdfplumber and add --verbose"
uv run resume <job_id> "rejected: out of scope"

# Render the flow graph
uv run plot          # writes codebuilder_flow.html
```

### HITL approval

After the planner runs, the flow pauses and either:

- POSTs the plan to `$CODEBUILDER_APPROVAL_WEBHOOK` and returns — your webhook later calls `uv run resume <job_id> <feedback>` once a human responds, **or**
- (no webhook configured) prompts on the console for `approved` / `amend: …` / `rejected: …`.

`amend` loops back through the planner with the prior plan + the amendment and gates again.

## Project layout

```
src/codebuilder/
├── main.py                 # CodebuilderFlow: ingest → plan → build → finalize
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
└── knowledge/              # Markdown knowledge sources loaded into each crew
```

History from past runs is summarised and fed to the planner on every new run against the same project. (Crew memory itself currently uses crewai's default global storage; per-project memory isolation was removed because mutating `CREWAI_STORAGE_DIR` at runtime collides with the flow's pending-feedback persistence and broke HITL resume — see `CLAUDE.md`.)

## Dev commands

```bash
uv run ruff check src
uv run pytest -q
uv add <pkg>                 # prefer this over hand-editing pyproject
```

## Notes

- Workspaces, history DB, and the local `.env` are gitignored — see `.gitignore`.
- All file I/O from agents is routed through `Workspace*Tool`, which enforces that relative paths cannot escape the job workspace. Never give agents a raw `FileReadTool` pointed at a real filesystem path.
- Final QA runs deterministic `ruff` + `pytest` across the whole workspace. If it fails, the writer gets one repair pass by default, QA reruns, and artifacts are still returned with the final QA status.
- Crew outputs are validated through pydantic schemas with guardrails (e.g. the planner's `Plan` must have 1–15 subtasks with non-empty `file_path` and `test_criteria`; deterministic review rejects placeholder/TODO-only files).
