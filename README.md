# CodeBuilder

CodeBuilder is a CrewAI Flow that turns a brief and optional project attachment into a
reviewable plan, pauses for human approval, and delegates implementation to Claude Code
agents. CrewAI provides the AMP/HITL lifecycle; one Claude planner and one Claude executor
perform the planning and coding.

```text
brief + attachment
        │
        ▼
ingest + deterministic preflight ──▶ Claude plan ──▶ HITL approval
                                                    │
                                                    ▼
                                             Claude build
                                                    │
                                                    ▼
                     deterministic QA + RPA wiring review + up to three repairs
                                                    │
                                                    ▼
                                    verified or failed project archive
```

Two modes are supported:

- `new_project`: build in `workspaces/<session_id>/output/`.
- `patch_existing`: materialize a Git/zip attachment under `inputs/`, resolve its project
  root, edit it in place, return its diff, and archive the complete project.

## Quality contract

When an attached project can be resolved, CodeBuilder runs preflight QA before planning.
Failures are non-terminal and are included, with bounded per-category output, in both the
planner and executor prompts. This lets the approved plan address observed defects instead
of discovering them after the build.

Preflight and final QA run the complete applicable project checks:

- `uv sync --locked` for installable Python projects;
- `ruff check .` and `ruff format --check .`;
- native project MyPy configuration (required for RPA projects);
- `.env.example` versus Pydantic `BaseSettings` names/prefixes and README
  `env`/`dotenv` snippets;
- RPA runtime dependencies (`pyodbc`, Windows-scoped `pywin32`) and console entry-point
  imports plus a safe `--help` smoke run;
- RPA production wiring: declared settings fields, typed injected dependencies, and
  externally managed client login/connect lifecycle;
- the full pytest suite, plus a second RPA pass with `.env.example` active, with a
  configurable 40-minute default timeout per pass.

All checks run and are aggregated; a passing test suite cannot hide lint, formatting,
typing, configuration, dependency, entry-point, or production-wiring failures. Final QA
covers the whole repository in both modes. Once deterministic QA passes, RPA jobs receive a
read-only semantic review of the real entry point, composition root, adapters, secrets, and
resource cleanup. The current source tree is its only evidence: approved plans, prior reports,
and old review findings are explicitly excluded, and every blocker must cite the current
file/symbol and broken runtime contract. Concrete blockers use the same bounded Claude repair
loop as deterministic failures. New projects receive Ruff's safe fixes and formatter before
each QA pass so model repairs can focus on semantic failures. Non-RPA jobs incur no review
call. Builder/reviewer crashes and exhausted budgets do not trigger another model call.

Every build directory is archived, even when the builder crashes, the budget is exhausted,
or QA remains red. These responses keep `status="failed"` and `qa_passed=false`, but still
return `project_archive`, `zip_path`/`zip_url`, artifacts, and a patch when available. Failed
archives contain a deterministic `CODEBUILDER_REPORT.md` with the reason, changed files,
preflight/final results, repair count, approved plan, and remaining work. The report is
injected into the zip and is not written into the customer source tree. Successful archives
do not contain it.

## Requirements and setup

- Python `>=3.10, <3.14`
- [`uv`](https://docs.astral.sh/uv/)
- `ANTHROPIC_API_KEY` for the Claude Agent SDK
- `OPENAI_API_KEY` only when using the default OpenAI HITL classifier

```bash
uv sync --locked
cp .env.example .env
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

CrewAI remains pinned in `pyproject.toml`; dependency upgrades are deliberate changes, not
part of generated-project remediation.

## Inputs and execution

Call `CodebuilderFlow().kickoff(inputs={...})` with:

| Input | Type | Notes |
|---|---|---|
| `session_id` | `str` | Caller/UI identity and workspace key. Never pass `id`; CrewAI owns `state.id`. |
| `project_name` | `str` | Display name and history fallback key. |
| `brief` | `str` | Requested behavior and acceptance criteria. |
| `goals` | `list[str]` | High-level goals. |
| `tech_stack` | `list[str]` | Technology hints. |
| `attachments` | `list[Attachment]` | Git, zip, PDF, or image inputs. |
| `language` | `str` | Optional output-language override. |

```bash
uv run kickoff  # local hardcoded smoke input
uv run plot     # render the Flow graph
```

The plan pauses at `@human_feedback`. With `CODEBUILDER_APPROVAL_WEBHOOK`, the provider
posts the pending approval and the caller later invokes `codebuilder.main.resume(job_id,
feedback)`. Without a webhook, the provider uses the console.

## Important configuration

See `.env.example` for every setting. The main operational controls are:

| Variable | Default | Purpose |
|---|---:|---|
| `CODEBUILDER_MAX_RUN_COST_USD` | unset | Build/review/repair cost safety cap. |
| `CODEBUILDER_MAX_FINAL_QA_REPAIRS` | `3` | Repair attempts after a normal final-QA failure. |
| `CODEBUILDER_REPAIR_EFFORT` | `high` | Claude reasoning effort for QA repair calls. |
| `CODEBUILDER_TEST_TIMEOUT_SECONDS` | `2400` | Timeout for each full pytest run. |
| `CODEBUILDER_PROVISION_PROJECT_ENV` | `true` | Allow project-local `uv sync`. |
| `CODEBUILDER_WORKSPACE_ROOT` | `./workspaces` | Per-job workspace root. |
| `CODEBUILDER_HISTORY_ENABLED` | `true` | Enable project-history observations. |
| `CODEBUILDER_APPROVAL_WEBHOOK` | unset | HITL notification target. |
| `CODEBUILDER_PROGRESS_WEBHOOK` | unset | Best-effort progress callback. |
| `CODEBUILDER_ARTIFACT_BUCKET` | unset | S3 artifact/archive bucket. |

## Completion contract

Consumers must gate on `qa_passed` or `qa_report.passed`, never on archive presence.

- `project_archive`: primary complete-package deliverable, local path and optional URL.
- `zip_path` / `zip_url`: backward-compatible aliases.
- `artifact_urls`: uploaded archive and optional per-file artifacts.
- `patch`: audit diff for `patch_existing`.
- `preflight_qa_report`: original attached-project QA evidence when preflight ran.
- `qa_report` / `qa_report_markdown`: final deterministic results.
- `final_qa_repair_attempts`: number of repair model calls.

## Repository layout

```text
src/codebuilder/
├── cc_agent.py             # Claude planner/executor SDK wrappers and cost controls
├── main.py                 # CrewAI Flow, prompts, salvage packaging, completion payload
├── runtime_qa.py           # Package QA and deterministic config/runtime checks
├── schemas.py              # Plan, state, QA, and artifact contracts
├── history.py              # Best-effort per-project SQLite history
├── feedback_provider.py    # Webhook/console HITL provider
├── tools/                  # Project env, QA runners, git, attachments, S3
└── skills/                 # Claude rpa and code-review-gate skills
tests/                      # Flow, QA, security, and artifact regressions
```

Keep `plan`, `revise_plan`, `build`, and `finalize` asynchronous: AMP resumes them inside an
existing event loop. Preserve `session_id` versus CrewAI `state.id`, and keep `.claude/`,
virtual environments, caches, and Git metadata out of diffs and archives.
