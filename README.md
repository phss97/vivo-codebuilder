# CodeBuilder

CodeBuilder is a CrewAI Flow that turns a brief and optional project attachment into an
approved, revisioned specification and then executes its work-package DAG with test-driven
gates. CrewAI provides the AMP/HITL lifecycle; narrow Claude Agent SDK roles perform intake,
planning, test authoring, implementation, and semantic review.

```text
brief + attachment
        │
        ▼
ingest + preflight ─▶ read-only intake ──questions─▶ HITL answers
        │                    │ enough
        │                    ▼
        └────────────▶ structured spec ──▶ HITL approve / amend / reject
                                            │ approve
                                            ▼
              per package: tests ─▶ freeze ─▶ code ─▶ commands + review
                                            │ green
                                            ▼
                                  promote into last-green tree
                                            │
                                            ▼
                            release archive or quarantine evidence
```

Two modes are supported:

- `new_project`: start from an empty baseline and promote verified packages into
  `workspaces/<session_id>/output/`.
- `patch_existing`: materialize a Git/zip attachment under `inputs/`, resolve its project
  root, copy it to a last-green output tree, and leave the attachment untouched.

The primary routed flow requires a structured specification. Legacy Markdown plans remain
accepted by lower-level validation and direct-call compatibility paths, but do not bypass
the intake and structured-plan gates.

## Quality contract

When an attached project can be resolved, CodeBuilder runs preflight QA before planning.
Failures are non-terminal and are included, with bounded per-category output, in both the
intake and planner prompts. Patch jobs also snapshot their runtime and development dependency
names; later QA rejects dependency removals. This lets the approved spec address observed
defects without silently shrinking the existing contract.

The read-only intake analyst inspects the brief, attachments, source tree, authoritative
assets, stack, and available verification commands. Any blocking question or missing command
keeps the job at the intake HITL gate. Once intake is ready, the read-only planner produces a
strict, revisioned spec containing:

- authoritative assets with immutable hashes;
- exact package, module, symbol, field, environment-variable, and entry-point identifiers;
- one terminology registry for translated human-facing prose;
- explicit lint, typecheck, test, build, and integration commands;
- a dependency-ordered work-package DAG, where each package declares what to build,
  expected behavior, success criteria, tests, files, and public API.

Code identifiers are copied verbatim and are never translated. The terminology registry is
the single source of truth for prose translations, so later agents reuse an existing term
instead of inventing synonyms. The human can approve, amend, or reject the rendered spec;
each amendment creates another revision and returns to the same approval gate.

Each approved work package runs in an isolated copy of the last-green tree:

1. The test author may write only the declared test files.
2. CodeBuilder snapshots and freezes those tests, then records the pre-implementation
   verification results.
3. The executor may change only files declared by that package. Changes to frozen tests or
   files outside the package are restored and reported as blockers.
4. Deterministic QA checks the exact spec contract and runs every approved command without a
   shell against an isolated disposable project copy in an OS sandbox. Non-build commands are
   read-only; build outputs are discarded with that copy. A read-only semantic reviewer runs
   only after those checks pass. Host reads and network are denied; a spec must explicitly
   approve network when needed. Linux child processes stay in Bubblewrap's PID namespace;
   macOS verification commands must execute directly because child creation is denied.
5. Only a fully green stage is promoted into the canonical last-green tree.

Code-owned blockers enter the bounded package repair loop
(`CODEBUILDER_MAX_FINAL_QA_REPAIRS`, default 3); test-, spec-, environment-, exhausted-budget-,
or reviewer-infrastructure blockers fail closed at the QA HITL gate. The human may retry with
guidance, amend the spec, skip the failed package and all of its dependents, or terminate.
Independent packages may continue after a skip, but any skipped package blocks release.

After all packages pass, CodeBuilder runs the full spec contract, approved commands, and
semantic review once more. A successful release archive contains the last-green project plus
`approved-spec.json` and `QA.md`. A failed, skipped, or terminated run never returns a runnable
project archive; it returns a quarantine archive containing the last-green tree, only the
approved paths from the failed stage, and spec/QA evidence. When a run cost cap is configured,
up to $10 remains reserved for QA review and repair.

## Requirements and setup

- Python `>=3.10, <3.14`
- [`uv`](https://docs.astral.sh/uv/)
- Verification isolation: macOS `sandbox-exec` or Linux `bwrap`. Other environments fail
  closed. On macOS, use direct verifier commands (for example, `python -m pytest`); child
  process creation is denied so background work cannot escape QA.
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

Intake questions, spec review, and unresolved QA each pause at `@human_feedback`. With
`CODEBUILDER_APPROVAL_WEBHOOK`, the provider posts the pending decision and the caller later
invokes `codebuilder.main.resume(job_id, feedback)`. Without a webhook, the provider uses the
console. The completion payload's `phase` identifies the active UI state.

## Important configuration

See `.env.example` for every setting. The main operational controls are:

| Variable | Default | Purpose |
|---|---:|---|
| `CODEBUILDER_PLANNER_MODEL` | `claude-opus-5` | Exact model requested for planning. |
| `CODEBUILDER_EXECUTOR_MODEL` | `claude-sonnet-5` | Exact model requested for build, review, and repair. |
| `CODEBUILDER_MAX_RUN_COST_USD` | unset | Build/review/repair cap; up to $10 is reserved for QA review/repair. |
| `CODEBUILDER_MAX_FINAL_QA_REPAIRS` | `3` | Repair attempts after a normal final-QA failure. |
| `CODEBUILDER_REPAIR_EFFORT` | `high` | Claude reasoning effort for QA repair calls. |
| `CODEBUILDER_EXECUTOR_EFFORT` | `medium` | Base build effort; RPA and failed attached packages are elevated to `high`. |
| `CODEBUILDER_TEST_TIMEOUT_SECONDS` | `2400` | Timeout for each full pytest run. |
| `CODEBUILDER_PROVISION_PROJECT_ENV` | `true` | Allow project-local `uv sync`. |
| `CODEBUILDER_WORKSPACE_ROOT` | `./workspaces` | Per-job workspace root. |
| `CODEBUILDER_HISTORY_ENABLED` | `true` | Enable project-history observations. |
| `CODEBUILDER_APPROVAL_WEBHOOK` | unset | HITL notification target. |
| `CODEBUILDER_PROGRESS_WEBHOOK` | unset | Best-effort progress callback. |
| `CODEBUILDER_ARTIFACT_BUCKET` | unset | S3 artifact/archive bucket. |

## Completion contract

Runnable archive fields are success-only; consumers must still gate on `qa_passed` or
`qa_report.passed`. Quarantine fields are failure evidence and must never be treated as a
deployable package.

- `project_archive`: primary complete-package deliverable, local path and optional URL.
- `zip_path` / `zip_url`: backward-compatible aliases.
- `quarantine_archive` / `quarantine_report`: failed-stage evidence and package outcomes.
- `approved_spec_hash`: hash binding package results and QA to the approved revision.
- `package_results`: passed, failed, or skipped result for each work package.
- `artifact_urls`: uploaded release or quarantine archive and optional per-file artifacts.
- `patch`: audit diff for `patch_existing`.
- `llm_usage`: requested and actual model IDs plus per-call cost/token metrics.
- `preflight_qa_report`: original attached-project QA evidence when preflight ran.
- `qa_report` / `qa_report_markdown`: deterministic, contract, and semantic results.
- `current_failure`: structured blocker ownership and evidence at the QA gate.
- `final_qa_repair_attempts`: total automatic repair model calls.

## Repository layout

```text
src/codebuilder/
├── cc_agent.py             # Scoped intake/planner/test/executor/reviewer wrappers
├── main.py                 # CrewAI Flow, HITL routing, package orchestration
├── runtime_qa.py           # Spec validation and deterministic command runners
├── package_workspace.py     # Isolated stages, promotion, and quarantine bundles
├── schemas.py              # Structured spec, state, QA, and artifact contracts
├── history.py              # Best-effort per-project SQLite history
├── feedback_provider.py    # Webhook/console HITL provider
├── tools/                  # Project env, QA runners, git, attachments, S3
└── skills/                 # Claude rpa and code-review-gate skills
tests/                      # Flow, QA, security, and artifact regressions
```

Keep SDK-calling listeners asynchronous: AMP resumes them inside an existing event loop.
Preserve `session_id` versus CrewAI `state.id`, and keep `.claude/`, virtual environments,
caches, secrets, and Git metadata out of diffs and archives.
