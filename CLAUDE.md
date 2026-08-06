# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`codebuilder` is a CrewAI **Flow** (see `[tool.crewai] type = "flow"` in `pyproject.toml`) that turns a brief and optional project into an approved, revisioned specification and executes its work-package DAG behind test-driven gates. **The analysis and implementation roles are Claude Agent SDK agents** (`claude-agent-sdk`), not CrewAI crews: read-only intake/planner/reviewer roles plus test-author and executor roles (see `src/codebuilder/cc_agent.py`). The CrewAI Flow shell owns the AMP lifecycle: `kickoff`, native `@human_feedback` gates, routing/resume, progress/completion webhooks, S3 upload, and per-project history.

Why this shape: the Flow-on-AMP surface (kickoff HTTP entry, `@human_feedback` pause + `feedback_ready` webhook with a `callback_url` for resume, the enterprise event stream, `/status`, `/healthcheck`) is all platform-provided. Rewriting codebuilder as a standalone service would mean re-deriving every bit of it. So we keep the Flow and swap only the engine. The `../claude_agent_amp_canary` project proved a CrewAI Flow can call the Agent SDK on AMP: the SDK wheel bundles a platform-specific `claude` CLI binary (the `uv.lock` includes `manylinux` wheels), so `uv sync` on AMP's Linux container provides the CLI with no npm step — it just needs `ANTHROPIC_API_KEY` in the environment.

Async HITL pause/resume works because CrewAI auto-creates a `SQLiteFlowPersistence()` when `HumanFeedbackPending` is raised, and `CodebuilderFlow.from_pending(job_id).resume(feedback)` reads from that same default DB. `@persist()` on the flow class is intentionally NOT applied.

`AGENTS.md` is the auto-generated CrewAI reference; note that the crews it describes have been removed — the only CrewAI concept still in use here is `Flow`.

## Commands

Dependency management uses `uv` (see `uv.lock`). Python must be `>=3.10, <3.14`.

```bash
uv sync                        # install deps
uv add <pkg>                   # add a dep (don't hand-edit pyproject)

# Entrypoints (pyproject [project.scripts] → src/codebuilder/main.py)
uv run kickoff                 # start a job with the hardcoded test inputs in main.py::kickoff()
uv run run_crew                # alias for kickoff
uv run plot                    # render codebuilder_flow.html (flow graph)

uv run ruff check src
uv run pytest -q
uv run pytest -q tests/test_cc_flow.py::test_run_planner_returns_plan   # single test
```

Resume a paused job programmatically: `codebuilder.main.resume(job_id, feedback)`.

## Architecture

### The five scoped agent calls (src/codebuilder/cc_agent.py)

Five thin wrappers around the SDK's `query()`:

- `run_intake(cwd, prompt, ...)` — read-only (`Read/Grep/Glob/Skill`), structured as `IntakeAssessment`; proves scope/assets/stack/verification sufficiency or returns blocking questions.
- `run_planner(cwd, prompt, ...)` — read-only and structured as `Plan`; produces the revisioned identifier, terminology, command, and work-package contract.
- `run_test_author(cwd, prompt, declared_test_files, ...)` — workspace tools enabled with a declared-file-only prompt; the flow's before/after snapshots enforce that boundary.
- `run_executor(cwd, prompt, ...)` — workspace tools enabled with `permission_mode="bypassPermissions"`; implements one package or one repair in an isolated stage.
- `run_reviewer(cwd, prompt, ...)` — read-only and structured as `ProductionReview`; runs after deterministic structured QA passes and reports semantic blockers with ownership and evidence.

All wrappers set `setting_sources=["project"]` + `skills=["rpa","code-review-gate"]` so the SDK discovers the CC skills copied into the workspace (see Skills). Models are env-configurable: `CODEBUILDER_PLANNER_MODEL` (default `claude-opus-5`), `CODEBUILDER_EXECUTOR_MODEL` (default `claude-sonnet-5`), plus `*_FALLBACK_MODEL` (default aliases `opus`/`sonnet`). **Reasoning effort** (the biggest token lever) is `CODEBUILDER_PLANNER_EFFORT` (default `high`) / `CODEBUILDER_EXECUTOR_EFFORT` (default `medium`) / `CODEBUILDER_REPAIR_EFFORT` (default `high`). RPA builds and attached packages with failed preflight are elevated to high build effort. `query_fn` is injectable (default the real `query`) so tests pass a fake async generator — no subprocess/network.

**Cost and model visibility + cap.** Every agent call reports the requested model, actual model IDs from `ResultMessage.model_usage`, `total_cost_usd`, tokens, and turns via an `on_usage` callback → persisted in the final `llm_usage` result, logged, and emitted as an `llm_usage` progress event (on success *and* failure, so fallbacks and wasted spend are visible). Test-author, executor, and reviewer calls share the build budget and raise `CCBudgetExceeded` once their streamed estimate crosses the remaining cap. The legacy whole-build path reserves up to $10 for final review/repair; structured execution continuously computes the persisted remaining budget. Estimate rates: `CODEBUILDER_COST_PER_MTOK_INPUT`/`_OUTPUT`. It's a safety valve, not billing (may overshoot by a turn).

**Model-ID caveat:** the pinned SDK bundles a specific `claude` CLI version; if it doesn't recognize a pinned model ID the fallback alias is used. Verify the actual IDs against a live run (needs the API key).

### Flow (src/codebuilder/main.py)

`CodebuilderFlow(Flow[CodebuilderState])` is routed in phases:

1. **Ingest and intake.** `ingest` materializes attachments in the per-job workspace, installs skills, derives `project_key`, snapshots patch dependency names, and runs diagnostic preflight QA. `assess_intake` is read-only and inspects the actual workspace. Any blocking question or missing verification command routes to `request_intake_feedback`; answers are accumulated and reassessed until ready or rejected.
2. **Specification.** `plan` accepts only a validated structured `Plan`, binds authoritative-asset hashes, renders its compatibility Markdown view, and pauses for `spec_approved | spec_amend | job_rejected`. `revise_plan` increments the revision and re-gates on every spec or QA amendment. **Recoverability invariant:** resumed HITL listeners must not strand the pending row; planner/reassessment failures fall back to another human gate.
3. **Structured package execution.** `_build_structured` copies the attachment or empty baseline into `output/`, records the approved spec hash, orders the package DAG, and keeps `output/` as the canonical last-green tree. `_run_package` stages one copy, calls the test author, rejects undeclared test writes, freezes declared tests, records pre-implementation command results, then calls the executor. Any executor edit outside the package or to frozen tests is restored and reported.
4. **Package QA and repair.** `_structured_qa` checks the approved file/identifier contract, runs the spec's argv-based commands without a shell inside an OS filesystem sandbox, then calls the read-only semantic reviewer. Only code-owned blockers receive bounded automatic repair; test/spec/environment blockers route directly to `review_qa_failure`. Human outcomes are `qa_retry | qa_amend | qa_skip | qa_terminate`. Skip transitively marks dependents while independent packages continue; any skip still blocks release.
5. **Promotion and completion.** `_promote_green_stage` copies only approved package paths into the canonical tree. After all packages, the full contract/commands/reviewer run again. `finalize` produces a successful release zip with `approved-spec.json` and `QA.md`, or uploads the quarantine evidence already prepared for failed/skipped/terminated work. Failed state clears all runnable archive fields.

**All SDK-calling listeners are `async def`.** They run on AMP's `resume_async` path (an already-running event loop), so `asyncio.run(query(...))` would raise "cannot be called from a running event loop" — `query()` is an async iterator we `async for` over directly. There must be no `asyncio.run(` in `main.py` (a test enforces this).

The old Markdown-plan/whole-build/final-QA path remains for direct-call and payload compatibility. New routed jobs must pass intake and return `Plan.is_structured`; do not extend the legacy path with new behavior.

**Two identifiers, do not collapse them.** `state.session_id` (caller-supplied) is the user/UI identifier (workspace dir, project_key fallback, S3 first segment, webhook `session_id`). `state.id` (= flow_id, auto-generated) is the flow-execution identifier (history rows, S3 second segment, `from_pending`). **Never pass `id` in kickoff inputs** — it overrides `state.id` before OTel's first span and strands AMP traces (CON-101 / COR-48). **Do not** mutate `CREWAI_STORAGE_DIR` at runtime — HITL resume depends on the default `SQLiteFlowPersistence()` location staying stable.

**HITL wiring.** `feedback_provider.WebhookFeedbackProvider` POSTs to `$CODEBUILDER_APPROVAL_WEBHOOK` and raises `HumanFeedbackPending`; unset → `ConsoleProvider`. Resume goes through `resume(job_id, feedback)`. The `@human_feedback` classifier LLM is `CODEBUILDER_GUARDRAIL_LLM` (default `openai/gpt-5.4-mini`). Intake answers re-run sufficiency analysis; spec amendments re-run planning; QA amendments re-run planning; each re-pauses before execution. The frontend must route feedback according to the returned `phase`, not assume every pause is plan approval.

### Structured specification and translation contract

`Plan` is the machine contract: legacy display fields plus `revision`, `package_name`, hashed `authoritative_assets`, `identifier_contract`, `terminology`, `verification_commands`, and `work_packages`. Each `WorkPackageSpec` declares dependencies, build intent, expected behavior, success criteria, test cases, exact file paths/kinds, and public API. `validate_plan` rejects unsafe paths, duplicate/dangling/cyclic package relationships, incomplete criterion/test mappings, invalid commands, and missing required structured fields. `plan.render_markdown()` is only the sanitized approval view; downstream execution uses the structured fields.

`IdentifierContract` values (`packages`, `modules`, `symbols`, `fields`, `environment_variables`, `entry_points`) are copied verbatim. Never translate or synonymize them. `TerminologyEntry` is the sole registry for human-facing prose translations: reuse the existing canonical translation before creating another term. Comments/docstrings follow `Plan.language`; machine identifiers do not.

### QA (src/codebuilder/runtime_qa.py)

The structured path uses `check_spec_contract(plan)` plus `run_verification_commands(plan.verification_commands)`. Commands are argv arrays executed with `shell=False`, a sanitized environment, declared cwd, timeout, required/non-required status, and bounded captured evidence inside an isolated disposable project copy (`sandbox-exec` on macOS or `bwrap` on Linux; otherwise fail closed). Non-build commands are read-only; `category=build` may write only inside the disposable copy and its outputs are discarded. Network is denied unless the approved command sets `network=true`. Bubblewrap contains Linux child processes in a PID namespace; macOS commands must be direct because child creation is denied. Package QA narrows the file contract to the active package; final QA checks the complete plan. The read-only semantic reviewer runs only after deterministic checks pass and fails closed if it cannot provide a valid result.

`run_final_qa` remains the aggregate compatibility/preflight runner: `uv sync --locked`, Ruff lint/format, native MyPy, `.env.example`/`BaseSettings` and README parity, runtime dependencies, console-entry smoke checks, RPA wiring checks, and full pytest. It also enforces the captured patch dependency baseline. Keep the generated-project `ensure_project_env` → `project_python` path intact.

### Schemas (src/codebuilder/schemas.py)

Trust boundaries use strict structured models: `IntakeAssessment`, `Plan`/`WorkPackageSpec`, `VerificationCommand`/`CommandResult`, `QAIssue`, `QAReport`, `PackageResult`, `ProductionReview`, `ProjectArchiveRef`, and `QuarantineArchiveRef`. `QAIssue.owner` (`code | test | spec | environment`) controls automatic repair eligibility. `CodebuilderState` persists the approved spec hash, package cursor/results/repair counts, frozen test hashes, last-green/stage paths, skipped packages, current failure, and quarantine evidence across HITL resumes. Existing completion aliases remain for compatible consumers.

### Per-project history (src/codebuilder/history.py)

Standalone SQLite `project_history` at `$CODEBUILDER_HISTORY_DB`, one row per terminated job keyed by `(project_key, job_id)`. `project_key_from(state)` (canonical git URL, else slug of `project_name`), `record(state)` (upsert on finalize/reject/build-failure), `summarize_for_planner(project_key)` (markdown priors fed into the planner prompt). Observability only — every call site wraps `record(...)` in try/except. `files_touched`/`reviewer_issues` columns are now empty (the executor writes files directly).

### Tools (src/codebuilder/tools/)

`package_workspace.py` is the structured execution boundary: safe tree staging, content snapshots, changed-path validation, exact-path restore/promotion, and quarantine zip creation. It rejects paths outside the workspace and excludes secrets, archives, Git metadata, skills, virtualenvs, and caches. `s3_artifacts.py` handles uploads/presigned URLs; `project_env.py` and `lint_runner_tool.py` support compatibility QA; `git_tool.py` handles baselines/diffs; `attachment_tool.py` materializes `inputs/`. `workspace_tool.py` stays because attachment/lint helpers import `resolve_within`; its CrewAI `Tool` classes are unused.

### Skills (src/codebuilder/skills/)

`rpa` (canonical Portuguese RPA standard) and `code-review-gate` (domain-agnostic acceptance checklist) are CC Skills (SKILL.md with YAML frontmatter). `_install_skills()` copies them into the workspace and each execution stage so the SDK discovers them from the agent's `cwd` with `setting_sources=["project"]` + `skills=[...]`. The `.claude/` dir is excluded from diffs, release/quarantine zips, and S3 uploads. Add a domain skill by dropping it in `skills/` and adding its slug to `cc_agent.SKILLS`.

## Conventions specific to this repo

- Kickoff payloads MUST use `session_id` (not `id`). Optional `language` and `project_name` are honored (auto-merged into state).
- Default models live in `cc_agent.py` env vars, not YAML. `ANTHROPIC_API_KEY` is required; `OPENAI_API_KEY` only feeds the default guardrail classifier.
- The stable project identifier is `state.project_key` — derive via `history.project_key_from(state)`, never reconstruct from the name.
- Runnable archive fields are success-only. `quarantine_archive` is evidence, never a release.
- Keep `main.py` free of `asyncio.run(` — the SDK-calling methods are `async` and run in AMP's loop.
- Preserve exact identifier values across every agent. Translate only human-facing prose through `Plan.terminology`.
- The frontend (`../codebuilder-web`) renders phase-aware intake/spec/QA cards and the sanitized `plan.plan_markdown` compatibility view. Changing HITL outcomes or the `Plan`/failure payload is a two-repo change.
