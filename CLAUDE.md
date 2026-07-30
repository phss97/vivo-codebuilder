# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`codebuilder` is a CrewAI **Flow** (see `[tool.crewai] type = "flow"` in `pyproject.toml`) that ingests a project brief, plans it with human-in-the-loop (HITL) approval, and then builds code in a per-job workspace. **The planning and coding are done by Claude Agent SDK agents** (`claude-agent-sdk`), not CrewAI crews: an Opus planner and a Sonnet executor (see `src/codebuilder/cc_agent.py`). The CrewAI Flow shell is kept only for the platform machinery AMP and the frontend depend on — the `kickoff` entry, the `@human_feedback` HITL gate, progress/completion webhooks, S3 upload, and per-project history.

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

### The three agent calls (src/codebuilder/cc_agent.py)

Three thin wrappers around the SDK's `query()`:

- `run_planner(cwd, prompt, ...)` — read-only (`Read/Grep/Glob/Skill`), `permission_mode="default"`, and `output_format={"type":"json_schema","schema": Plan.model_json_schema()}` so it returns a validated `Plan` via `ResultMessage.structured_output`. Raises `CCAgentError` on schema-retry exhaustion / no output.
- `run_executor(cwd, prompt, ...)` — full tools (`Read/Write/Edit/Bash/Glob/Grep/Skill`) with `permission_mode="bypassPermissions"` (the ONLY mode that auto-approves arbitrary Bash like `uv sync`/`pytest` headlessly — `acceptEdits` does not). Writes the whole package under `cwd`; returns the transcript (the deliverable is on disk).
- `run_reviewer(cwd, prompt, ...)` — RPA-only, read-only (`Read/Grep/Glob/Skill`) and structured as `ProductionReview`. It runs only after deterministic QA passes and reports execution blockers in production wiring; non-RPA jobs skip it. Its prompt treats the current source tree as the only evidence, excludes approved plans/prior reports, and requires each blocker to cite the current file/symbol and broken contract.

Both set `setting_sources=["project"]` + `skills=["rpa","code-review-gate"]` so the SDK discovers the CC skills copied into the workspace (see Skills). Models are env-configurable: `CODEBUILDER_PLANNER_MODEL` (default `claude-opus-5`), `CODEBUILDER_EXECUTOR_MODEL` (default `claude-sonnet-5`), plus `*_FALLBACK_MODEL` (default aliases `opus`/`sonnet`). **Reasoning effort** (the biggest token lever) is `CODEBUILDER_PLANNER_EFFORT` (default `high`) / `CODEBUILDER_EXECUTOR_EFFORT` (default `medium`) / `CODEBUILDER_REPAIR_EFFORT` (default `high`). RPA builds and attached packages with failed preflight are elevated to high build effort. `query_fn` is injectable (default the real `query`) so tests pass a fake async generator — no subprocess/network.

**Cost and model visibility + cap.** Every planner/executor/reviewer call reports the requested model, actual model IDs from `ResultMessage.model_usage`, `total_cost_usd`, tokens, and turns via an `on_usage` callback → persisted in the final `llm_usage` result, logged, and emitted as an `llm_usage` progress event (on success *and* failure, so fallbacks and wasted spend are visible). Executor and reviewer calls share the build budget and raise `CCBudgetExceeded` once their streamed estimate crosses the remaining cap. Estimate rates: `CODEBUILDER_COST_PER_MTOK_INPUT`/`_OUTPUT`. It's a safety valve, not billing (may overshoot by a turn).

**Model-ID caveat:** the pinned SDK bundles a specific `claude` CLI version; if it doesn't recognize a pinned model ID the fallback alias is used. Verify the actual IDs against a live run (needs the API key).

### Flow (src/codebuilder/main.py)

`CodebuilderFlow(Flow[CodebuilderState])`, five methods wired by decorators:

1. `ingest` (`@start`) — CrewAI auto-merges `inputs={...}` into `self.state` before this runs. Coerces attachments, creates `workspaces/<session_id|flow_id>/{inputs,output}`, materializes attachments, copies the CC skills into `<workspace>/.claude/skills/`, derives `state.project_key`, and resolves an attached project root. When a project exists, it runs full deterministic preflight QA and stores `state.preflight_qa_report`. Preflight failure is diagnostic and never stops planning; bounded per-category evidence is passed to both planner and executor.
2. `plan` (`@listen(ingest)` + `@human_feedback`), **`async`** — builds the planner prompt (`_planner_prompt`) from brief/goals/tech_stack/attachments + prior history, calls `cc_agent.run_planner(cwd=workspace)`, stores `state.plan`, resolves `state.language`, then pauses for review. `@human_feedback` emits `approved | amend | rejected`.
3. `revise_plan` (`@listen("amend")` + `@human_feedback`), **`async`** — re-runs the planner with the prior plan + `state.amendments`, then re-gates. **Recoverability invariant:** runs *during* resume after the pending row is cleared, so it must NEVER raise — any planner failure falls back to the prior plan (annotated with an `open_question`) so the re-gate still fires. See `_prior_plan_snapshot`.
4. `build` (`@listen("approved")`), **`async`** — picks `build_dir` (`patch_existing` → `_resolve_patch_root` under `inputs/`, with a git baseline commit for non-git zips; `new_project` → `output/` + `git init`), copies skills into `build_dir/.claude/skills/` (for the executor), then makes **one** `cc_agent.run_executor(cwd=build_dir)` call to build the whole package. An executor crash is caught and reported via QA, not raised.
4b. `build` cost cap — when `CODEBUILDER_MAX_RUN_COST_USD` is set, `run_executor` gets `budget_usd` (cap minus build-so-far). On `CCBudgetExceeded`, `build` marks the job failed and stops making model calls. No changelog/wrap-up agent runs.
5. `finalize` (`@listen(build)`), **`async`** — always runs deterministic package QA when `build_dir` exists, including after an executor crash or budget stop. New projects first receive Ruff's safe fixes and formatter. Once deterministic QA passes, an RPA build gets the read-only production-wiring review. Deterministic or semantic blockers share a bounded executor repair loop (`CODEBUILDER_MAX_FINAL_QA_REPAIRS`, default 3) using `CODEBUILDER_REPAIR_EFFORT`; failed builders and reviewer infrastructure failures never trigger repair. It then captures the patch, zips, uploads, records history, and emits the completion payload. Any failed archive gets a deterministic `CODEBUILDER_REPORT.md` injected by `_zip_build` without touching the source tree. The report contains changed files, preflight/final results, plan, repair count, and next work. Successful archives omit it. Failed payloads remain `status=failed`, `qa_passed=false` while retaining the archive/artifacts/patch when available.

**`plan`/`revise_plan`/`build`/`finalize` are `async def`.** They run on AMP's `resume_async` path (an already-running event loop), so `asyncio.run(query(...))` would raise "cannot be called from a running event loop" — `query()` is an async iterator we `async for` over directly. There must be no `asyncio.run(` in `main.py` (a test enforces this).

**Two identifiers, do not collapse them.** `state.session_id` (caller-supplied) is the user/UI identifier (workspace dir, project_key fallback, S3 first segment, webhook `session_id`). `state.id` (= flow_id, auto-generated) is the flow-execution identifier (history rows, S3 second segment, `from_pending`). **Never pass `id` in kickoff inputs** — it overrides `state.id` before OTel's first span and strands AMP traces (CON-101 / COR-48). **Do not** mutate `CREWAI_STORAGE_DIR` at runtime — HITL resume depends on the default `SQLiteFlowPersistence()` location staying stable.

**HITL wiring.** `feedback_provider.WebhookFeedbackProvider` POSTs to `$CODEBUILDER_APPROVAL_WEBHOOK` and raises `HumanFeedbackPending`; unset → `ConsoleProvider`. Resume goes through `resume(job_id, feedback)`. The `@human_feedback` classifier LLM is `CODEBUILDER_GUARDRAIL_LLM` (default `openai/gpt-5.4-mini`). amend→approve is a two-resume sequence: the first resume runs `revise_plan` and re-pauses; the second routes to `build`.

### The plan is native Markdown

`Plan` (schemas.py) is a light envelope, not a work-package tree:
`{project_name, mode, tech_stack, language, domain, plan_markdown, open_questions, assumptions}`. The planner writes the plan body as Markdown in `plan_markdown`; the frontend renders it verbatim (sanitized) on the approval card, and the executor consumes it directly. `validate_plan` (runtime_qa.py) only checks `plan_markdown` non-empty + valid `mode`. There is no `SubTask`/`FileSkeleton` and no per-file/work-package validation — the executor consumes the plan holistically.

### QA (src/codebuilder/runtime_qa.py)

`run_final_qa` is package-level and aggregate: `uv sync --locked`, Ruff lint/format, native MyPy, `.env.example`/`BaseSettings` consistency plus README `env`/`dotenv` snippet parity, runtime dependencies, console-entry import plus RPA `--help`, and full pytest. RPA additionally checks dynamic access to undeclared settings, `Any/getattr` on injected dependencies, unused client lifecycle pairs, and reruns pytest with `.env.example` temporarily active (restored in `finally`). Every check runs even if an earlier one fails. MyPy absence is blocking for RPA; generic projects may skip it. `CODEBUILDER_TEST_TIMEOUT_SECONDS` defaults to 2400 per pytest pass. Keep the generated-project `ensure_project_env` → `project_python` path intact.

### Schemas (src/codebuilder/schemas.py)

`Plan` (above), internal `ProductionReview`, `QAReport`, `ArtifactRef`, `ProjectArchiveRef`, `Attachment`, and `CodebuilderState`. Semantic findings are folded into existing QA integration notes; the completion payload shape (`zip_url`, `qa_report`, `qa_report_markdown`, `artifact_urls`, `qa_passed`, `project_archive`, `patch`, `status`) remains unchanged. `CodebuilderState.language` carries the caller override or planner-detected output language.

### Per-project history (src/codebuilder/history.py)

Standalone SQLite `project_history` at `$CODEBUILDER_HISTORY_DB`, one row per terminated job keyed by `(project_key, job_id)`. `project_key_from(state)` (canonical git URL, else slug of `project_name`), `record(state)` (upsert on finalize/reject/build-failure), `summarize_for_planner(project_key)` (markdown priors fed into the planner prompt). Observability only — every call site wraps `record(...)` in try/except. `files_touched`/`reviewer_issues` columns are now empty (the executor writes files directly).

### Tools (src/codebuilder/tools/)

`s3_artifacts.py` (upload_file/upload_workspace, presigned URLs; `.claude` and `.git` etc. in `SKIP_DIRS`), `project_env.py` (`ensure_project_env`/`project_python`, including locked-sync markers), `lint_runner_tool.py` (Ruff lint/format, native MyPy, pytest via the project interpreter), `git_tool.py` (clone/init_and_commit/diff; `_HARNESS_EXCLUDES` keeps `.venv`/`.claude`/caches out of baselines and diffs), `attachment_tool.py` (materialize attachments into `inputs/`). `workspace_tool.py` stays because `attachment_tool`/`lint_runner_tool` import its `resolve_within`; its CrewAI `Tool` classes are unused (CC agents use native file tools scoped by `cwd`).

### Skills (src/codebuilder/skills/)

`rpa` (canonical Portuguese RPA standard) and `code-review-gate` (domain-agnostic acceptance checklist) are CC Skills (SKILL.md with YAML frontmatter). `_install_skills()` copies them into `<workspace>/.claude/skills/` (planner) and `build_dir/.claude/skills/` (executor) so the SDK discovers them from the agent's `cwd` with `setting_sources=["project"]` + `skills=[...]`. The `.claude/` dir is excluded from git diffs, zips, and S3 uploads. Add a new domain skill by dropping it in `skills/` and adding its slug to `cc_agent.SKILLS`.

## Conventions specific to this repo

- Kickoff payloads MUST use `session_id` (not `id`). Optional `language` and `project_name` are honored (auto-merged into state).
- Default models live in `cc_agent.py` env vars, not YAML. `ANTHROPIC_API_KEY` is required; `OPENAI_API_KEY` only feeds the default guardrail classifier.
- The stable project identifier is `state.project_key` — derive via `history.project_key_from(state)`, never reconstruct from the name.
- Archive fields are not success-only. Gate on `qa_passed`/`qa_report.passed`.
- Keep `main.py` free of `asyncio.run(` — the SDK-calling methods are `async` and run in AMP's loop.
- The frontend (`../codebuilder-web`) renders `plan.plan_markdown` (sanitized) + `open_questions`/`assumptions`. Changing the `Plan` shape is a two-repo change.
