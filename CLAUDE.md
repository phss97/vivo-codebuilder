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

### The two agents (src/codebuilder/cc_agent.py)

Two thin wrappers around the SDK's `query()`:

- `run_planner(cwd, prompt, ...)` — read-only (`Read/Grep/Glob/Skill`), `permission_mode="default"`, and `output_format={"type":"json_schema","schema": Plan.model_json_schema()}` so it returns a validated `Plan` via `ResultMessage.structured_output`. Raises `CCAgentError` on schema-retry exhaustion / no output.
- `run_executor(cwd, prompt, ...)` — full tools (`Read/Write/Edit/MultiEdit/Bash/Glob/Grep/Skill`) with `permission_mode="bypassPermissions"` (the ONLY mode that auto-approves arbitrary Bash like `uv sync`/`pytest` headlessly — `acceptEdits` does not). Writes the whole package under `cwd`; returns the transcript (the deliverable is on disk).

Both set `setting_sources=["project"]` + `skills=["rpa","code-review-gate"]` so the SDK discovers the CC skills copied into the workspace (see Skills). Models are env-configurable: `CODEBUILDER_PLANNER_MODEL` (default `claude-opus-4-8`), `CODEBUILDER_EXECUTOR_MODEL` (default `claude-sonnet-5`), plus `*_FALLBACK_MODEL` (default aliases `opus`/`sonnet`). `query_fn` is injectable (default the real `query`) so tests pass a fake async generator — no subprocess/network.

**Model-ID caveat:** the pinned SDK bundles a specific `claude` CLI version; if it doesn't recognize a pinned model ID the fallback alias is used. Verify the actual IDs against a live run (needs the API key).

### Flow (src/codebuilder/main.py)

`CodebuilderFlow(Flow[CodebuilderState])`, five methods wired by decorators:

1. `ingest` (`@start`) — CrewAI auto-merges `inputs={...}` into `self.state` before this runs. Coerces attachments, creates `workspaces/<session_id|flow_id>/{inputs,output}`, materializes attachments, copies the CC skills into `<workspace>/.claude/skills/` (for the planner), and derives `state.project_key`.
2. `plan` (`@listen(ingest)` + `@human_feedback`), **`async`** — builds the planner prompt (`_planner_prompt`) from brief/goals/tech_stack/attachments + prior history, calls `cc_agent.run_planner(cwd=workspace)`, stores `state.plan`, resolves `state.language`, then pauses for review. `@human_feedback` emits `approved | amend | rejected`.
3. `revise_plan` (`@listen("amend")` + `@human_feedback`), **`async`** — re-runs the planner with the prior plan + `state.amendments`, then re-gates. **Recoverability invariant:** runs *during* resume after the pending row is cleared, so it must NEVER raise — any planner failure falls back to the prior plan (annotated with an `open_question`) so the re-gate still fires. See `_prior_plan_snapshot`.
4. `build` (`@listen("approved")`), **`async`** — picks `build_dir` (`patch_existing` → `_resolve_patch_root` under `inputs/`, with a git baseline commit for non-git zips; `new_project` → `output/` + `git init`), copies skills into `build_dir/.claude/skills/` (for the executor), then makes **one** `cc_agent.run_executor(cwd=build_dir)` call to build the whole package. An executor crash is caught and reported via QA, not raised.
5. `finalize` (`@listen(build)`), **`async`** — runs light deterministic QA (`_run_final_qa`), one optional executor repair pass on failure (`CODEBUILDER_MAX_FINAL_QA_REPAIRS`, default 1), captures the git diff for patch jobs, zips + uploads to S3, records history, and returns the completion payload with `qa_passed` as the safety signal. Failed-QA payloads may still carry `project_archive`/`zip_url`/`artifact_urls` — callers must gate on `qa_passed`.

**`plan`/`revise_plan`/`build`/`finalize` are `async def`.** They run on AMP's `resume_async` path (an already-running event loop), so `asyncio.run(query(...))` would raise "cannot be called from a running event loop" — `query()` is an async iterator we `async for` over directly. There must be no `asyncio.run(` in `main.py` (a test enforces this).

**Two identifiers, do not collapse them.** `state.session_id` (caller-supplied) is the user/UI identifier (workspace dir, project_key fallback, S3 first segment, webhook `session_id`). `state.id` (= flow_id, auto-generated) is the flow-execution identifier (history rows, S3 second segment, `from_pending`). **Never pass `id` in kickoff inputs** — it overrides `state.id` before OTel's first span and strands AMP traces (CON-101 / COR-48). **Do not** mutate `CREWAI_STORAGE_DIR` at runtime — HITL resume depends on the default `SQLiteFlowPersistence()` location staying stable.

**HITL wiring.** `feedback_provider.WebhookFeedbackProvider` POSTs to `$CODEBUILDER_APPROVAL_WEBHOOK` and raises `HumanFeedbackPending`; unset → `ConsoleProvider`. Resume goes through `resume(job_id, feedback)`. The `@human_feedback` classifier LLM is `CODEBUILDER_GUARDRAIL_LLM` (default `openai/gpt-5.4-mini`). amend→approve is a two-resume sequence: the first resume runs `revise_plan` and re-pauses; the second routes to `build`.

### The plan is native Markdown

`Plan` (schemas.py) is a light envelope, not a work-package tree:
`{project_name, mode, tech_stack, language, domain, plan_markdown, open_questions, assumptions}`. The planner writes the plan body as Markdown in `plan_markdown`; the frontend renders it verbatim (sanitized) on the approval card, and the executor consumes it directly. `validate_plan` (runtime_qa.py) only checks `plan_markdown` non-empty + valid `mode`. There is no `SubTask`/`FileSkeleton` and no per-file/work-package validation — the executor consumes the plan holistically.

### QA (src/codebuilder/runtime_qa.py)

Light and deterministic: `run_final_qa` provisions the project env (`ensure_project_env` → `uv sync`), then runs `ruff` + `pytest` over the build dir. New-project jobs pass `require_installable=True` (a failed `uv sync` IS the QA failure). Patch jobs degrade to the orchestrator interpreter and allow "no tests" when the repo has none. QA runs in the **generated project's** environment, not codebuilder's — keep the `ensure_project_env` → `project_python` path intact; never run ruff/pytest against a generated src-layout project with bare `sys.executable`. The old gates (per-file deterministic review, mypy symbol-drift, import-completeness, .env consistency, RPA architecture gate, packaging remediation) are gone — the executor writes and self-checks the whole package; this is the cheap insurance on top.

### Schemas (src/codebuilder/schemas.py)

`Plan` (above), `QAReport`, `ArtifactRef`, `ProjectArchiveRef`, `Attachment`, and `CodebuilderState`. The completion payload shape (`zip_url`, `qa_report`, `qa_report_markdown`, `artifact_urls`, `qa_passed`, `project_archive`, `patch`, `status`) is a frontend contract — preserve it. `CodebuilderState.language` carries the caller override or planner-detected output language.

### Per-project history (src/codebuilder/history.py)

Standalone SQLite `project_history` at `$CODEBUILDER_HISTORY_DB`, one row per terminated job keyed by `(project_key, job_id)`. `project_key_from(state)` (canonical git URL, else slug of `project_name`), `record(state)` (upsert on finalize/reject/build-failure), `summarize_for_planner(project_key)` (markdown priors fed into the planner prompt). Observability only — every call site wraps `record(...)` in try/except. `files_touched`/`reviewer_issues` columns are now empty (the executor writes files directly).

### Tools (src/codebuilder/tools/)

`s3_artifacts.py` (upload_file/upload_workspace, presigned URLs; `.claude` and `.git` etc. in `SKIP_DIRS`), `project_env.py` (`ensure_project_env`/`project_python` — the generated project's own venv), `lint_runner_tool.py` (ruff/pytest via the project interpreter), `git_tool.py` (clone/init_and_commit/diff; `_HARNESS_EXCLUDES` keeps `.venv`/`.claude`/caches out of baselines and diffs), `attachment_tool.py` (materialize attachments into `inputs/`). `workspace_tool.py` stays because `attachment_tool`/`lint_runner_tool` import its `resolve_within`; its CrewAI `Tool` classes are unused (CC agents use native file tools scoped by `cwd`).

### Skills (src/codebuilder/skills/)

`rpa` (canonical Portuguese RPA standard) and `code-review-gate` (domain-agnostic acceptance checklist) are CC Skills (SKILL.md with YAML frontmatter). `_install_skills()` copies them into `<workspace>/.claude/skills/` (planner) and `build_dir/.claude/skills/` (executor) so the SDK discovers them from the agent's `cwd` with `setting_sources=["project"]` + `skills=[...]`. The `.claude/` dir is excluded from git diffs, zips, and S3 uploads. Add a new domain skill by dropping it in `skills/` and adding its slug to `cc_agent.SKILLS`.

## Conventions specific to this repo

- Kickoff payloads MUST use `session_id` (not `id`). Optional `language` and `project_name` are honored (auto-merged into state).
- Default models live in `cc_agent.py` env vars, not YAML. `ANTHROPIC_API_KEY` is required; `OPENAI_API_KEY` only feeds the default guardrail classifier.
- The stable project identifier is `state.project_key` — derive via `history.project_key_from(state)`, never reconstruct from the name.
- Archive fields are not success-only. Gate on `qa_passed`/`qa_report.passed`.
- Keep `main.py` free of `asyncio.run(` — the SDK-calling methods are `async` and run in AMP's loop.
- The frontend (`../codebuilder-web`) renders `plan.plan_markdown` (sanitized) + `open_questions`/`assumptions`. Changing the `Plan` shape is a two-repo change.
