# 2026-06-26 - QA Deliverable Contract Recovery

Audience: coding agents working in this repo and the companion frontend at `/Users/pedro/CrewAI/customers/vivo/pocs/codebuilder-web`.

## Documentation changes in this turn

- Updated `README.md` to describe patch preflight QA, concrete plan validation, full-suite pytest behavior, test/production API drift checks, and success-only runnable archive fields.
- Updated `CLAUDE.md` to align agent guidance with the current patch root resolution, `_build_subtask()` failure behavior, final QA gates, `preflight_qa_report`, and failed-completion payload contract.
- Added this changelog entry so future coding agents can preserve the backend/frontend contract without re-deriving it from commit history.

## Backend changes

- Added `CodebuilderState.preflight_qa_report` as optional deterministic context for `patch_existing` planning.
- `patch_existing` now resolves the attached project root before planning/building. Git attachments usually use `inputs/repo`; zip attachments may need wrapper-directory stripping before planned paths are applied.
- Patch preflight runs full deterministic QA once before planning when a project root can be resolved. The planner receives the truncated QA report and should plan concrete fixes from that evidence.
- `runtime_qa.validate_plan()` rejects placeholder file/test paths such as `FILES_TO_BE_DETERMINED_BY_*`, `TESTS_TO_BE_DETERMINED_BY_*`, `TBD`, and `PLACEHOLDER`.
- Diagnostic-only RPA/code plans are invalid. A plan must name real production, test, or build target files.
- `_build_subtask()` returns its final `ReviewResult`. The build stops on the first unresolved non-passing subtask and records `state.status = "failed"` with a `QAReport`.
- Final QA runs pytest even when deterministic lint/type gates fail, so repair receives acceptance-test failures too.
- Patch final QA runs the full pytest suite whenever test files exist. "No tests collected" is non-blocking only when the project truly has no test files.
- The type gate includes generated tests so `[call-arg]` and `[attr-defined]` drift between tests and production APIs fails QA.
- Failed QA completion payloads omit runnable archive fields: no `project_archive`, `zip_path`, `zip_url`, or shippable archive entries in `artifact_urls`.

## Frontend companion changes

- The companion web app was updated to treat failed-QA/no-archive completion payloads as failed, non-runnable results.
- Heroku received the frontend update only. At the time of this changelog, the local frontend repo was intentionally ahead of its GitHub remote by that commit.

## Contract to preserve

- Never hand a caller a runnable archive when final QA is still failing.
- Do not convert failed archives into user-facing deliverables. Keep any failed archive internal/debug-only.
- Planner prompts and `_planner_inputs()` must stay in sync when adding template variables such as `preflight_qa_report`.
- Generated tests must target the real API already written in production files. If tests and production drift, final QA should fail.
- Repair should receive the full QA picture, not just the first deterministic gate failure.

## Verification from this turn

- Backend tests: `rtk uv run python -m pytest -q` passed with `144 passed`.
- Backend lint: `rtk uv run ruff check src tests` passed.
- Frontend tests: `rtk uv run python -m unittest discover -s tests -q` passed.
- Frontend smoke: Flask import/route smoke passed.

## Useful commits

- Backend: `62a4979 Harden codebuilder QA deliverables`.
- Frontend: `b158bab Handle failed QA completion payloads`.
