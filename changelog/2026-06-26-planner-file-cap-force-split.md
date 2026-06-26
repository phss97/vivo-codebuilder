# 2026-06-26 - Planner Work-Package File Cap: Force-Split Instead of Crash

Audience: coding agents working in this repo and the companion frontend at `/Users/pedro/CrewAI/customers/vivo/pocs/codebuilder-web`.

## What changed

- The per-work-package file cap (`MAX_FILES_PER_WORK_PACKAGE`) is raised from `6` to `8`. The Writer produces one `CodeBundleArtifact` per work package and handles ~8 files comfortably, so the common case where the planner bundles 7 files is now simply valid.
- The planner guardrail (`_require_nonempty_plan`) now also **force-splits** any work package over the cap: it fails the guardrail with a "split this into cohesive packages" message naming the offending subtask, so CrewAI re-prompts the planner in-loop (up to `guardrail_max_retries`, default `3`) before the plan ever reaches `validate_plan`.
- The two structural caps (`MAX_WORK_PACKAGES`, `MAX_FILES_PER_WORK_PACKAGE`) moved to `schemas.py` as the single source of truth, shared by the guardrail, `validate_plan`, and the planner prompt.

## Why

A real job crashed during planning with `ValueError('Invalid plan: subtask s03 contains 7 files; work packages may contain at most 6 files')`. The initial `plan()` flow step does not wrap `validate_plan` in try/except, so the oversized-package `ValueError` propagated uncaught and killed the flow ("Algo deu errado"). The old guardrail only rejected empty plans, so the oversized plan sailed through to the hard `validate_plan` check.

## Implementation summary

- `schemas.py`: added `MAX_WORK_PACKAGES = 24` and `MAX_FILES_PER_WORK_PACKAGE = 8`.
- `runtime_qa.py`: imports both caps from `schemas`; `validate_plan` logic unchanged (still the authoritative backstop, now reading the bumped value).
- `crews/planner_crew/planner_crew.py`: `_require_nonempty_plan` rejects oversized packages with a split instruction.
- `crews/planner_crew/config/tasks.yaml`, `README.md`, `CLAUDE.md`: prose updated `6 -> 8` and to describe the force-split behaviour.

## Rejected approach

Deterministic mechanical splitting of an oversized package was rejected: a work package is cohesive (`FileSkeleton.public_api` declares symbols other files import) and `build` writes subtasks sequentially, so blindly chunking files could strand a base module after its importers and break the build. Only the planner knows the dependency structure, so splitting is the planner's job.

## Accepted residual

If the planner can't land an all-≤8 generation within the guardrail's retry budget (raised to `guardrail_max_retries=5` on the expand/amend tasks), CrewAI raises out of `kickoff()` — **before** `validate_plan` is reached, so the un-wrapped `plan()` step crashes. Uncommon, but **possible on large plans**, not near-zero: the planner is told to "prefer fewer work packages," which pushes files-per-package up as project size grows, and the guardrail fails the *whole* plan for one oversized subtask, so each retry regenerates everything and can whack-a-mole (fix s03, break s14). The extra retries thin this tail; a true no-crash guarantee would need a fallback re-kickoff *without* the guardrail (on exhaustion there is no Plan object to soft-accept), which is heavier than the problem warrants today. Documented, not fully closed.

## Verification

- `test_validate_plan_rejects_oversized_work_package` updated to 9 files / "at most 8 files".
- New `test_guardrail_force_splits_oversized_work_package` proves the guardrail rejects a 9-file package (forcing a retry) and accepts an 8-file package.
- `uv run pytest -q tests/test_planning_and_imports.py` and `uv run ruff check src`.
