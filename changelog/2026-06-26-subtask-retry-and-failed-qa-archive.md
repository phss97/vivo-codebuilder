# 2026-06-26 - Subtask Retry Loop and Failed-QA Archives

Audience: coding agents working in this repo and the companion frontend at `/Users/pedro/CrewAI/customers/vivo/pocs/codebuilder-web`.

## What changed

- `CODEBUILDER_MAX_SUBTASK_RETRIES` now defaults to `3`, so each subtask gets one initial writer attempt plus three retries.
- Failed deterministic subtask review now calls the existing `ReviewerCrew.crew()` with the deterministic error output. Reviewer issues and suggestions are fed back to the writer on the next retry.
- If a subtask still fails after all retries, CodeBuilder stops later subtasks to avoid token burn, but still runs `finalize()`.
- `finalize()` now runs final QA, records the failed subtask in the QA report, zips the current workspace, and returns a deterministic `qa_report_markdown`.
- Failed QA payloads may include `project_archive`, `zip_path`, `zip_url`, archive entries in `artifact_urls`, and `qa_report_markdown`. `qa_passed` / `qa_report.passed` is the safety signal.

## Implementation summary

- Backend orchestration now retries failed subtasks through the writer/reviewer loop before giving up on that subtask.
- The reviewer receives deterministic failures such as no-op `modify` outputs and returns concrete feedback for the next writer attempt.
- Failed subtask exhaustion no longer prevents final QA, patch generation, archive creation, or the final completion payload.
- Failed-QA archives are exposed deliberately as salvage artifacts, with `qa_report_markdown` carrying the warnings and next-fix context.
- The frontend now preserves `qa_report_markdown` and renders it beside the download when a failed run still has an archive.

## Deployment summary

- Backend commit: `8c674da` pushed to GitHub `origin/main`.
- Frontend commit: `b02dfb9` pushed to Heroku `main` and released as Heroku `v34`.

## Contract superseded

This supersedes the earlier same-day contract in `2026-06-26-qa-deliverable-contract.md` that made archive fields success-only.

## Verification

- New backend regressions cover retry default, reviewer feedback on modify no-op failures, failed-subtask finalization, and failed-QA archive/report payloads.
