# 2026-06-26 - Subtask Retry Loop and Failed-QA Archives

Audience: coding agents working in this repo and the companion frontend at `/Users/pedro/CrewAI/customers/vivo/pocs/codebuilder-web`.

## What changed

- `CODEBUILDER_MAX_SUBTASK_RETRIES` now defaults to `3`, so each subtask gets one initial writer attempt plus three retries.
- Failed deterministic subtask review now calls the existing `ReviewerCrew.crew()` with the deterministic error output. Reviewer issues and suggestions are fed back to the writer on the next retry.
- If a subtask still fails after all retries, CodeBuilder stops later subtasks to avoid token burn, but still runs `finalize()`.
- `finalize()` now runs final QA, records the failed subtask in the QA report, zips the current workspace, and returns a deterministic `qa_report_markdown`.
- Failed QA payloads may include `project_archive`, `zip_path`, `zip_url`, archive entries in `artifact_urls`, and `qa_report_markdown`. `qa_passed` / `qa_report.passed` is the safety signal.

## Contract superseded

This supersedes the earlier same-day contract in `2026-06-26-qa-deliverable-contract.md` that made archive fields success-only.

## Verification

- New backend regressions cover retry default, reviewer feedback on modify no-op failures, failed-subtask finalization, and failed-QA archive/report payloads.
