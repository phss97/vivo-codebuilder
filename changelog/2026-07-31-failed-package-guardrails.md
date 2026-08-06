# 2026-07-31 - Failed Package Guardrails

- Patch jobs preserve the dependency names declared at ingest; final QA blocks removals.
- A configured run-cost cap reserves up to $10 for final review and repair.
- Reaching the initial build allocation is recoverable through final QA and the repair loop.
- Runnable archive fields are success-only. Failed runs return QA evidence and an optional patch.

This supersedes the failed-archive delivery contract from
`2026-06-26-subtask-retry-and-failed-qa-archive.md`.
