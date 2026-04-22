# Reviewer Checklist

For every CodeArtifact, confirm ALL of the following before marking `passed=True`:

## Correctness
- The file implements the SubTask's description and satisfies its `test_criteria`.
- No obvious logic bugs (off-by-one, wrong comparison operator, swapped args).
- Handles the documented edge cases; fails loudly on undocumented ones.

## Safety
- No hard-coded credentials, API keys, or absolute paths specific to one machine.
- No `eval`, `exec`, or `os.system` on untrusted input.
- File writes/deletes are scoped to the workspace — no paths that escape it.

## Style
- Matches guidance in `python_best_practices.md`.
- Readable identifiers, consistent formatting, no dead code.
- Type hints on public functions.

## Tests
- If the file under review is itself a test file, confirm it covers the
  golden path plus at least one edge case and runs with plain `pytest`.
- If the file is production code, do NOT fail it for lacking inline tests —
  tests belong in a sibling subtask under `tests/`. Missing tests is a
  planner concern, not a per-file reviewer concern.
- Tests don't require network access or external services.

## Dependencies
- Any `import` not in stdlib is justified by SubTask tech_notes.
- No duplicate dependency with different version pins.

## Fail fast
- If `lint_runner` returns a report starting with "SKIP:", the tool was
  unavailable — record it in `suggestions` but do not fail the artifact on
  that basis alone. Review logic manually instead.
- If `lint_runner` returns actual lint violations, set `passed=False` and
  list each issue in `issues`.
- Only fail an artifact for reasons inside the file under review. Do not
  fail it for issues in other files (those are separate subtasks / QA).

## Integration (QA agent only)
- All subtasks' files are consistent: shared schemas match, imports resolve, entry points exist.
- Final project has a `README.md` with setup + run instructions.
- Running the documented entry command in a clean venv produces no errors.
