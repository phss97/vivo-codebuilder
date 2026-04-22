# RPA Script Patterns

## Shape of a good RPA script
- Single entry point: `python script.py [args]`.
- Read inputs from args or a `config.yaml`, never hard-code paths.
- Log what you're about to do *before* you do it (visible dry-run).
- Support a `--dry-run` flag that skips side effects.

## File operations
- Use `pathlib.Path`, not string concatenation.
- Always resolve and verify the target is inside the intended directory.
- Batch rename: compute full rename plan, then apply, so partial failures are recoverable.

## Browser / UI automation
- Prefer `playwright` over `selenium` for new scripts (async, fewer quirks).
- Use explicit waits (`wait_for_selector`), not `time.sleep`.
- Capture screenshots on failure for debugging.

## Scheduling
- Keep schedule concerns out of the script. Use cron / Task Scheduler / APScheduler externally.
- The script must be idempotent — safe to re-run if a schedule fires twice.

## Logging & observability
- Use `logging`, configured once at entry. Structured JSON logs if the output is machine-consumed.
- Exit codes: 0 = success, nonzero = caller should alert.

## Credentials
- Never hard-code credentials. Read from env vars or an OS keychain.
- Document required env vars in the README.

## Error recovery
- Network calls: retry with exponential backoff (3 attempts typical), then fail loud.
- Critical side effects (moves, deletes): log a manifest before each batch.
