"""Claude Agent SDK wrappers: the planner and the executor.

Both are thin ``query()`` calls. The planner is read-only and returns a
structured :class:`Plan` (via the SDK's ``output_format`` JSON-schema mode). The
executor gets full tools + ``bypassPermissions`` (the only mode that
auto-approves arbitrary Bash like ``uv sync``/``pytest``) and builds the whole
package in the job workspace.

``query_fn`` is injectable so tests can pass a fake async generator instead of
spawning the real ``claude`` subprocess — same pattern as the canary.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from claude_agent_sdk import ClaudeAgentOptions, query

from codebuilder.schemas import Plan

log = logging.getLogger(__name__)

# Model IDs are env-configurable. Defaults target the latest tier; the CLI
# accepts the "opus"/"sonnet" aliases as a fallback if a pinned ID isn't
# recognized by the bundled CLI version.
PLANNER_MODEL = os.environ.get("CODEBUILDER_PLANNER_MODEL", "claude-opus-4-8")
EXECUTOR_MODEL = os.environ.get("CODEBUILDER_EXECUTOR_MODEL", "claude-sonnet-5")
PLANNER_FALLBACK_MODEL = os.environ.get("CODEBUILDER_PLANNER_FALLBACK_MODEL", "opus")
EXECUTOR_FALLBACK_MODEL = os.environ.get("CODEBUILDER_EXECUTOR_FALLBACK_MODEL", "sonnet")

# Domain skills packaged under src/codebuilder/skills/ and copied into the job
# workspace at ingest so the SDK discovers them from the agent's cwd.
SKILLS = ["rpa", "code-review-gate"]

ProgressCallback = Callable[[Any], None]


class CCAgentError(RuntimeError):
    """The CC agent failed to produce usable output."""


def _executor_max_turns() -> int | None:
    raw = os.environ.get("CODEBUILDER_EXECUTOR_MAX_TURNS")
    if not raw:
        return None  # ponytail: no cap — CC stops when the build is done; set the env var if a runaway needs bounding
    try:
        return int(raw)
    except ValueError:
        log.warning("CODEBUILDER_EXECUTOR_MAX_TURNS=%r is not an int; ignoring", raw)
        return None


def _is_result_message(message: Any) -> bool:
    return type(message).__name__ == "ResultMessage" or hasattr(message, "structured_output")


def _with_stderr(msg: str, stderr_lines: list[str]) -> str:
    """Append the claude CLI's captured stderr so a subprocess crash surfaces the
    real reason instead of the SDK's opaque 'Check stderr output for details'."""
    detail = "\n".join(stderr_lines[-40:]).strip()
    return f"{msg}\n--- claude stderr ---\n{detail}" if detail else msg


def _assistant_text(message: Any) -> str:
    """Best-effort text extraction across SDK message shapes (message types vary
    by SDK version, so stay duck-typed)."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(getattr(block, "text", "") or "")
            for block in content
            if getattr(block, "text", "")
        ]
        return "\n".join(p for p in parts if p)
    text = getattr(message, "text", "")
    return str(text) if text else ""


async def run_planner(
    *,
    cwd: str | Path,
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    query_fn: Callable[..., Any] = query,
) -> Plan:
    """Read-only planning pass. Returns a validated :class:`Plan` via the SDK's
    structured-output mode. Raises :class:`CCAgentError` on schema-retry
    exhaustion or missing output."""
    stderr_lines: list[str] = []
    options = ClaudeAgentOptions(
        cwd=str(cwd),
        model=model or PLANNER_MODEL,
        fallback_model=PLANNER_FALLBACK_MODEL,
        allowed_tools=["Read", "Grep", "Glob", "Skill"],
        disallowed_tools=["Write", "Edit", "MultiEdit", "Bash"],
        permission_mode="default",
        setting_sources=["project"],
        skills=SKILLS,
        output_format={"type": "json_schema", "schema": Plan.model_json_schema()},
        system_prompt=system_prompt,
        stderr=stderr_lines.append,
    )

    result = None
    try:
        async for message in query_fn(prompt=prompt, options=options):
            if _is_result_message(message):
                result = message
    except Exception as exc:  # noqa: BLE001 — surface the CLI stderr, not the opaque wrapper
        raise CCAgentError(_with_stderr(f"planner query failed: {exc}", stderr_lines)) from exc
    if result is None:
        raise CCAgentError(_with_stderr("planner produced no result message", stderr_lines))
    if getattr(result, "subtype", None) == "error_max_structured_output_retries":
        raise CCAgentError("planner exhausted structured-output retries without a valid Plan")
    data = getattr(result, "structured_output", None)
    if not data:
        raise CCAgentError("planner returned no structured output")
    return Plan.model_validate(data)


async def run_executor(
    *,
    cwd: str | Path,
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    on_message: ProgressCallback | None = None,
    query_fn: Callable[..., Any] = query,
) -> str:
    """Autonomous build pass. Writes files under ``cwd`` with full tools and
    ``bypassPermissions``. Returns the assistant transcript (the deliverable is
    on disk; the transcript is for logging/progress). ``on_message`` is invoked
    for every streamed message so callers can emit progress events."""
    stderr_lines: list[str] = []
    options = ClaudeAgentOptions(
        cwd=str(cwd),
        model=model or EXECUTOR_MODEL,
        fallback_model=EXECUTOR_FALLBACK_MODEL,
        allowed_tools=["Read", "Write", "Edit", "MultiEdit", "Bash", "Glob", "Grep", "Skill"],
        permission_mode="bypassPermissions",
        setting_sources=["project"],
        skills=SKILLS,
        system_prompt=system_prompt,
        max_turns=_executor_max_turns(),
        stderr=stderr_lines.append,
        # AMP runs the job container as root, and the CLI refuses
        # bypassPermissions (= --dangerously-skip-permissions) as root unless
        # IS_SANDBOX=1 marks the environment as sandboxed. The per-job container
        # is exactly that. Merged into the subprocess env; harmless off-AMP.
        env={"IS_SANDBOX": "1"},
    )

    transcript: list[str] = []
    result = None
    try:
        async for message in query_fn(prompt=prompt, options=options):
            if on_message is not None:
                try:
                    on_message(message)
                except Exception as exc:  # noqa: BLE001 — progress emission must never break the build
                    log.warning("executor progress callback failed: %s", exc)
            text = _assistant_text(message)
            if text:
                transcript.append(text)
            if _is_result_message(message):
                result = message
    except Exception as exc:  # noqa: BLE001 — surface the CLI stderr, not the opaque wrapper
        raise CCAgentError(_with_stderr(f"executor query failed: {exc}", stderr_lines)) from exc

    if result is not None and getattr(result, "is_error", False):
        log.warning(
            "executor result reported an error (subtype=%s); QA will catch a bad build",
            getattr(result, "subtype", "?"),
        )
    return "\n".join(transcript)
