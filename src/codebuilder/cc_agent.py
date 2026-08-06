"""Thin Claude Agent SDK wrappers for the CodeBuilder roles.

Intake, planning, and semantic review are read-only structured ``query()``
calls. Test authoring and implementation share the same workspace tool setup;
the caller supplies an isolated workspace and enforces the approved file
boundary after each pass.

``query_fn`` is injectable so tests can pass a fake async generator instead of
spawning the real ``claude`` subprocess — same pattern as the canary.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, TypeVar, cast

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
from pydantic import BaseModel

from codebuilder.schemas import IntakeAssessment, Plan, ProductionReview

log = logging.getLogger(__name__)

# Model IDs are env-configurable. Defaults target the latest tier; the CLI
# accepts the "opus"/"sonnet" aliases as a fallback if a pinned ID isn't
# recognized by the bundled CLI version.
PLANNER_MODEL = os.environ.get("CODEBUILDER_PLANNER_MODEL", "claude-opus-5")
EXECUTOR_MODEL = os.environ.get("CODEBUILDER_EXECUTOR_MODEL", "claude-sonnet-5")
PLANNER_FALLBACK_MODEL = os.environ.get("CODEBUILDER_PLANNER_FALLBACK_MODEL", "opus")
EXECUTOR_FALLBACK_MODEL = os.environ.get(
    "CODEBUILDER_EXECUTOR_FALLBACK_MODEL", "sonnet"
)

# Domain skills packaged under src/codebuilder/skills/ and copied into the job
# workspace at ingest so the SDK discovers them from the agent's cwd.
SKILLS = ["rpa", "code-review-gate"]

# HTTP statuses where retrying the whole query() is worthwhile. When the CLI's
# underlying API call fails with one of these, the SDK surfaces it as a result
# with is_error=True, subtype="success", and api_error_status set (see the SDK's
# ResultMessage.api_error_status). A 400/401/403/404 is NOT here — those are real
# bugs (bad model id, bad key, too-long prompt) and retrying just burns credits.
_TRANSIENT_API_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}

Effort = Literal["low", "medium", "high", "xhigh", "max"]
_EFFORT_LEVELS: set[str] = {"low", "medium", "high", "xhigh", "max"}

ProgressCallback = Callable[[Any], None]
UsageCallback = Callable[[dict], None]
StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


_INTAKE_SYSTEM_PROMPT = """\
You are the read-only intake analyst. Inspect the supplied repository and
evidence, then decide only whether there is enough information to write an
implementable specification with credible verification commands. Ask only
genuinely blocking questions. Do not plan or implement. Preserve every code,
schema, API, environment, and entry-point identifier verbatim; output language
applies only to human-facing prose, comments, and docstrings."""

_REVIEWER_SYSTEM_PROMPT = """\
You are the read-only semantic QA reviewer. Compare the current source and tests
against the approved specification and success criteria supplied in the prompt.
Report only concrete, evidence-backed issues. Do not edit files, propose weaker
tests, or treat a historical report as proof of current behavior."""


class CCAgentError(RuntimeError):
    """The CC agent failed to produce usable output."""


class CCBudgetExceeded(CCAgentError):
    """The executor hit its cost budget mid-run and was stopped. Carries the
    estimated spend and the transcript accumulated so far; the partial build is
    already on disk."""

    def __init__(self, cost_usd: float, transcript: str) -> None:
        super().__init__(
            f"executor stopped at cost budget (est. ${cost_usd:.2f} spent)"
        )
        self.cost_usd = cost_usd
        self.transcript = transcript


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        log.warning("%s=%r is not an int; using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _effort(name: str, default: Effort) -> Effort:
    val = os.environ.get(name, default)
    if val not in _EFFORT_LEVELS:
        log.warning(
            "%s=%r invalid (want one of %s); using %s",
            name,
            val,
            sorted(_EFFORT_LEVELS),
            default,
        )
        return default
    return cast(Effort, val)


# Reasoning effort — the biggest token lever. Claude 5 defaults to high;
# explicitly keep the executor at medium for simple builds and the planner high.
PLANNER_EFFORT = _effort("CODEBUILDER_PLANNER_EFFORT", "high")
EXECUTOR_EFFORT = _effort("CODEBUILDER_EXECUTOR_EFFORT", "medium")
REPAIR_EFFORT = _effort("CODEBUILDER_REPAIR_EFFORT", "high")

# Per-MTok rates for the MID-RUN cost estimate that drives the budget cap.
# Defaults are derived from the executor model (so pointing it at Opus doesn't
# silently undercount ~2.5×) and skew to STANDARD (not intro) pricing — a slight
# overestimate makes the cap stop a touch early, which is the safe direction for
# "don't overspend". This is a safety valve, not billing — the authoritative
# per-call cost is ResultMessage.total_cost_usd.
_MODEL_RATES = {  # (input, output) $/MTok, conservative/standard
    "fable": (10.0, 50.0),
    "opus": (5.0, 25.0),
    "sonnet": (3.0, 15.0),
    "haiku": (1.0, 5.0),
}


def _rates_for(model: str) -> tuple[float, float]:
    m = (model or "").lower()
    for key, rates in _MODEL_RATES.items():
        if key in m:
            return rates
    return (3.0, 15.0)  # default to Sonnet-tier


_default_rates = _rates_for(EXECUTOR_MODEL)
_COST_PER_MTOK_INPUT = _env_float("CODEBUILDER_COST_PER_MTOK_INPUT", _default_rates[0])
_COST_PER_MTOK_OUTPUT = _env_float(
    "CODEBUILDER_COST_PER_MTOK_OUTPUT", _default_rates[1]
)


def _usage_get(usage: Any, key: str) -> int:
    if isinstance(usage, dict):
        val = usage.get(key)
    else:
        val = getattr(usage, key, None)
    return int(val) if isinstance(val, (int, float)) else 0


def _estimate_cost_usd(usage: Any) -> float:
    """Rough $ estimate for one turn's usage. cache writes ≈1.25×, reads ≈0.1×."""
    if usage is None:
        return 0.0
    inp = _usage_get(usage, "input_tokens")
    cache_w = _usage_get(usage, "cache_creation_input_tokens")
    cache_r = _usage_get(usage, "cache_read_input_tokens")
    out = _usage_get(usage, "output_tokens")
    input_cost = (
        (inp + cache_w * 1.25 + cache_r * 0.1) * _COST_PER_MTOK_INPUT / 1_000_000
    )
    output_cost = out * _COST_PER_MTOK_OUTPUT / 1_000_000
    return input_cost + output_cost


def _usage_summary(result: Any, stage: str) -> dict | None:
    if result is None:
        return None
    usage = getattr(result, "usage", None)
    model_usage = getattr(result, "model_usage", None)
    return {
        "stage": stage,
        "actual_models": (
            sorted(str(model) for model in model_usage)
            if isinstance(model_usage, dict)
            else []
        ),
        "cost_usd": getattr(result, "total_cost_usd", None),
        "num_turns": getattr(result, "num_turns", None),
        "duration_ms": getattr(result, "duration_ms", None),
        "input_tokens": _usage_get(usage, "input_tokens"),
        "output_tokens": _usage_get(usage, "output_tokens"),
        "cache_read_tokens": _usage_get(usage, "cache_read_input_tokens"),
        "cache_creation_tokens": _usage_get(usage, "cache_creation_input_tokens"),
    }


def _report_usage(
    result: Any,
    stage: str,
    requested_model: str | None,
    on_usage: UsageCallback | None,
) -> None:
    summary = _usage_summary(result, stage)
    if summary is None:
        return
    summary["requested_model"] = requested_model
    log.info(
        "agent usage [%s]: requested_model=%s actual_models=%s cost=$%s turns=%s "
        "in=%s out=%s cache_read=%s cache_write=%s",
        stage,
        requested_model,
        summary["actual_models"],
        summary["cost_usd"],
        summary["num_turns"],
        summary["input_tokens"],
        summary["output_tokens"],
        summary["cache_read_tokens"],
        summary["cache_creation_tokens"],
    )
    if on_usage is not None:
        try:
            on_usage(summary)
        except Exception as exc:  # noqa: BLE001 — usage reporting must never break the run
            log.warning("on_usage callback failed: %s", exc)


def _api_attempts() -> int:
    # Total attempts = retries + 1. Retries only fire on transient API errors
    # (429/5xx). Default 1: an executor retry re-runs the whole build, so keep
    # the credit multiplier low; bump CODEBUILDER_AGENT_API_RETRIES if the API
    # is flaky.
    return _env_int("CODEBUILDER_AGENT_API_RETRIES", 1, minimum=0) + 1


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
    return type(message).__name__ == "ResultMessage" or hasattr(
        message, "structured_output"
    )


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


class _Outcome:
    """Carries the last result message + transcript out of one query() pass,
    even when the pass ends by raising (the error result arrives just before the
    subprocess exits non-zero and raises)."""

    def __init__(self) -> None:
        self.result: Any = None
        self.parts: list[str] = []

    @property
    def transcript(self) -> str:
        return "\n".join(self.parts)


async def _drive_once(
    options: ClaudeAgentOptions,
    prompt: str,
    query_fn: Callable[..., Any],
    outcome: _Outcome,
    on_message: ProgressCallback | None,
    budget_usd: float | None = None,
) -> None:
    est_cost = 0.0
    async for message in query_fn(prompt=prompt, options=options):
        if on_message is not None:
            try:
                on_message(message)
            except Exception as exc:  # noqa: BLE001 — progress must never break the run
                log.warning("progress callback failed: %s", exc)
        text = _assistant_text(message)
        if text:
            outcome.parts.append(text)
        if _is_result_message(message):
            outcome.result = message
        elif budget_usd is not None:
            # Accumulate per-turn (non-result) usage; stop before the cap blows.
            est_cost += _estimate_cost_usd(getattr(message, "usage", None))
            if est_cost >= budget_usd:
                raise CCBudgetExceeded(est_cost, outcome.transcript)


async def _run_query(
    *,
    label: str,
    options: ClaudeAgentOptions,
    prompt: str,
    query_fn: Callable[..., Any],
    stderr_lines: list[str],
    on_message: ProgressCallback | None = None,
    on_usage: UsageCallback | None = None,
    budget_usd: float | None = None,
) -> _Outcome:
    """Run query() with bounded retries on transient API errors. Reports usage
    on success and failure. Raises CCBudgetExceeded (not retried) when the cost
    cap trips, else CCAgentError with the real HTTP status / CLI stderr."""
    attempts = _api_attempts()
    for attempt in range(1, attempts + 1):
        outcome = _Outcome()
        try:
            await _drive_once(
                options, prompt, query_fn, outcome, on_message, budget_usd
            )
            _report_usage(outcome.result, label, options.model, on_usage)
            return outcome
        except CCBudgetExceeded:
            raise  # a hard stop — never retried, never wrapped
        except Exception as exc:  # noqa: BLE001 — classify, then retry or surface
            status = getattr(outcome.result, "api_error_status", None)
            if status in _TRANSIENT_API_STATUSES and attempt < attempts:
                delay = min(30, 2**attempt)
                log.warning(
                    "%s: transient API error HTTP %s (attempt %d/%d); retrying in %ds",
                    label,
                    status,
                    attempt,
                    attempts,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            _report_usage(
                outcome.result, label, options.model, on_usage
            )  # surface the wasted spend
            suffix = f" (HTTP {status})" if status else ""
            raise CCAgentError(
                _with_stderr(f"{label} query failed{suffix}: {exc}", stderr_lines)
            ) from exc
    raise CCAgentError(f"{label} query failed after {attempts} attempts")  # unreachable


def _system_prompt(required: str, additional: str | None) -> str:
    return f"{required}\n\n{additional}" if additional else required


async def _run_read_only_structured(
    *,
    label: str,
    output_name: str,
    output_model: type[StructuredOutput],
    cwd: str | Path,
    prompt: str,
    system_prompt: str | None,
    model: str,
    fallback_model: str,
    effort: Effort,
    budget_usd: float | None,
    on_usage: UsageCallback | None,
    query_fn: Callable[..., Any],
) -> StructuredOutput:
    """Run one read-only SDK role and validate its structured result."""
    stderr_lines: list[str] = []
    options = ClaudeAgentOptions(
        cwd=str(cwd),
        model=model,
        fallback_model=fallback_model,
        effort=effort,
        tools=["Read", "Grep", "Glob", "Skill"],
        allowed_tools=["Read", "Grep", "Glob", "Skill"],
        disallowed_tools=["Write", "Edit", "Bash"],
        permission_mode="default",
        setting_sources=["project"],
        skills=SKILLS,
        output_format={
            "type": "json_schema",
            "schema": output_model.model_json_schema(),
        },
        system_prompt=system_prompt,
        stderr=stderr_lines.append,
    )
    outcome = await _run_query(
        label=label,
        options=options,
        prompt=prompt,
        query_fn=query_fn,
        stderr_lines=stderr_lines,
        on_usage=on_usage,
        budget_usd=budget_usd,
    )
    result = outcome.result
    if result is None:
        raise CCAgentError(
            _with_stderr(f"{output_name} produced no result message", stderr_lines)
        )
    if getattr(result, "subtype", None) == "error_max_structured_output_retries":
        raise CCAgentError(
            f"{output_name} exhausted structured-output retries without a valid output"
        )
    data = getattr(result, "structured_output", None)
    if not data:
        raise CCAgentError(
            _with_stderr(f"{output_name} returned no structured output", stderr_lines)
        )
    return output_model.model_validate(data)


async def run_intake(
    *,
    cwd: str | Path,
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    on_usage: UsageCallback | None = None,
    query_fn: Callable[..., Any] = query,
) -> IntakeAssessment:
    """Read-only sufficiency assessment before specification planning."""
    return await _run_read_only_structured(
        label="intake",
        output_name="intake analyst",
        output_model=IntakeAssessment,
        cwd=cwd,
        prompt=prompt,
        system_prompt=_system_prompt(_INTAKE_SYSTEM_PROMPT, system_prompt),
        model=model or PLANNER_MODEL,
        fallback_model=PLANNER_FALLBACK_MODEL,
        effort=PLANNER_EFFORT,
        budget_usd=None,
        on_usage=on_usage,
        query_fn=query_fn,
    )


async def run_planner(
    *,
    cwd: str | Path,
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    on_usage: UsageCallback | None = None,
    query_fn: Callable[..., Any] = query,
) -> Plan:
    """Read-only planning pass. Returns a validated :class:`Plan` via the SDK's
    structured-output mode. Raises :class:`CCAgentError` on schema-retry
    exhaustion or missing output."""
    return await _run_read_only_structured(
        label="planner",
        output_name="planner",
        output_model=Plan,
        cwd=cwd,
        prompt=prompt,
        system_prompt=system_prompt,
        model=model or PLANNER_MODEL,
        fallback_model=PLANNER_FALLBACK_MODEL,
        effort=PLANNER_EFFORT,
        budget_usd=None,
        on_usage=on_usage,
        query_fn=query_fn,
    )


async def run_reviewer(
    *,
    cwd: str | Path,
    prompt: str,
    system_prompt: str | None = None,
    budget_usd: float | None = None,
    on_usage: UsageCallback | None = None,
    query_fn: Callable[..., Any] = query,
) -> ProductionReview:
    """Read-only semantic review against any approved specification."""
    return await _run_read_only_structured(
        label="semantic_reviewer",
        output_name="semantic reviewer",
        output_model=ProductionReview,
        cwd=cwd,
        prompt=prompt,
        system_prompt=_system_prompt(_REVIEWER_SYSTEM_PROMPT, system_prompt),
        model=EXECUTOR_MODEL,
        fallback_model=EXECUTOR_FALLBACK_MODEL,
        effort=EXECUTOR_EFFORT,
        budget_usd=budget_usd,
        on_usage=on_usage,
        query_fn=query_fn,
    )


def _test_author_system_prompt(
    declared_test_files: list[str], additional: str | None
) -> str:
    files: list[str] = []
    for raw in declared_test_files:
        value = raw.strip().replace("\\", "/")
        path = PurePosixPath(value)
        if (
            not value
            or value.startswith("/")
            or (len(value) > 1 and value[1] == ":")
            or ".." in path.parts
            or path.as_posix() == "."
        ):
            raise ValueError(f"unsafe declared test path: {raw!r}")
        normalized = path.as_posix()
        if normalized not in files:
            files.append(normalized)
    if not files:
        raise ValueError("at least one declared test file is required")

    allowlist = "\n".join(f"- {path}" for path in files)
    required = f"""\
You are the test-author stage. Create or edit only the approved test files below:
{allowlist}

Do not modify production source, configuration, documentation, the approved
specification, or tests outside this list. Write tests directly from the
approved success criteria and use every identifier exactly as declared. You may
run diagnostic, syntax, collection, and test commands, but a pre-implementation
behavior failure is expected and must never be hidden by weakening a test. Stop
and report the ambiguity if the approved contract is insufficient."""
    return _system_prompt(required, additional)


def _stage_write_guard(stage: Path):
    root = stage.resolve()

    async def guard(hook_input, _tool_use_id, _context):
        raw = hook_input.get("tool_input", {}).get("file_path")
        if not raw:
            return {}
        path = Path(raw)
        candidate = (root / path if not path.is_absolute() else path).resolve()
        if candidate == root or root in candidate.parents:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"Writes are restricted to the isolated stage: {root}"
                ),
            }
        }

    return guard


async def _run_workspace_agent(
    *,
    label: str,
    cwd: str | Path,
    prompt: str,
    system_prompt: str | None,
    model: str | None,
    effort: Effort | None,
    max_turns: int | None,
    budget_usd: float | None,
    on_message: ProgressCallback | None,
    on_usage: UsageCallback | None,
    query_fn: Callable[..., Any],
) -> str:
    """Run one mutating role in a caller-owned isolated workspace."""
    stderr_lines: list[str] = []
    options = ClaudeAgentOptions(
        cwd=str(cwd),
        model=model or EXECUTOR_MODEL,
        fallback_model=EXECUTOR_FALLBACK_MODEL,
        effort=effort or EXECUTOR_EFFORT,
        tools=[
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "Skill",
        ],
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Skill"],
        permission_mode="bypassPermissions",
        hooks={
            "PreToolUse": [
                HookMatcher(
                    matcher="Write|Edit|MultiEdit|NotebookEdit",
                    hooks=[_stage_write_guard(Path(cwd))],
                )
            ]
        },
        sandbox={
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
        },
        setting_sources=["project"],
        skills=SKILLS,
        system_prompt=system_prompt,
        max_turns=max_turns if max_turns is not None else _executor_max_turns(),
        stderr=stderr_lines.append,
        # AMP runs each job in its own root-owned sandbox. The CLI requires this
        # marker before it accepts bypassPermissions as root.
        env={"IS_SANDBOX": "1"},
    )
    outcome = await _run_query(
        label=label,
        options=options,
        prompt=prompt,
        query_fn=query_fn,
        stderr_lines=stderr_lines,
        on_message=on_message,
        on_usage=on_usage,
        budget_usd=budget_usd,
    )
    if outcome.result is not None and getattr(outcome.result, "is_error", False):
        log.warning(
            "%s result reported an error (subtype=%s); QA will catch bad output",
            label,
            getattr(outcome.result, "subtype", "?"),
        )
    return outcome.transcript


async def run_test_author(
    *,
    cwd: str | Path,
    prompt: str,
    declared_test_files: list[str],
    system_prompt: str | None = None,
    model: str | None = None,
    effort: Effort | None = None,
    max_turns: int | None = None,
    budget_usd: float | None = None,
    on_message: ProgressCallback | None = None,
    on_usage: UsageCallback | None = None,
    query_fn: Callable[..., Any] = query,
) -> str:
    """Write only the approved tests in a caller-enforced staging workspace."""
    return await _run_workspace_agent(
        label="test_author",
        cwd=cwd,
        prompt=prompt,
        system_prompt=_test_author_system_prompt(declared_test_files, system_prompt),
        model=model,
        effort=effort,
        max_turns=max_turns,
        budget_usd=budget_usd,
        on_message=on_message,
        on_usage=on_usage,
        query_fn=query_fn,
    )


async def run_executor(
    *,
    cwd: str | Path,
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    effort: Effort | None = None,
    max_turns: int | None = None,
    budget_usd: float | None = None,
    on_message: ProgressCallback | None = None,
    on_usage: UsageCallback | None = None,
    query_fn: Callable[..., Any] = query,
) -> str:
    """Autonomous build pass. Writes files under ``cwd`` with full tools and
    ``bypassPermissions``. Returns the assistant transcript (the deliverable is
    on disk; the transcript is for logging/progress). ``on_message`` is invoked
    for every streamed message so callers can emit progress events. When
    ``budget_usd`` is set, raises :class:`CCBudgetExceeded` once the estimated
    spend crosses it — the partial build is already on disk."""
    return await _run_workspace_agent(
        label="executor",
        cwd=cwd,
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        effort=effort,
        max_turns=max_turns,
        budget_usd=budget_usd,
        on_message=on_message,
        on_usage=on_usage,
        query_fn=query_fn,
    )
