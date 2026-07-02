"""Light deterministic QA for generated workspaces.

The heavy CrewAI-era gates (per-file deterministic review, symbol-drift mypy,
import-completeness, .env consistency, RPA architecture gate) are gone — the CC
executor writes and self-checks the whole package. What remains is the cheap
insurance the plan calls for: ruff + pytest + uv-installability, plus a minimal
plan validity check.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from codebuilder.schemas import ArtifactRef, Plan, QAReport
from codebuilder.tools import LintRunnerTool, TestRunnerTool
from codebuilder.tools.project_env import ensure_project_env

log = logging.getLogger(__name__)

MAX_QA_OUTPUT_CHARS = 12000

_QA_TEST_SKIP_DIRS = {
    ".git", ".venv", "__pycache__", "build", "dist", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


def is_pass(output: str) -> bool:
    normalized = output.strip()
    return normalized == "PASS" or normalized.startswith("PASS\n")


def is_skip(output: str) -> bool:
    return output.strip().startswith("SKIP:")


def is_no_tests_collected(output: str) -> bool:
    return output.strip().startswith("SKIP: no tests collected")


def truncate(value: str, limit: int = MAX_QA_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n\n[truncated {omitted} chars]"


def _is_test_file(path: str) -> bool:
    p = Path(path)
    return "tests" in p.parts or p.name.startswith("test_") or p.name.endswith("_test.py")


def has_pytest_files(build_dir: str) -> bool:
    root = Path(build_dir)
    if not root.is_dir():
        return False
    for path in root.rglob("*.py"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part in _QA_TEST_SKIP_DIRS for part in rel.parts):
            continue
        if _is_test_file(rel.as_posix()):
            return True
    return False


def _run_scoped_lint(lint_tool, paths: list[str]) -> str:
    """Lint each path individually and aggregate. Any SKIP short-circuits to SKIP
    (required gate unavailable); any failure makes the result a failure. Used by
    patch jobs so pre-existing lint debt in untouched files can't fail QA."""
    failures: list[str] = []
    for path in paths:
        output = lint_tool._run(path)
        if is_skip(output):
            return output
        if not is_pass(output):
            failures.append(output)
    return "\n".join(failures) if failures else "PASS"


def artifact_refs(refs: list[dict] | list[ArtifactRef] | None) -> list[ArtifactRef]:
    converted: list[ArtifactRef] = []
    for ref in refs or []:
        converted.append(ref if isinstance(ref, ArtifactRef) else ArtifactRef(**ref))
    return converted


def validate_plan(plan: Plan | None) -> Plan:
    """Minimal plan sanity check. The executor consumes the Markdown body
    holistically, so all we require is a non-empty plan and a valid mode."""
    if not isinstance(plan, Plan):
        raise ValueError("Planner did not return a valid Plan object.")
    issues: list[str] = []
    if not plan.plan_markdown.strip():
        issues.append("plan_markdown is empty")
    if plan.mode not in ("new_project", "patch_existing"):
        issues.append(f"invalid mode: {plan.mode!r}")
    if issues:
        raise ValueError("Invalid plan: " + "; ".join(issues))
    return plan


def qa_report_for_repair(report: QAReport) -> str:
    """Compact JSON view of a failed QA report for the repair prompt."""
    payload = report.model_dump()
    payload["artifact_urls"] = []
    payload["lint_output"] = truncate(payload.get("lint_output") or "")
    payload["test_output"] = truncate(payload.get("test_output") or "")
    return json.dumps(payload, indent=2, ensure_ascii=False)


def run_final_qa(
    build_dir: str,
    *,
    artifact_urls: list[dict] | list[ArtifactRef] | None = None,
    changed_paths: list[str] | None = None,
    require_installable: bool = False,
    allow_no_tests: bool = False,
) -> QAReport:
    """Ruff + pytest + uv-installability over the build dir.

    Provisions the project's own environment first (``uv sync``) so pytest runs
    with the generated package importable and its dependencies present. With
    ``require_installable`` (new-project jobs) a failed sync IS the QA failure —
    a package that doesn't install is not a working deliverable — and the sync
    output is surfaced for the repair pass. Patch jobs degrade gracefully: the
    user's project may legitimately not be uv-installable, so QA falls back to
    the orchestrator's interpreter. ``allow_no_tests`` turns pytest's "no tests
    collected" into a non-blocking warning (patch jobs against repos with no
    tests).

    ``changed_paths`` (patch mode) scopes **ruff** to just the files the executor
    touched — pre-existing lint debt in untouched customer files must not fail
    the job. ``None`` lints the whole build dir. Pytest always runs over the
    whole dir so a change can't silently break the rest of the suite.
    """
    sync_error = ensure_project_env(build_dir)
    if sync_error and require_installable:
        return QAReport(
            passed=False,
            test_output=truncate(sync_error),
            integration_notes=(
                "Project environment provisioning failed: `uv sync` could not "
                "install the generated project from its pyproject.toml. Fix the "
                "project metadata/dependencies — the sync output is in test_output."
            ),
            artifact_urls=artifact_refs(artifact_urls),
        )

    lint_tool = LintRunnerTool(workspace_dir=build_dir)
    if changed_paths is None:
        lint_output = lint_tool._run(".")
    else:
        py_changed = [p for p in changed_paths if p.endswith((".py", ".pyi"))]
        lint_output = _run_scoped_lint(lint_tool, py_changed) if py_changed else "PASS"
    lint_ok = is_pass(lint_output)

    project_has_tests = has_pytest_files(build_dir)
    test_tool = TestRunnerTool(workspace_dir=build_dir)
    test_output = test_tool._run(".")
    no_tests_warning = (
        allow_no_tests
        and not project_has_tests
        and lint_ok
        and is_no_tests_collected(test_output)
    )
    test_ok = is_pass(test_output) or no_tests_warning

    lint_scope = (
        "the whole build directory" if changed_paths is None
        else f"{len([p for p in changed_paths if p.endswith(('.py', '.pyi'))])} changed file(s)"
    )
    notes = [f"Deterministic QA ran ruff over {lint_scope} and pytest over the build directory."]
    if is_skip(lint_output):
        notes.append(f"Lint was not executed: {lint_output}")
    if no_tests_warning:
        notes.append("No pytest tests were collected; QA validated with ruff only.")
    elif is_skip(test_output):
        notes.append(f"Tests were not executed: {test_output}")
    if not lint_ok and not is_skip(lint_output):
        notes.append("Lint failed.")
    if not test_ok and not is_skip(test_output):
        notes.append("Tests failed.")

    return QAReport(
        passed=lint_ok and test_ok,
        lint_output=lint_output,
        test_output=test_output,
        integration_notes=" ".join(notes),
        artifact_urls=artifact_refs(artifact_urls),
    )
