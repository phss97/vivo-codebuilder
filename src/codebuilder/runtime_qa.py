"""Deterministic runtime review and QA gates for generated workspaces."""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codebuilder.schemas import (
    ArtifactRef,
    CodeArtifact,
    Plan,
    QAReport,
    ReviewResult,
    SubTask,
)
from codebuilder.tools import LintRunnerTool, TestRunnerTool
from codebuilder.tools.workspace_tool import resolve_within

log = logging.getLogger(__name__)

MAX_QA_OUTPUT_CHARS = 12000

_TODO_TOKEN_RE = re.compile(r"\b(todo|fixme|placeholder|stub)\b", re.IGNORECASE)
_PLACEHOLDER_LINE_RE = re.compile(
    r"(pass|\.\.\.|raise\s+NotImplementedError(?:\([^)]*\))?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DeterministicReview:
    result: ReviewResult
    needs_fallback: bool = False


def is_pass(output: str) -> bool:
    normalized = output.strip()
    return normalized == "PASS" or normalized.startswith("PASS\n")


def is_skip(output: str) -> bool:
    return output.strip().startswith("SKIP:")


def truncate(value: str, limit: int = MAX_QA_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n\n[truncated {omitted} chars]"


def _is_test_file(path: str) -> bool:
    p = Path(path)
    return "tests" in p.parts or p.name.startswith("test_") or p.name.endswith("_test.py")


def looks_like_placeholder(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    if _TODO_TOKEN_RE.search(stripped):
        return True

    meaningful_lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not meaningful_lines:
        return True
    return all(_PLACEHOLDER_LINE_RE.fullmatch(line) for line in meaningful_lines)


def artifact_refs(refs: list[dict] | list[ArtifactRef] | None) -> list[ArtifactRef]:
    converted: list[ArtifactRef] = []
    for ref in refs or []:
        converted.append(ref if isinstance(ref, ArtifactRef) else ArtifactRef(**ref))
    return converted


def persist_artifact(artifact: CodeArtifact, build_dir: str) -> str:
    """Ensure ``artifact.content`` lives on disk under ``build_dir``."""
    try:
        target = resolve_within(build_dir, artifact.file_path)
    except ValueError as exc:
        return str(exc)

    if artifact.content:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact.content, encoding="utf-8")
        return ""

    if target.is_file():
        artifact.content = target.read_text(encoding="utf-8", errors="replace")
        return ""

    return (
        f"Writer returned empty content and did not write {artifact.file_path}; "
        "include the full file text in CodeArtifact.content."
    )


def validate_plan(plan: Plan | None) -> Plan:
    if not isinstance(plan, Plan):
        raise ValueError("Planner did not return a valid Plan object.")

    issues: list[str] = []
    if not 1 <= len(plan.subtasks) <= 60:
        issues.append("plan must contain between 1 and 60 subtasks")
    for subtask in plan.subtasks:
        if not subtask.file_path.strip():
            issues.append(f"subtask {subtask.id} has an empty file_path")
        if not subtask.test_criteria.strip():
            issues.append(f"subtask {subtask.id} has empty test_criteria")
    if issues:
        raise ValueError("Invalid plan: " + "; ".join(issues))
    return plan


def plan_summary(plan: Plan | None) -> str:
    if not plan:
        return "(no plan available)"
    return json.dumps(
        {
            "project_name": plan.project_name,
            "mode": plan.mode,
            "tech_stack": plan.tech_stack,
            "subtasks": [
                {
                    "id": s.id,
                    "title": s.title,
                    "file_path": s.file_path,
                    "test_criteria": s.test_criteria,
                }
                for s in plan.subtasks
            ],
        },
        indent=2,
    )


def qa_report_for_repair(report: QAReport) -> str:
    payload = report.model_dump()
    payload["lint_output"] = truncate(payload.get("lint_output") or "")
    payload["test_output"] = truncate(payload.get("test_output") or "")
    return json.dumps(payload, indent=2)


def run_deterministic_review(
    subtask: SubTask,
    artifact: CodeArtifact,
    build_dir: str,
    *,
    existing_snapshot: str = "",
    lint_runner: Any | None = None,
    test_runner: Any | None = None,
) -> DeterministicReview:
    """Review an artifact with local path, content, lint, and test checks."""
    issues: list[str] = []
    suggestions: list[str] = []

    if artifact.subtask_id != subtask.id:
        issues.append(
            f"Artifact subtask_id '{artifact.subtask_id}' does not match planned subtask '{subtask.id}'."
        )
    if artifact.file_path != subtask.file_path:
        issues.append(
            f"Artifact file_path '{artifact.file_path}' does not match planned path '{subtask.file_path}'."
        )

    try:
        target = resolve_within(build_dir, artifact.file_path)
    except ValueError as exc:
        issues.append(str(exc))
        return DeterministicReview(
            ReviewResult(subtask_id=subtask.id, passed=False, issues=issues)
        )

    actual = ""
    if not target.is_file():
        issues.append(f"Artifact file was not written to workspace: {artifact.file_path}")
    else:
        actual = target.read_text(encoding="utf-8", errors="replace")
        if artifact.content and actual != artifact.content:
            issues.append(
                f"Workspace file '{artifact.file_path}' does not match CodeArtifact.content."
            )

    if (
        subtask.change_type == "modify"
        and existing_snapshot
        and actual
        and actual.strip() == existing_snapshot.strip()
    ):
        issues.append(
            f"Modify subtask '{subtask.id}' produced no change: "
            f"'{artifact.file_path}' is identical to the pre-existing file. "
            "Re-read the file, apply the described transformation, and return the modified content."
        )

    content_to_check = actual or artifact.content
    if looks_like_placeholder(content_to_check):
        issues.append("Artifact content is empty or contains placeholder/TODO-only output.")

    if not issues:
        lint_tool = lint_runner or LintRunnerTool(workspace_dir=build_dir)
        lint_output = lint_tool._run(artifact.file_path)
        if is_skip(lint_output):
            issues.append(f"required quality gate skipped for {artifact.file_path}: {lint_output}")
        elif not is_pass(lint_output):
            issues.append(f"ruff failed for {artifact.file_path}:\n{lint_output}")

    if not issues and _is_test_file(artifact.file_path):
        test_tool = test_runner or TestRunnerTool(workspace_dir=build_dir)
        test_output = test_tool._run(artifact.file_path)
        if is_skip(test_output):
            issues.append(f"required quality gate skipped for {artifact.file_path}: {test_output}")
        elif not is_pass(test_output):
            issues.append(f"pytest failed for {artifact.file_path}:\n{test_output}")

    if issues:
        return DeterministicReview(
            ReviewResult(subtask_id=subtask.id, passed=False, issues=issues, suggestions=suggestions)
        )

    return DeterministicReview(
        ReviewResult(
            subtask_id=subtask.id,
            passed=True,
            suggestions=["Deterministic path, content, lint, and test checks passed."],
        )
    )


def run_final_qa(
    build_dir: str,
    *,
    artifact_urls: list[dict] | list[ArtifactRef] | None = None,
    lint_runner: Any | None = None,
    test_runner: Any | None = None,
) -> QAReport:
    """Build the final QA report from required workspace lint and tests."""
    lint_tool = lint_runner or LintRunnerTool(workspace_dir=build_dir)
    test_tool = test_runner or TestRunnerTool(workspace_dir=build_dir)

    lint_output = lint_tool._run(".")
    test_output = test_tool._run(".")

    lint_ok = is_pass(lint_output)
    test_ok = is_pass(test_output)

    notes = ["Deterministic QA ran ruff check and pytest over the whole workspace."]
    if is_skip(lint_output):
        notes.append(f"Lint was not executed: {lint_output}")
    if is_skip(test_output):
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


def _package_dirs(build_dir: Path) -> list[Path]:
    src_dir = build_dir / "src"
    if not src_dir.is_dir():
        return []
    return sorted(
        child
        for child in src_dir.iterdir()
        if child.is_dir() and (child / "__init__.py").is_file()
    )


def _has_path_fragment(build_dir: Path, fragment: str) -> bool:
    needle = fragment.lower()
    return any(needle in path.as_posix().lower() for path in build_dir.rglob("*"))


def _pyproject_text(build_dir: Path) -> str:
    path = build_dir / "pyproject.toml"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").lower()


def run_rpa_deterministic_gate(build_dir: str) -> ReviewResult:
    """Deterministic structure check for RPA projects."""
    root = Path(build_dir)
    issues: list[str] = []

    pyproject = _pyproject_text(root)
    if not pyproject:
        issues.append("Missing pyproject.toml for a distributable Python package.")
    else:
        for expected in ("3.13", "ruff", "pytest", "pytest-cov", "mypy", "hatchling"):
            if expected not in pyproject:
                issues.append(f"pyproject.toml must declare RPA baseline dependency/config: {expected}.")

    if not (root / ".env.example").is_file():
        issues.append("Missing .env.example for runtime configuration and CCM integration.")

    packages = _package_dirs(root)
    if not packages:
        issues.append("Missing src/<package>/ package layout.")

    for package in packages[:1]:
        for layer in ("domain", "application", "infrastructure"):
            if not (package / layer).is_dir():
                issues.append(f"Missing Clean Architecture layer: {package.relative_to(root)}/{layer}.")

    for component in ("orchestrator", "producer", "consumer"):
        if not _has_path_fragment(root, component):
            issues.append(f"Missing RPA {component} component.")

    if not _has_path_fragment(root, "config"):
        issues.append("Missing configuration module for .env/CCM integration.")
    if not _has_path_fragment(root, "logging"):
        issues.append("Missing logging/traceability module.")

    test_files = list((root / "tests").rglob("test_*.py")) if (root / "tests").is_dir() else []
    if not test_files:
        issues.append("Missing pytest tests for generated business behavior.")

    return ReviewResult(
        subtask_id="architecture_gate",
        passed=not issues,
        issues=issues,
        suggestions=[] if issues else ["RPA architecture gate passed."],
    )


def _rpa_full_gate(build_dir: str, plan: Plan | None) -> ReviewResult:
    """RPA deterministic check + LLM reviewer pass."""
    from codebuilder.crews.reviewer_crew import ReviewerCrew
    from codebuilder.tools.workspace_tool import WorkspaceListTool

    deterministic = run_rpa_deterministic_gate(build_dir)
    if not deterministic.passed:
        return deterministic

    listing_tool = WorkspaceListTool(workspace_dir=build_dir)
    try:
        result = ReviewerCrew(workspace_dir=build_dir).architecture_gate_crew().kickoff(
            inputs={
                "workspace_dir": build_dir,
                "workspace_listing": listing_tool._run("."),
                "plan_summary": plan_summary(plan),
                "domain": "rpa",
            }
        )
    except Exception as exc:  # noqa: BLE001 - acceptance gate failure should be visible
        return ReviewResult(
            subtask_id="architecture_gate",
            passed=False,
            issues=[f"Architecture gate reviewer failed: {exc}"],
        )

    if isinstance(result.pydantic, ReviewResult):
        return result.pydantic
    return ReviewResult(
        subtask_id="architecture_gate",
        passed=False,
        issues=["Architecture gate reviewer did not return a valid ReviewResult."],
    )


# Registry of domain slug → architecture gate. Add entries here when a new
# domain skill (e.g. "python-package", "flask-api") needs its own structural
# acceptance check. Each gate runs its own deterministic + LLM passes and
# returns a ReviewResult with subtask_id="architecture_gate".
_ARCHITECTURE_GATES: dict[str, Any] = {
    "rpa": _rpa_full_gate,
}


def _own_top_packages(build_dir: Path) -> list[str]:
    """Return top-level Python package names that belong to *this* project.

    Looks under ``build_dir/src/<pkg>/__init__.py`` first (src layout) and
    falls back to ``build_dir/<pkg>/__init__.py`` for flat layouts. Anything
    outside this list is treated as external by the import-completeness gate.
    """
    candidates: list[str] = []
    src_dir = build_dir / "src"
    roots = [src_dir, build_dir] if src_dir.is_dir() else [build_dir]
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and (child / "__init__.py").is_file():
                candidates.append(child.name)
    return candidates


def _resolve_local_module(
    module: str, build_dir: Path, own_packages: set[str]
) -> Path | None:
    """Return the candidate file path for a module if it should resolve here.

    Returns ``None`` when the module's top-level package is not one of the
    project's own packages (stdlib / third-party / external lib reference).
    Otherwise returns the *expected* file Path even if it does not exist —
    the caller decides what to do with a missing file.
    """
    if not module:
        return None
    top = module.split(".")[0]
    if top not in own_packages:
        return None
    rel = Path(*module.split("."))
    src_dir = build_dir / "src"
    for root in (src_dir, build_dir):
        if not root.is_dir():
            continue
        candidate_file = (root / rel).with_suffix(".py")
        candidate_pkg = root / rel / "__init__.py"
        if candidate_file.is_file() or candidate_pkg.is_file():
            return candidate_file if candidate_file.exists() else candidate_pkg
        if (root / top).is_dir():
            return candidate_file
    return None


def run_import_completeness_gate(
    build_dir: str, plan: Plan | None
) -> tuple[list[str], list[SubTask]]:
    """Detect project-local imports that point to modules that were never generated.

    Walks every ``.py`` under ``build_dir`` and parses imports with ``ast``.
    Only flags imports whose top-level package matches one of the project's
    own packages AND that does not appear in ``plan.external_packages``.
    Returns ``(missing_paths, stub_subtasks)`` where ``stub_subtasks`` has at
    most :data:`_MAX_STUB_SUBTASKS` entries the caller can feed back through
    ``_build_subtask``.
    """
    root = Path(build_dir)
    if not root.is_dir():
        return [], []

    own_packages = set(_own_top_packages(root))
    if not own_packages:
        return [], []

    external = {pkg for pkg in (plan.external_packages if plan else []) if pkg}
    own_packages -= external

    missing: dict[str, set[str]] = {}
    for py_file in root.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.level:
                continue  # relative imports — skip, they resolve at runtime
            resolved = _resolve_local_module(node.module, root, own_packages)
            if resolved is None or resolved.exists():
                continue
            try:
                rel_path = resolved.relative_to(root)
            except ValueError:
                continue
            symbols = {alias.name for alias in node.names if alias.name != "*"}
            missing.setdefault(str(rel_path), set()).update(symbols)

    if not missing:
        return [], []

    missing_paths = sorted(missing.keys())
    stub_subtasks: list[SubTask] = []
    for i, rel_path in enumerate(missing_paths):
        if i >= _MAX_STUB_SUBTASKS:
            break
        symbols = sorted(missing[rel_path])
        symbol_list = ", ".join(symbols) if symbols else "(no specific symbols requested)"
        stub_subtasks.append(
            SubTask(
                id=f"stub_{i:02d}",
                title=f"Generate missing module {rel_path}",
                description=(
                    f"Module '{rel_path}' is imported by other project files but was "
                    f"never generated. Create it now with the symbols importers expect: "
                    f"{symbol_list}."
                ),
                file_path=rel_path,
                change_type="create",
                tech_notes=(
                    f"Required exported symbols: {symbol_list}. "
                    "Use workspace_read on importers (search the workspace) "
                    "to confirm signatures before writing."
                ),
                test_criteria=(
                    "File exists, exports the named symbols, passes ruff, and "
                    "does not cause ModuleNotFoundError at pytest collection."
                ),
            )
        )

    return missing_paths, stub_subtasks


_MAX_STUB_SUBTASKS = 8


def run_full_architecture_gate(build_dir: str, plan: Plan | None) -> ReviewResult:
    """Dispatch to the architecture gate registered for ``plan.domain``.

    Returns a pass-through ReviewResult when no domain gate matches, so
    projects outside the registered domains finalize on lint/test only.
    """
    domain = (plan.domain if plan else "") or ""
    gate = _ARCHITECTURE_GATES.get(domain)
    if gate is None:
        suggestion = (
            f"No architecture gate registered for domain {domain!r}."
            if domain
            else "Plan did not declare a domain; skipping domain architecture gate."
        )
        return ReviewResult(
            subtask_id="architecture_gate",
            passed=True,
            suggestions=[suggestion],
        )
    return gate(build_dir, plan)
