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
    MAX_FILES_PER_WORK_PACKAGE,
    MAX_WORK_PACKAGES,
    ArtifactRef,
    CodeBundleArtifact,
    CodeArtifact,
    FileSkeleton,
    Plan,
    QAReport,
    ReviewResult,
    SubTask,
)
from codebuilder.tools import LintRunnerTool, TestRunnerTool, TypeCheckRunnerTool
from codebuilder.tools.project_env import ensure_project_env
from codebuilder.tools.workspace_tool import resolve_within

log = logging.getLogger(__name__)

MAX_QA_OUTPUT_CHARS = 12000
# Largest preimage of a `modify` target shown to the writer in full. Files above
# this are shown truncated (with a marker); the review guards below then forbid
# rewriting such a file shorter, so its unshown tail can't be silently dropped.
MODIFY_PREIMAGE_LIMIT = 60_000

_TODO_TOKEN_RE = re.compile(r"\b(todo|fixme|placeholder|stub)\b", re.IGNORECASE)
# Matches the truncation markers emitted by WorkspaceReadTool and the preimage
# injector; a persisted artifact carrying one means a truncated read was written
# back as the file.
_TRUNCATION_MARKER_RE = re.compile(r"\[truncated \d+")
_PLACEHOLDER_LINE_RE = re.compile(
    r"(pass|\.\.\.|raise\s+NotImplementedError(?:\([^)]*\))?)",
    re.IGNORECASE,
)
_PLACEHOLDER_PLAN_RE = re.compile(
    r"(^|[\W_])(files_to_be_determined_by|tests_to_be_determined_by|tbd|placeholder)([\W_]|$)",
    re.IGNORECASE,
)
_REAL_TARGET_NAMES = {
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "build.spec",
    "build.ps1",
    "build.bat",
    ".env.example",
}
_QA_TEST_SKIP_DIRS = {
    ".git", ".venv", "__pycache__", "build", "dist", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


@dataclass(frozen=True)
class DeterministicReview:
    result: ReviewResult
    needs_fallback: bool = False


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


_INDEX_SKIP_DIRS = {
    ".git", ".venv", "__pycache__", "build", "dist", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "tests",
}


def _arg_names(fn: ast.FunctionDef | ast.AsyncFunctionDef, *, skip_self: bool = False) -> list[str]:
    args = fn.args
    names = [a.arg for a in (args.posonlyargs + args.args)]
    if skip_self and names and names[0] in ("self", "cls"):
        names = names[1:]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return names


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 — best-effort rendering
        return getattr(node, "id", "")


def _func_sig(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    ret = f" -> {_unparse(fn.returns)}" if fn.returns is not None else ""
    return "(" + ", ".join(_arg_names(fn)) + ")" + ret


def _class_fields(cls: ast.ClassDef) -> list[str]:
    """Real attribute names: class-level annotations/assignments plus
    ``self.x`` assignments in ``__init__`` (pydantic models, dataclasses, and
    hand-rolled classes all surface their fields this way)."""
    fields: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name and not name.startswith("__") and name not in seen:
            seen.add(name)
            fields.append(name)

    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            _add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    _add(target.id)
    for stmt in cls.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                    targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                    for target in targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and not target.attr.startswith("_")
                        ):
                            _add(target.attr)
    return fields


def extract_module_api(source: str) -> str:
    """Extract a compact, *real* public-API summary from Python source via AST.

    Returns one line per top-level public class (with its field/attr names and
    ``__init__`` parameters) and public function (with signature), plus
    module-level public constants. Private (leading-underscore) names are
    omitted. Returns ``""`` when there is no public API or the source cannot be
    parsed — drift prevention is best-effort and must never raise.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ""
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            bases = ", ".join(_unparse(b) for b in node.bases)
            header = f"class {node.name}" + (f"({bases})" if bases else "")
            detail: list[str] = []
            fields = _class_fields(node)
            if fields:
                detail.append("fields=[" + ", ".join(fields) + "]")
            init = next(
                (
                    _arg_names(s, skip_self=True)
                    for s in node.body
                    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == "__init__"
                ),
                [],
            )
            if init:
                detail.append("__init__(" + ", ".join(init) + ")")
            methods = [
                s.name
                for s in node.body
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not s.name.startswith("_")
            ]
            if methods:
                detail.append("methods=[" + ", ".join(methods) + "]")
            lines.append(header + (": " + "; ".join(detail) if detail else ""))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                lines.append("def " + node.name + _func_sig(node))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                lines.append(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    lines.append(target.id)
    return "\n".join(lines)


def _module_dotted_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if not parts:
        return ""
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def build_symbol_index(build_dir: str, *, paths: list[str] | None = None) -> dict[str, str]:
    """Map each already-written project module (dotted path) → its real public
    API summary, extracted from disk.

    Grounds writers/repair in the *actual* code (real field names, constructor
    signatures, function signatures) instead of the planner's declared
    ``public_api`` — which carries top-level names but not class fields, the gap
    that lets a caller invent ``settings.sap_host``. ``paths`` (relative to
    ``build_dir``) scopes the walk; ``None`` walks the whole tree minus tests
    and generated dirs. Best-effort: unparseable files are skipped.
    """
    root = Path(build_dir)
    if not root.is_dir():
        return {}
    if paths is not None:
        files = [root / p for p in paths]
    else:
        files = [
            p
            for p in root.rglob("*.py")
            if not any(part in _INDEX_SKIP_DIRS for part in p.relative_to(root).parts)
        ]
    index: dict[str, str] = {}
    for path in files:
        if not path.is_file() or path.suffix != ".py":
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        api = extract_module_api(source)
        if not api:
            continue
        module = _module_dotted_path(path, root)
        if module:
            index[module] = api
    return index


def _is_basesettings_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if name == "BaseSettings":
            return True
    return False


def _settings_env_prefix(cls: ast.ClassDef) -> str:
    """Extract ``env_prefix`` from ``model_config = SettingsConfigDict(...)`` or
    a nested ``class Config``. Returns ``""`` when none is declared."""
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            if "model_config" in targets:
                for kw in stmt.value.keywords:
                    if kw.arg == "env_prefix" and isinstance(kw.value, ast.Constant):
                        return str(kw.value.value)
        if isinstance(stmt, ast.ClassDef) and stmt.name == "Config":
            for sub in stmt.body:
                if (
                    isinstance(sub, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "env_prefix" for t in sub.targets)
                    and isinstance(sub.value, ast.Constant)
                ):
                    return str(sub.value.value)
    return ""


def _settings_required_fields(cls: ast.ClassDef) -> list[str]:
    """Required (no-default) field names of a settings class — these are the
    env vars an operator MUST provide. Fields with a default are optional."""
    required: list[str] = []
    for stmt in cls.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        name = stmt.target.id
        if name == "model_config" or name.startswith("_"):
            continue
        annotation = stmt.annotation
        is_classvar = (isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name)
                       and annotation.value.id == "ClassVar") or (
            isinstance(annotation, ast.Name) and annotation.id == "ClassVar")
        if is_classvar:
            continue
        if stmt.value is None:  # no default → required
            required.append(name)
    return required


def _parse_env_example_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key.lower().startswith("export "):
            key = key[len("export "):].strip()
        if key:
            keys.add(key.upper())
    return keys


def check_env_example_consistency(build_dir: str) -> list[str]:
    """Flag required Settings fields whose env var is absent from .env.example.

    Catches the "config won't load even when filled in" symptom: a generated
    ``.env.example`` whose keys don't match the env vars pydantic-settings
    reads (wrong names, or missing the ``env_prefix``). Comparison is
    case-insensitive. Returns ``[]`` when there is no ``.env.example`` or no
    ``BaseSettings`` subclass — best-effort, never raises.
    """
    root = Path(build_dir)
    env_file = root / ".env.example"
    if not env_file.is_file():
        return []
    try:
        declared = _parse_env_example_keys(env_file.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return []

    issues: list[str] = []
    seen: set[str] = set()
    for py_file in root.rglob("*.py"):
        if any(part in _INDEX_SKIP_DIRS for part in py_file.relative_to(root).parts):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or not _is_basesettings_class(node):
                continue
            prefix = _settings_env_prefix(node)
            for field in _settings_required_fields(node):
                key = f"{prefix}{field}".upper()
                if key not in declared and key not in seen:
                    seen.add(key)
                    issues.append(
                        f".env.example is missing `{key}` (required by Settings field "
                        f"`{field}` in {py_file.relative_to(root).as_posix()})."
                    )
    return issues


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


def _bundle_path_issues(bundle: CodeBundleArtifact, subtask: SubTask) -> list[str]:
    issues: list[str] = []
    if bundle.subtask_id != subtask.id:
        issues.append(
            f"Bundle subtask_id '{bundle.subtask_id}' does not match planned subtask '{subtask.id}'."
        )

    expected_paths = [f.path for f in subtask.files]
    actual_paths = [a.file_path for a in bundle.artifacts]
    expected_set = set(expected_paths)
    actual_set = set(actual_paths)

    duplicate_expected = sorted({p for p in expected_paths if expected_paths.count(p) > 1})
    duplicate_actual = sorted({p for p in actual_paths if actual_paths.count(p) > 1})
    for path in duplicate_expected:
        issues.append(f"duplicate planned file path: {path}")
    for path in duplicate_actual:
        issues.append(f"duplicate artifact path: {path}")
    for path in sorted(expected_set - actual_set):
        issues.append(f"missing planned artifact: {path}")
    for path in sorted(actual_set - expected_set):
        issues.append(f"unexpected artifact path: {path}")

    for artifact in bundle.artifacts:
        if artifact.subtask_id != subtask.id:
            issues.append(
                f"Artifact {artifact.file_path} subtask_id '{artifact.subtask_id}' "
                f"does not match planned subtask '{subtask.id}'."
            )
    return issues


def persist_bundle_artifact(
    bundle: CodeBundleArtifact,
    subtask: SubTask,
    build_dir: str,
) -> list[str]:
    """Persist a writer bundle only when its paths exactly match the plan."""
    issues = _bundle_path_issues(bundle, subtask)
    if issues:
        return issues

    errors: list[str] = []
    for artifact in bundle.artifacts:
        error = persist_artifact(artifact, build_dir)
        if error:
            errors.append(error)
    return errors


def validate_plan(plan: Plan | None) -> Plan:
    if not isinstance(plan, Plan):
        raise ValueError("Planner did not return a valid Plan object.")

    issues: list[str] = []
    if _PLACEHOLDER_PLAN_RE.search(plan.project_name):
        issues.append(f"project_name contains placeholder text: {plan.project_name}")
    if not 1 <= len(plan.subtasks) <= MAX_WORK_PACKAGES:
        issues.append(f"plan must contain between 1 and {MAX_WORK_PACKAGES} work packages")
    seen_paths: set[str] = set()
    real_targets = 0
    for subtask in plan.subtasks:
        for label, value in (
            ("id", subtask.id),
            ("title", subtask.title),
        ):
            if _PLACEHOLDER_PLAN_RE.search(value):
                issues.append(f"subtask {subtask.id} {label} contains placeholder text: {value}")
        if not subtask.files:
            issues.append(f"subtask {subtask.id} must contain at least one file")
        if len(subtask.files) > MAX_FILES_PER_WORK_PACKAGE:
            issues.append(
                f"subtask {subtask.id} contains {len(subtask.files)} files; "
                f"work packages may contain at most {MAX_FILES_PER_WORK_PACKAGE} files"
            )
        for planned_file in subtask.files:
            path = planned_file.path.strip()
            if not path:
                issues.append(f"subtask {subtask.id} has an empty file path")
            if _PLACEHOLDER_PLAN_RE.search(path):
                issues.append(f"subtask {subtask.id} path contains placeholder text: {path}")
            if "*" in path or "?" in path:
                issues.append(f"subtask {subtask.id} path must be concrete, not a wildcard: {path}")
            if not planned_file.purpose.strip():
                issues.append(f"subtask {subtask.id} file {planned_file.path!r} has empty purpose")
            if planned_file.path in seen_paths:
                issues.append(f"duplicate planned file path: {planned_file.path}")
            seen_paths.add(planned_file.path)
            p = Path(path)
            if p.suffix in {".py", ".pyi"} or p.name in _REAL_TARGET_NAMES:
                real_targets += 1
        if not subtask.test_criteria.strip():
            issues.append(f"subtask {subtask.id} has empty test_criteria")
        if _PLACEHOLDER_PLAN_RE.search(subtask.test_criteria):
            issues.append(f"subtask {subtask.id} test_criteria contains placeholder text")
    codeish = plan.domain == "rpa" or any(
        token in " ".join(plan.tech_stack).lower()
        for token in ("python", "rpa", "code", "pytest", "mypy", "ruff")
    )
    if codeish and real_targets == 0:
        issues.append("diagnostic-only plan has no real production, test, or build target files")
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
                    "files": [f.model_dump() for f in s.files],
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
    payload["type_output"] = truncate(payload.get("type_output") or "")
    return json.dumps(payload, indent=2)


def run_deterministic_review(
    subtask: SubTask,
    artifact: CodeArtifact,
    build_dir: str,
    *,
    planned_file: FileSkeleton | None = None,
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
    if planned_file is None:
        matching = [f for f in subtask.files if f.path == artifact.file_path]
        planned_file = matching[0] if matching else (subtask.files[0] if len(subtask.files) == 1 else None)

    if planned_file is None:
        issues.append(
            f"Artifact file_path '{artifact.file_path}' is not one of planned paths: "
            f"{', '.join(subtask.file_paths)}."
        )
    elif artifact.file_path != planned_file.path:
        issues.append(
            f"Artifact file_path '{artifact.file_path}' does not match planned path '{planned_file.path}'."
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
        planned_file is not None
        and planned_file.change_type == "modify"
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

    # Data-loss guards: a truncated read written back as the file, or a large
    # file rewritten shorter than its preimage (whose tail the writer never saw).
    if _TRUNCATION_MARKER_RE.search(content_to_check):
        issues.append(
            f"Artifact '{artifact.file_path}' contains a truncation marker — a truncated "
            "read was written as the file. Return the COMPLETE file content."
        )
    if (
        planned_file is not None
        and planned_file.change_type == "modify"
        and existing_snapshot
        and actual
        and len(existing_snapshot) > MODIFY_PREIMAGE_LIMIT
        and len(actual) < len(existing_snapshot)
    ):
        issues.append(
            f"Modify target '{artifact.file_path}' is large ({len(existing_snapshot)} chars) but the "
            f"returned file is shorter ({len(actual)} chars); content may have been lost. Make "
            "targeted edits that preserve the rest of the file instead of rewriting it wholesale."
        )

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


def run_bundle_deterministic_review(
    subtask: SubTask,
    bundle: CodeBundleArtifact,
    build_dir: str,
    *,
    existing_snapshots: dict[str, str] | None = None,
    lint_runner: Any | None = None,
    test_runner: Any | None = None,
) -> DeterministicReview:
    """Review every artifact in a bundled work package."""
    issues = _bundle_path_issues(bundle, subtask)
    if issues:
        return DeterministicReview(
            ReviewResult(subtask_id=subtask.id, passed=False, issues=issues)
        )

    planned_by_path = {f.path: f for f in subtask.files}
    snapshots = existing_snapshots or {}
    suggestions: list[str] = []
    for artifact in bundle.artifacts:
        review = run_deterministic_review(
            subtask,
            artifact,
            build_dir,
            planned_file=planned_by_path[artifact.file_path],
            existing_snapshot=snapshots.get(artifact.file_path, ""),
            lint_runner=lint_runner,
            test_runner=test_runner,
        )
        if not review.result.passed:
            issues.extend(review.result.issues)
        suggestions.extend(review.result.suggestions)

    if issues:
        return DeterministicReview(
            ReviewResult(subtask_id=subtask.id, passed=False, issues=issues)
        )
    return DeterministicReview(
        ReviewResult(
            subtask_id=subtask.id,
            passed=True,
            suggestions=suggestions
            or ["Deterministic bundle path, content, lint, and test checks passed."],
        )
    )


def _run_scoped_lint(lint_tool: Any, lint_paths: list[str]) -> str:
    """Lint each path individually and aggregate into one PASS/SKIP/report string.

    Used by patch jobs so pre-existing lint debt in files the writer never
    touched cannot fail final QA. Any SKIP makes the whole result a SKIP
    (required gate unavailable); any failure report makes it a failure.
    """
    failures: list[str] = []
    for path in lint_paths:
        output = lint_tool._run(path)
        if is_skip(output):
            return output
        if not is_pass(output):
            failures.append(output)
    if failures:
        return "\n".join(failures)
    return "PASS"


def _run_scoped_tests(test_tool: Any, test_paths: list[str]) -> str:
    """Run pytest against selected paths and aggregate the result.

    Patch jobs should not pay for the whole customer suite by default. Any
    non-PASS output is kept so the final-QA repair prompt still gets the exact
    failing collection/test text.
    """
    failures: list[str] = []
    for path in test_paths:
        output = test_tool._run(path)
        if not is_pass(output):
            failures.append(output)
    if failures:
        return "\n".join(failures)
    return "PASS"


_TYPE_GATE_CODES = ("attr-defined", "call-arg", "name-defined", "arg-type")


def filter_type_errors(output: str, codes: tuple[str, ...] = _TYPE_GATE_CODES) -> str:
    """Keep only mypy error lines tagged with a high-confidence drift code.

    The gate fails on cross-file symbol drift (``attr-defined``, ``call-arg``,
    ``name-defined``, ``arg-type``) — exactly the bugs ruff and the
    file-existence import gate miss. Annotation-completeness / untyped-dependency
    noise (``no-untyped-def``, ``import``, ``assignment``) is dropped so the gate
    never bricks a job on pre-existing issues the writer never introduced.
    """
    if not output:
        return ""
    kept = [
        line
        for line in output.splitlines()
        if ": error:" in line and any(f"[{code}]" in line for code in codes)
    ]
    return "\n".join(kept)


def _default_type_targets(build_dir: str) -> list[str]:
    """Type-check own packages plus generated tests, else the build dir."""
    root = Path(build_dir)
    targets = [p.relative_to(root).as_posix() for p in _package_dirs(root)]
    if has_pytest_files(build_dir) and (root / "tests").is_dir():
        targets.append("tests")
    if targets:
        return targets
    return ["."]


def _run_type_gate(
    type_tool: Any, targets: list[str], codes: tuple[str, ...] = _TYPE_GATE_CODES
) -> tuple[bool, str]:
    """Run mypy over ``targets`` → ``(passed, filtered_errors)``.

    A SKIP (mypy unavailable — e.g. a patch target that doesn't declare it) is
    non-blocking: the job still finalizes on lint + tests. Only the gated drift
    codes can fail the gate.
    """
    raws: list[str] = []
    for target in targets:
        out = type_tool._run(target)
        if is_skip(out):
            return True, ""
        if out and not is_pass(out):
            raws.append(out)
    errors = filter_type_errors("\n".join(raws), codes)
    return (not errors), errors


def run_final_qa(
    build_dir: str,
    *,
    artifact_urls: list[dict] | list[ArtifactRef] | None = None,
    lint_paths: list[str] | None = None,
    test_paths: list[str] | None = None,
    type_paths: list[str] | None = None,
    lint_runner: Any | None = None,
    test_runner: Any | None = None,
    type_runner: Any | None = None,
    require_installable: bool = False,
    allow_no_tests: bool = False,
    skip_pytest_on_deterministic_failure: bool = False,
) -> QAReport:
    """Build the final QA report from required workspace lint and tests.

    ``lint_paths`` scopes ruff to specific files (patch jobs lint only what
    the writer created/modified); ``None`` lints the whole build dir.
    ``test_paths`` scopes pytest to selected paths; ``None`` tests the whole
    build dir and ``[]`` means no relevant tests were found.
    ``allow_no_tests`` makes pytest's "no tests collected" outcome a
    non-blocking warning when lint passed, for patch jobs against existing
    repositories that may not have tests.

    Provisions the project's own environment first (``uv sync``) so pytest
    runs with the generated package importable and its dependencies present.
    With ``require_installable`` (new-project jobs), a failed sync IS the QA
    failure — a package that doesn't install is not a working deliverable —
    and the sync output is surfaced for the repair writer to fix
    ``pyproject.toml``. Patch jobs degrade gracefully: the user's project may
    legitimately not be uv-installable, so QA falls back to the
    orchestrator's interpreter as before.
    """
    sync_error = ensure_project_env(build_dir)
    if sync_error and require_installable:
        return QAReport(
            passed=False,
            lint_output="",
            test_output=truncate(sync_error),
            integration_notes=(
                "Project environment provisioning failed: `uv sync` could not "
                "install the generated project from its pyproject.toml. Fix the "
                "project metadata/dependencies — the sync output is in test_output."
            ),
            artifact_urls=artifact_refs(artifact_urls),
        )

    lint_tool = lint_runner or LintRunnerTool(workspace_dir=build_dir)

    if lint_paths:
        lint_output = _run_scoped_lint(lint_tool, lint_paths)
        lint_scope_note = f"ruff check scoped to {len(lint_paths)} changed file(s)"
    else:
        lint_output = lint_tool._run(".")
        lint_scope_note = "ruff check over the whole build directory"

    type_tool = type_runner or TypeCheckRunnerTool(workspace_dir=build_dir)
    type_targets = type_paths or _default_type_targets(build_dir)
    type_ok, type_errors = _run_type_gate(type_tool, type_targets)

    # .env.example must match the Settings env vars an operator fills in. Enforce
    # for new projects (require_installable); for patch jobs only note it, since
    # pre-existing .env debt the patch never touched must not fail the job.
    env_issues = check_env_example_consistency(build_dir)
    env_ok = (not env_issues) or not require_installable

    lint_ok = is_pass(lint_output)
    deterministic_failed = not (lint_ok and type_ok and env_ok)
    project_has_tests = has_pytest_files(build_dir)
    effective_test_paths = None if test_paths == [] and project_has_tests else test_paths
    if deterministic_failed and skip_pytest_on_deterministic_failure:
        test_output = "SKIP: pytest skipped because deterministic QA failed first."
        test_scope_note = "pytest skipped before execution"
    elif effective_test_paths == []:
        test_output = "SKIP: no tests collected under this path."
        test_scope_note = "pytest scoped to 0 path(s)"
    else:
        test_tool = test_runner or TestRunnerTool(workspace_dir=build_dir)
        if effective_test_paths is None:
            test_output = test_tool._run(".")
            test_scope_note = "pytest over the whole build directory"
        else:
            test_output = _run_scoped_tests(test_tool, effective_test_paths)
            test_scope_note = f"pytest scoped to {len(effective_test_paths)} path(s)"

    no_tests_warning = (
        allow_no_tests
        and not project_has_tests
        and lint_ok
        and is_no_tests_collected(test_output)
    )
    test_ok = is_pass(test_output) or no_tests_warning

    notes = [
        f"Deterministic QA ran {lint_scope_note}, {test_scope_note}, and mypy over the build directory."
    ]
    if is_skip(lint_output):
        notes.append(f"Lint was not executed: {lint_output}")
    if no_tests_warning:
        notes.append(
            "No pytest tests were collected in the existing project; "
            "QA validated changed files with ruff only."
        )
    elif test_output == "SKIP: pytest skipped because deterministic QA failed first.":
        notes.append("Pytest skipped because deterministic QA failed first.")
    elif is_skip(test_output):
        notes.append(f"Tests were not executed: {test_output}")
    if not lint_ok and not is_skip(lint_output):
        notes.append("Lint failed.")
    if not test_ok and not is_skip(test_output):
        notes.append("Tests failed.")
    if not type_ok:
        notes.append("Type check (mypy) found cross-file symbol drift.")
    if env_issues:
        verb = "is" if not env_ok else "may be"
        notes.append(f".env.example {verb} inconsistent with Settings: " + " ".join(env_issues))

    return QAReport(
        passed=lint_ok and test_ok and type_ok and env_ok,
        lint_output=lint_output,
        test_output=test_output,
        type_output=type_errors,
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

    # Windows .exe build kit (PyInstaller). The kit is multi-file and only the
    # single-file final-QA repair pass can patch the workspace afterwards, so a
    # missing kit must fail the gate rather than be silently shipped.
    if not (root / "build.spec").is_file():
        issues.append("Missing build.spec (PyInstaller spec) for the Windows .exe build kit.")
    if not ((root / "build.ps1").is_file() or (root / "build.bat").is_file()):
        issues.append("Missing Windows build script (build.ps1 or build.bat) for the .exe build kit.")
    if pyproject and "pyinstaller" not in pyproject.lower():
        issues.append("pyproject.toml must declare pyinstaller as a dev dependency for the .exe build kit.")

    return ReviewResult(
        subtask_id="architecture_gate",
        passed=not issues,
        issues=issues,
        suggestions=[] if issues else ["RPA architecture gate passed."],
    )


def rpa_packaging_remediation_subtask(build_dir: str) -> SubTask | None:
    """One writer work package filling in missing Windows build-kit files.

    The planner is told to include the kit (entry script, build.spec, build
    script, .env.example, pyinstaller dev dep), but when a writer bundle drops
    one of them the deterministic RPA gate would fail the whole job at
    finalize with no remediation path. This maps the file-level gate misses to
    a concrete SubTask that ``finalize`` runs through the normal writer loop
    before final QA. Returns ``None`` when the kit is complete. Layer/structure
    gate issues (missing domain/application/infrastructure dirs) are NOT
    remediated here — those mean the plan itself was wrong.
    """
    root = Path(build_dir)
    files: list[FileSkeleton] = []
    if not (root / ".env.example").is_file():
        files.append(
            FileSkeleton(
                path=".env.example",
                purpose=(
                    "Runtime configuration template listing every environment "
                    "variable the application reads (no real secrets)."
                ),
                change_type="create",
            )
        )
    if not (root / "build.spec").is_file():
        files.append(
            FileSkeleton(
                path="build.spec",
                purpose=(
                    "PyInstaller spec (onefile, console=True) whose Analysis "
                    "points at the project's real entry script."
                ),
                change_type="create",
            )
        )
    if not ((root / "build.ps1").is_file() or (root / "build.bat").is_file()):
        files.append(
            FileSkeleton(
                path="build.ps1",
                purpose=(
                    "Windows build script: uv sync, then pyinstaller build.spec "
                    "--noconfirm, producing dist/<name>.exe."
                ),
                change_type="create",
            )
        )
    pyproject = _pyproject_text(root)
    if pyproject and "pyinstaller" not in pyproject:
        files.append(
            FileSkeleton(
                path="pyproject.toml",
                purpose=(
                    "Add pyinstaller to the dev dependency group, keeping all "
                    "other metadata byte-for-byte intact."
                ),
                change_type="modify",
            )
        )
    if not files:
        return None

    file_list = ", ".join(f.path for f in files)
    return SubTask(
        id="rpa_packaging_kit",
        title="Complete the Windows .exe build kit",
        description=(
            "The RPA acceptance gate requires a complete Windows build kit, and "
            f"these files are missing or incomplete: {file_list}. Create/fix them "
            "following the 'Empacotamento — Executável Windows (.exe)' section of "
            "the rpa skill. Inspect the workspace first (workspace_list / "
            "workspace_read) to find the real entry script and package name so "
            "build.spec points at the actual __main__.py path."
        ),
        files=files,
        tech_notes=(
            "build.spec Analysis takes the entry-script .py path (e.g. "
            "src/<pkg>/__main__.py) with pathex=['src'] and datas including "
            ".env.example. build.ps1 runs `uv sync` then `uv run pyinstaller "
            "build.spec --noconfirm`. pyproject.toml edits must only add the "
            "pyinstaller dev dependency."
        ),
        test_criteria=(
            "All listed files exist with complete content; pyproject.toml "
            "declares pyinstaller as a dev dependency; the RPA deterministic "
            "packaging checks pass."
        ),
    )


def _rpa_full_gate(build_dir: str, plan: Plan | None, language: str = "English") -> ReviewResult:
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
                "language": language or "English",
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
# acceptance check. Each gate has signature (build_dir, plan, language) ->
# ReviewResult, runs its own deterministic + LLM passes, and returns a
# ReviewResult with subtask_id="architecture_gate".
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
                files=[
                    FileSkeleton(
                        path=rel_path,
                        purpose=f"Missing module exporting: {symbol_list}.",
                        change_type="create",
                    )
                ],
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


def run_full_architecture_gate(
    build_dir: str, plan: Plan | None, language: str = "English"
) -> ReviewResult:
    """Dispatch to the architecture gate registered for ``plan.domain``.

    Returns a pass-through ReviewResult when no domain gate matches, so
    projects outside the registered domains finalize on lint/test only.
    ``language`` is the resolved output language threaded to the LLM gate.
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
    return gate(build_dir, plan, language)
