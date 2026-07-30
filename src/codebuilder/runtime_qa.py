"""Deterministic, package-level QA for attached and generated Python projects."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from codebuilder.schemas import ArtifactRef, Plan, QAReport
from codebuilder.tools import LintRunnerTool, TestRunnerTool, TypeCheckRunnerTool
from codebuilder.tools.project_env import (
    ensure_project_env,
    project_python,
    provisioning_enabled,
)

MAX_QA_OUTPUT_CHARS = 12000
MAX_PROMPT_SECTION_CHARS = 6000

_QA_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
_DEPENDENCY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_ENV_FENCE = re.compile(r"```(?:dotenv|env)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ENV_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)


def is_pass(output: str) -> bool:
    normalized = output.strip()
    return normalized == "PASS" or normalized.startswith("PASS\n")


def is_skip(output: str) -> bool:
    return output.strip().startswith("SKIP:")


def truncate(value: str, limit: int = MAX_QA_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    for _ in range(2):
        marker = f"\n\n[truncated {omitted} chars]\n\n"
        available = max(0, limit - len(marker))
        head = available * 2 // 3
        tail = available - head
        omitted = len(value) - head - tail
    marker = f"\n\n[truncated {omitted} chars]\n\n"
    return f"{value[:head]}{marker}{value[-tail:] if tail else ''}"


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _QA_SKIP_DIRS]
        base = Path(directory)
        files.extend(base / name for name in filenames if name.endswith(".py"))
    return files


def _production_python_files(root: Path) -> list[Path]:
    """Return project source files, excluding tests and generated/build trees."""
    source_root = root / "src" if (root / "src").is_dir() else root
    return [
        path
        for path in _python_files(source_root)
        if "tests" not in path.relative_to(source_root).parts
    ]


def artifact_refs(refs: list[dict] | list[ArtifactRef] | None) -> list[ArtifactRef]:
    return [
        ref if isinstance(ref, ArtifactRef) else ArtifactRef(**ref)
        for ref in refs or []
    ]


def validate_plan(plan: Plan | None) -> Plan:
    """Validate the small structured envelope consumed by the executor."""
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


def qa_report_for_prompt(report: QAReport | None) -> str:
    """Bound each category independently before passing preflight evidence to an agent."""
    if report is None:
        return "(not run: no attached project was resolved)"
    sections = [
        f"Status: {'PASS' if report.passed else 'FAIL'}",
        f"### Lint and format\n{truncate(report.lint_output or '(no output)', MAX_PROMPT_SECTION_CHARS)}",
        f"### MyPy\n{truncate(report.type_output or '(no output)', MAX_PROMPT_SECTION_CHARS)}",
        f"### Install, configuration, dependencies, and entry points\n"
        f"{truncate(report.integration_notes or '(no output)', MAX_PROMPT_SECTION_CHARS)}",
        f"### Tests\n{truncate(report.test_output or '(no output)', MAX_PROMPT_SECTION_CHARS)}",
    ]
    return "\n\n".join(sections)


def qa_report_for_repair(report: QAReport) -> str:
    """Compact JSON view of a failed QA report for a repair pass."""
    payload = report.model_dump()
    payload["artifact_urls"] = []
    for key in ("lint_output", "type_output", "test_output", "integration_notes"):
        payload[key] = truncate(payload.get(key) or "")
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _literal_string(node: ast.AST | None) -> str | None:
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _settings_fields(tree: ast.Module) -> list[str]:
    """Return environment names declared by Pydantic BaseSettings classes."""
    found: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            base.id if isinstance(base, ast.Name) else base.attr
            for base in node.bases
            if isinstance(base, (ast.Name, ast.Attribute))
        }
        if "BaseSettings" not in bases:
            continue

        prefix = ""
        aliases: dict[str, str] = {}
        field_names: list[str] = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                targets = [
                    target.id for target in item.targets if isinstance(target, ast.Name)
                ]
                if "model_config" in targets and isinstance(item.value, ast.Call):
                    for keyword in item.value.keywords:
                        if keyword.arg == "env_prefix":
                            prefix = _literal_string(keyword.value) or prefix
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                name = item.target.id
                if name.startswith("_") or name == "model_config":
                    continue
                if isinstance(item.annotation, ast.Subscript):
                    annotation_name = getattr(item.annotation.value, "id", "")
                    if annotation_name == "ClassVar":
                        continue
                field_names.append(name)
                if isinstance(item.value, ast.Call):
                    for keyword in item.value.keywords:
                        if keyword.arg in {"alias", "validation_alias"}:
                            alias = _literal_string(keyword.value)
                            if alias:
                                aliases[name] = alias
            elif isinstance(item, ast.ClassDef) and item.name == "Config":
                for config_item in item.body:
                    if not isinstance(config_item, ast.Assign):
                        continue
                    if any(
                        isinstance(target, ast.Name) and target.id == "env_prefix"
                        for target in config_item.targets
                    ):
                        prefix = _literal_string(config_item.value) or prefix

        found.extend(
            (aliases.get(name) or f"{prefix}{name}").upper() for name in field_names
        )
    return found


def _base_settings_field_names(tree: ast.Module) -> set[str]:
    """Return Python attribute names declared on Pydantic BaseSettings classes."""
    fields: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            base.id if isinstance(base, ast.Name) else base.attr
            for base in node.bases
            if isinstance(base, (ast.Name, ast.Attribute))
        }
        if "BaseSettings" not in bases:
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(
                item.target, ast.Name
            ):
                continue
            name = item.target.id
            if name.startswith("_") or name == "model_config":
                continue
            if (
                isinstance(item.annotation, ast.Subscript)
                and getattr(item.annotation.value, "id", "") == "ClassVar"
            ):
                continue
            fields.add(name)
    return fields


def _is_any(annotation: ast.AST | None) -> bool:
    return (isinstance(annotation, ast.Name) and annotation.id == "Any") or (
        isinstance(annotation, ast.Attribute) and annotation.attr == "Any"
    )


def _self_attribute(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def check_rpa_production_contract(build_dir: str) -> str:
    """Catch type-bypassing wiring and unused client lifecycle methods in RPA source."""
    root = Path(build_dir)
    parsed: list[tuple[Path, ast.Module]] = []
    settings_fields: set[str] = set()
    for path in _production_python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        parsed.append((path, tree))
        settings_fields.update(_base_settings_field_names(tree))

    failures: list[str] = []
    lifecycle_classes: list[tuple[Path, str, tuple[str, str]]] = []
    calls_by_file: dict[Path, set[str]] = {}
    lifecycle_pairs = (("login", "logout"), ("connect", "disconnect"))

    for path, tree in parsed:
        relative = path.relative_to(root)
        calls_by_file[path] = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not (
                isinstance(node.func, ast.Name) and node.func.id == "getattr"
            ):
                continue
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
                continue
            attribute = node.args[1].value
            if not isinstance(attribute, str):
                continue
            receiver = _self_attribute(node.args[0])
            if receiver in {"settings", "_settings"} and settings_fields:
                if attribute not in settings_fields:
                    failures.append(
                        f"{relative}:{node.lineno}: settings field {attribute!r} "
                        "is not declared by BaseSettings."
                    )

        for class_node in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            method_names = {
                item.name
                for item in class_node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if any(token in class_node.name.lower() for token in ("client", "adapter")):
                for pair in lifecycle_pairs:
                    if set(pair).issubset(method_names):
                        lifecycle_classes.append((path, class_node.name, pair))

            init = next(
                (
                    item
                    for item in class_node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "__init__"
                ),
                None,
            )
            if init is None:
                continue
            any_parameters = {
                argument.arg
                for argument in [*init.args.posonlyargs, *init.args.args]
                if argument.arg != "self" and _is_any(argument.annotation)
            }
            injected: dict[str, str] = {}
            for item in ast.walk(init):
                if not isinstance(item, ast.Assign) or not isinstance(
                    item.value, ast.Name
                ):
                    continue
                if item.value.id not in any_parameters:
                    continue
                for target in item.targets:
                    target_name = _self_attribute(target)
                    if target_name:
                        injected[target_name] = item.value.id
            for item in ast.walk(class_node):
                if not isinstance(item, ast.Call) or not (
                    isinstance(item.func, ast.Name) and item.func.id == "getattr"
                ):
                    continue
                dependency = _self_attribute(item.args[0]) if item.args else None
                if dependency in injected:
                    failures.append(
                        f"{relative}:{item.lineno}: injected dependency "
                        f"{injected[dependency]!r} is typed Any and accessed dynamically; "
                        "use its real Protocol or concrete contract."
                    )

    for defining_path, class_name, (open_method, close_method) in lifecycle_classes:
        external_calls = set().union(
            *(calls for path, calls in calls_by_file.items() if path != defining_path)
        )
        missing = [
            method
            for method in (open_method, close_method)
            if method not in external_calls
        ]
        if missing:
            relative = defining_path.relative_to(root)
            failures.append(
                f"{relative}: {class_name} defines {open_method}()/{close_method}() "
                f"but production source never calls: {', '.join(missing)}."
            )

    return "PASS" if not failures else "\n".join(dict.fromkeys(failures))


def check_env_example(build_dir: str) -> str:
    """Compare Settings and README env snippets with the canonical example."""
    root = Path(build_dir)
    settings: list[str] = []
    parse_errors: list[str] = []
    for path in _python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_errors.append(f"{path.relative_to(root)}: {exc}")
            continue
        settings.extend(_settings_fields(tree))

    if not settings:
        return (
            "PASS"
            if not parse_errors
            else "AST scan warnings:\n" + "\n".join(parse_errors)
        )

    env_path = root / ".env.example"
    if not env_path.is_file():
        return "Missing .env.example for BaseSettings configuration."
    documented = {
        line.split("=", 1)[0].strip().upper()
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    missing = sorted(set(settings) - documented)
    readme_keys: set[str] = set()
    readme_path = root / "README.md"
    if readme_path.is_file():
        try:
            readme = readme_path.read_text(encoding="utf-8")
            readme_keys = {
                match.group(1)
                for block in _ENV_FENCE.findall(readme)
                for match in _ENV_ASSIGNMENT.finditer(block)
            }
        except (OSError, UnicodeError) as exc:
            parse_errors.append(f"README.md: {exc}")
    stale_readme = sorted(readme_keys - documented)
    if not missing and not stale_readme and not parse_errors:
        return "PASS"
    messages: list[str] = []
    if missing:
        messages.append(
            ".env.example is missing BaseSettings keys: " + ", ".join(missing)
        )
    if stale_readme:
        messages.append(
            "README.md documents environment keys not present in .env.example: "
            + ", ".join(stale_readme)
        )
    if parse_errors:
        messages.append("AST scan warnings:\n" + "\n".join(parse_errors))
    return "\n".join(messages)


def _runtime_dependency_values(pyproject: dict[str, Any]) -> list[str]:
    project = pyproject.get("project") or {}
    return [
        value for value in (project.get("dependencies") or []) if isinstance(value, str)
    ]


def _dependency_names(values: list[str]) -> set[str]:
    names: set[str] = set()
    for value in values:
        match = _DEPENDENCY_NAME.match(value.strip())
        if match:
            names.add(match.group(0).lower().replace("_", "-").replace(".", "-"))
    return names


def _load_pyproject(root: Path) -> tuple[dict[str, Any] | None, str]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return None, "SKIP: no pyproject.toml"
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle), ""
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return None, f"Invalid pyproject.toml: {exc}"


def _source_signals(root: Path) -> tuple[bool, bool]:
    uses_pyodbc_url = False
    imports_win32com = False
    for path in _python_files(root):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        uses_pyodbc_url = uses_pyodbc_url or "mssql+pyodbc" in source.lower()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports_win32com = imports_win32com or any(
                    alias.name == "win32com" or alias.name.startswith("win32com.")
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports_win32com = imports_win32com or (
                    node.module == "win32com" or node.module.startswith("win32com.")
                )
    if not uses_pyodbc_url:
        for path in (root / ".env.example", root / "README.md"):
            if path.is_file():
                try:
                    uses_pyodbc_url = (
                        "mssql+pyodbc" in path.read_text(encoding="utf-8").lower()
                    )
                except (OSError, UnicodeError):
                    pass
    return uses_pyodbc_url, imports_win32com


def _check_entry_points(
    root: Path, pyproject: dict[str, Any], *, smoke: bool = False
) -> str:
    scripts = (pyproject.get("project") or {}).get("scripts") or {}
    if not scripts:
        return "SKIP: no [project.scripts] entries"
    if not isinstance(scripts, dict):
        return "Invalid [project.scripts]: expected a TOML table."
    code = (
        "import functools,importlib,sys; "
        "obj=functools.reduce(getattr, sys.argv[2].split('.'), "
        "importlib.import_module(sys.argv[1])); "
        "assert callable(obj), f'{sys.argv[1]}:{sys.argv[2]} is not callable'"
    )
    smoke_code = (
        "import functools,importlib,sys; "
        "name,module,target=sys.argv[1:4]; "
        "obj=functools.reduce(getattr,target.split('.'),importlib.import_module(module)); "
        "sys.argv=[name,'--help']; result=obj(); "
        "raise SystemExit(result if isinstance(result,int) else 0)"
    )
    failures: list[str] = []
    for name, target in scripts.items():
        if not isinstance(target, str) or ":" not in target:
            failures.append(f"{name}: invalid target {target!r}")
            continue
        module, callable_name = target.split(":", 1)
        callable_name = callable_name.split("[", 1)[0]
        try:
            process = subprocess.run(
                [project_python(str(root)), "-c", code, module, callable_name],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{name}: {exc}")
            continue
        if process.returncode != 0:
            output = ((process.stdout or "") + (process.stderr or "")).strip()
            failures.append(
                f"{name} ({target}): {output or f'exit {process.returncode}'}"
            )
            continue
        if smoke:
            try:
                with tempfile.TemporaryDirectory(
                    prefix="codebuilder-entrypoint-"
                ) as smoke_cwd:
                    env = dict(os.environ)
                    python_path = env.get("PYTHONPATH")
                    env["PYTHONPATH"] = str(root) + (
                        os.pathsep + python_path if python_path else ""
                    )
                    process = subprocess.run(
                        [
                            project_python(str(root)),
                            "-c",
                            smoke_code,
                            name,
                            module,
                            callable_name,
                        ],
                        cwd=smoke_cwd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=env,
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append(f"{name} --help: {exc}")
                continue
            if process.returncode != 0:
                output = ((process.stdout or "") + (process.stderr or "")).strip()
                failures.append(
                    f"{name} --help ({target}): "
                    f"{output or f'exit {process.returncode}'}"
                )
    return (
        "PASS"
        if not failures
        else "Console entry point failures:\n" + "\n".join(failures)
    )


def check_runtime_contract(build_dir: str, smoke_entry_points: bool = False) -> str:
    """Validate dependency declarations and installed console entry points."""
    root = Path(build_dir)
    pyproject, load_note = _load_pyproject(root)
    if pyproject is None:
        return load_note
    dependency_values = _runtime_dependency_values(pyproject)
    declared = _dependency_names(dependency_values)
    uses_pyodbc_url, imports_win32com = _source_signals(root)
    failures: list[str] = []
    if uses_pyodbc_url and "pyodbc" not in declared:
        failures.append("mssql+pyodbc is used but dependency `pyodbc` is not declared.")
    if imports_win32com:
        pywin32_specs = [
            value.lower()
            for value in dependency_values
            if "pywin32" in _dependency_names([value])
        ]
        if not pywin32_specs:
            failures.append(
                "win32com is imported but Windows-scoped dependency `pywin32` is not declared."
            )
        elif not any(
            ("sys_platform" in value and "==" in value and "win32" in value)
            or ("platform_system" in value and "==" in value and "windows" in value)
            for value in pywin32_specs
        ):
            failures.append(
                "`pywin32` must be scoped to Windows with a `sys_platform == 'win32'` marker."
            )
    entry_points = _check_entry_points(root, pyproject, smoke=smoke_entry_points)
    if not is_pass(entry_points) and not is_skip(entry_points):
        failures.append(entry_points)
    return "PASS" if not failures else "\n".join(failures)


def _run_with_example_env(build_dir: str) -> str:
    """Run tests with .env.example active, restoring the workspace afterward."""
    root = Path(build_dir)
    env_path = root / ".env"
    example_path = root / ".env.example"
    if env_path.exists():
        return "SKIP: existing .env was already active during the main pytest run"
    if not example_path.is_file():
        return "SKIP: no .env.example"
    output = ""
    cleanup_error = ""
    try:
        shutil.copyfile(example_path, env_path)
        output = TestRunnerTool(
            workspace_dir=build_dir,
            provision_environment=False,
        )._run(".")
    except OSError as exc:
        output = f"Could not activate .env.example for tests: {exc}"
    finally:
        try:
            env_path.unlink(missing_ok=True)
        except OSError as exc:
            cleanup_error = f"Could not remove temporary .env after tests: {exc}"
    return "\n".join(part for part in (output, cleanup_error) if part)


def run_final_qa(
    build_dir: str,
    *,
    artifact_urls: list[dict] | list[ArtifactRef] | None = None,
    require_installable: bool = False,
    require_typecheck: bool = False,
    locked_sync: bool = True,
) -> QAReport:
    """Run every applicable deterministic check and aggregate all failures."""
    has_pyproject = (Path(build_dir) / "pyproject.toml").is_file()
    if has_pyproject and require_installable and not provisioning_enabled():
        sync_output = "SKIP: project environment provisioning is disabled"
    else:
        sync_output = ensure_project_env(build_dir, locked=locked_sync)
    sync_ok = not sync_output or not require_installable

    lint_output = LintRunnerTool(
        workspace_dir=build_dir,
        provision_environment=False,
    )._run(".")
    lint_ok = is_pass(lint_output)

    type_output = TypeCheckRunnerTool(
        workspace_dir=build_dir,
        native_config=True,
        provision_environment=False,
    )._run(".")
    type_ok = is_pass(type_output) or (is_skip(type_output) and not require_typecheck)

    env_output = check_env_example(build_dir)
    env_ok = is_pass(env_output)
    runtime_output = check_runtime_contract(build_dir, require_typecheck)
    runtime_ok = is_pass(runtime_output) or is_skip(runtime_output)

    wiring_output = (
        check_rpa_production_contract(build_dir)
        if require_typecheck
        else "SKIP: not RPA"
    )
    wiring_ok = is_pass(wiring_output) or is_skip(wiring_output)

    test_output = TestRunnerTool(
        workspace_dir=build_dir,
        provision_environment=False,
    )._run(".")
    test_ok = is_pass(test_output)
    example_env_test_output = (
        _run_with_example_env(build_dir) if require_typecheck else "SKIP: not RPA"
    )
    example_env_test_ok = is_pass(example_env_test_output) or is_skip(
        example_env_test_output
    )
    combined_test_output = test_output
    if require_typecheck:
        combined_test_output = (
            f"{test_output}\n\n.env.example environment:\n{example_env_test_output}"
        )

    checks = [
        f"uv sync --locked: {'PASS' if not sync_output else 'FAIL'}",
        f"ruff check + format: {'PASS' if lint_ok else 'FAIL'}",
        f"mypy: {'PASS' if type_ok else 'FAIL'}",
        f".env.example consistency: {'PASS' if env_ok else 'FAIL'}",
        f"runtime dependencies and entry points: {'PASS' if runtime_ok else 'FAIL'}",
        f"RPA production wiring: {'PASS' if wiring_ok else 'FAIL'}",
        f"pytest: {'PASS' if test_ok else 'FAIL'}",
        f"pytest with .env.example: {'PASS' if example_env_test_ok else 'FAIL'}",
    ]
    details: list[str] = []
    if sync_output:
        details.append(f"uv sync --locked output:\n{truncate(sync_output)}")
    if is_skip(type_output) and not require_typecheck:
        details.append(f"MyPy was not applicable: {type_output}")
    if not env_ok:
        details.append(f"Configuration contract:\n{env_output}")
    if not runtime_ok:
        details.append(f"Runtime contract:\n{runtime_output}")
    if not wiring_ok:
        details.append(f"RPA production wiring:\n{wiring_output}")
    if not example_env_test_ok:
        details.append(
            "Pytest with .env.example:\n" + truncate(example_env_test_output)
        )
    notes = "\n".join([*checks, *details])

    return QAReport(
        passed=(
            sync_ok
            and lint_ok
            and type_ok
            and env_ok
            and runtime_ok
            and wiring_ok
            and test_ok
            and example_env_test_ok
        ),
        lint_output=truncate(lint_output),
        type_output=truncate(type_output),
        test_output=truncate(combined_test_output),
        integration_notes=truncate(notes),
        artifact_urls=artifact_refs(artifact_urls),
    )
