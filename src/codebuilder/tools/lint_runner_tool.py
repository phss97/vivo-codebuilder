import os
import subprocess
import sys
import tempfile
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from codebuilder.tools.project_env import ensure_project_env, project_python
from codebuilder.tools.workspace_tool import resolve_within


def _run(cmd: list[str], cwd: str, timeout: int = 120) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"


def _run_tool_module(
    module: str,
    args: list[str],
    workspace_dir: str,
    timeout: int = 120,
    *,
    provision_environment: bool = True,
) -> tuple[int, str]:
    """Run ``python -m <module>`` preferring the project's own venv.

    The generated project's interpreter (provisioned by ``ensure_project_env``)
    carries the project itself plus its dependencies, so pytest can import the
    src-layout package. When the project venv lacks the tool (non-RPA projects
    may not declare ruff/pytest), retry with the orchestrator's interpreter —
    the pre-existing behavior. The bare ``No module named <tool>`` text only
    appears when ``-m`` itself fails; import errors inside test runs quote the
    module name, so they don't trigger the fallback.
    """
    if provision_environment:
        ensure_project_env(workspace_dir)
    interpreter = project_python(workspace_dir)
    code, out = _run(
        [interpreter, "-m", module, *args], cwd=workspace_dir, timeout=timeout
    )
    if (
        provision_environment
        and interpreter != sys.executable
        and f"No module named {module}" in out
    ):
        code, out = _run(
            [sys.executable, "-m", module, *args], cwd=workspace_dir, timeout=timeout
        )
    return code, out


# Lint subjects are Python sources only. Plans legitimately include README.md,
# pyproject.toml, .env.example and build.spec — ruff lints any explicitly
# passed file, and build.spec (Python syntax with PyInstaller-injected globals
# like Analysis/PYZ/EXE) always fails F821, so explicit non-.py paths return
# PASS, mirroring ruff's own directory-scan semantics.
_PYTHON_SUFFIXES = {".py", ".pyi"}

# "SKIP: <reason>" signals the reviewer that the tool was unavailable rather
# than that the code is broken.
_SKIP_MISSING_MODULE = (
    "SKIP: {module} not installed in the runtime; review logic manually."
)


def apply_ruff_fixes(workspace_dir: str) -> str:
    """Apply Ruff's safe fixes and formatter to a generated project."""
    outputs: list[str] = []
    for args in (["check", "--fix", "."], ["format", "."]):
        code, out = _run(
            [sys.executable, "-m", "ruff", *args],
            cwd=workspace_dir,
        )
        if "No module named ruff" in out:
            return _SKIP_MISSING_MODULE.format(module="ruff")
        if code != 0:
            outputs.append(out or f"ruff {' '.join(args)} exit {code}")
    return "\n".join(outputs) if outputs else "PASS"


class _LintInput(BaseModel):
    path: str = Field(default=".", description="Relative path to lint")


class LintRunnerTool(BaseTool):
    name: str = "lint_runner"
    description: str = (
        "Run ruff lint and format checks on a path in the workspace. "
        "Returns 'PASS' if clean, otherwise the ruff report."
    )
    args_schema: Type[BaseModel] = _LintInput
    workspace_dir: str
    provision_environment: bool = True

    def _run(self, path: str = ".") -> str:
        try:
            target = resolve_within(self.workspace_dir, path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if target.is_file() and target.suffix not in _PYTHON_SUFFIXES:
            return "PASS"
        code, out = _run_tool_module(
            "ruff",
            ["check", str(target)],
            self.workspace_dir,
            provision_environment=self.provision_environment,
        )
        if "No module named ruff" in out:
            return _SKIP_MISSING_MODULE.format(module="ruff")
        failures: list[str] = []
        if code != 0:
            failures.append(f"ruff check:\n{out or f'ruff exit {code}'}")
        format_code, format_out = _run_tool_module(
            "ruff",
            ["format", "--check", str(target)],
            self.workspace_dir,
            provision_environment=self.provision_environment,
        )
        if "No module named ruff" in format_out:
            return _SKIP_MISSING_MODULE.format(module="ruff")
        if format_code != 0:
            failures.append(
                f"ruff format --check:\n{format_out or f'ruff exit {format_code}'}"
            )
        return "\n\n".join(failures) if failures else "PASS"


class _TypeCheckInput(BaseModel):
    path: str = Field(default=".", description="Relative path to type-check")


def _module_importable(workspace_dir: str, module: str) -> bool:
    """True when ``module`` imports under the project's interpreter."""
    try:
        code, _ = _run(
            [project_python(workspace_dir), "-c", f"import {module}"],
            cwd=workspace_dir,
            timeout=30,
        )
        return code == 0
    except Exception:  # noqa: BLE001 — detection is best-effort
        return False


def _write_mypy_config(workspace_dir: str) -> str:
    """Write a self-contained mypy config the gate controls, returning its path.

    Using our own config (via ``--config-file``) ignores the project's
    ``[tool.mypy]`` so a generated ``strict = true`` can't flood us with
    annotation noise. Critically, it enables the pydantic mypy plugin when
    pydantic is installed — without it mypy treats every ``Settings()`` /
    pydantic model construction as missing all fields and emits bogus
    ``call-arg`` errors on correct code (the BaseSettings false positive).
    """
    # follow_imports = silent keeps this compatibility mode focused on the
    # explicit target. Strict package QA uses native_config=True instead.
    lines = ["[mypy]", "ignore_missing_imports = True", "follow_imports = silent"]
    if _module_importable(workspace_dir, "pydantic"):
        lines.append("plugins = pydantic.mypy")
    fd, path = tempfile.mkstemp(prefix="codebuilder-mypy-", suffix=".ini")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


class TypeCheckRunnerTool(BaseTool):
    name: str = "type_checker"
    description: str = (
        "Run mypy on a path in the workspace and return the output. Returns "
        "'PASS' if mypy reports no errors, otherwise the mypy report."
    )
    args_schema: Type[BaseModel] = _TypeCheckInput
    workspace_dir: str
    native_config: bool = False
    provision_environment: bool = True

    def _run(self, path: str = ".") -> str:
        try:
            target = resolve_within(self.workspace_dir, path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if target.is_file() and target.suffix not in _PYTHON_SUFFIXES:
            return "PASS"
        # ensure the env first so pydantic-plugin detection sees the project venv.
        if self.provision_environment:
            ensure_project_env(self.workspace_dir)
        config = None if self.native_config else _write_mypy_config(self.workspace_dir)
        args = [
            "--no-error-summary",
            "--hide-error-context",
            "--no-color-output",
            "--no-pretty",
            str(target),
        ]
        if config:
            args[0:0] = ["--config-file", config]
        try:
            code, out = _run_tool_module(
                "mypy",
                args,
                self.workspace_dir,
                timeout=180,
                provision_environment=self.provision_environment,
            )
        finally:
            if config:
                try:
                    os.unlink(config)
                except OSError:
                    pass
        if "No module named mypy" in out:
            return _SKIP_MISSING_MODULE.format(module="mypy")
        if code == 0:
            return "PASS"
        return out or f"mypy exit {code}"


class _TestInput(BaseModel):
    path: str = Field(default=".", description="Relative path of tests to run")


class TestRunnerTool(BaseTool):
    name: str = "test_runner"
    description: str = "Run pytest against a path in the workspace. Returns 'PASS' or the pytest output."
    args_schema: Type[BaseModel] = _TestInput
    workspace_dir: str
    provision_environment: bool = True

    def _run(self, path: str = ".") -> str:
        try:
            target = resolve_within(self.workspace_dir, path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        # Keep the project's native pytest configuration (including coverage
        # gates), but disable fail-fast so the report covers the full suite.
        raw_timeout = os.environ.get("CODEBUILDER_TEST_TIMEOUT_SECONDS", "2400")
        try:
            timeout = max(1, int(raw_timeout))
        except ValueError:
            timeout = 2400
        code, out = _run_tool_module(
            "pytest",
            [
                "-q",
                "--no-header",
                "--maxfail=0",
                str(target),
            ],
            self.workspace_dir,
            timeout=timeout,
            provision_environment=self.provision_environment,
        )
        if code == 0:
            return "PASS\n" + out
        if "No module named pytest" in out:
            return _SKIP_MISSING_MODULE.format(module="pytest")
        # Let package QA decide whether no collected tests are acceptable.
        if code == 5:
            return "SKIP: no tests collected under this path."
        return out or f"pytest exit {code}"
