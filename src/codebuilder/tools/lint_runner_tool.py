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
    ensure_project_env(workspace_dir)
    interpreter = project_python(workspace_dir)
    code, out = _run([interpreter, "-m", module, *args], cwd=workspace_dir, timeout=timeout)
    if interpreter != sys.executable and f"No module named {module}" in out:
        code, out = _run([sys.executable, "-m", module, *args], cwd=workspace_dir, timeout=timeout)
    return code, out


# Lint subjects are Python sources only. Plans legitimately include README.md,
# pyproject.toml, .env.example and build.spec — ruff lints any explicitly
# passed file, and build.spec (Python syntax with PyInstaller-injected globals
# like Analysis/PYZ/EXE) always fails F821, so explicit non-.py paths return
# PASS, mirroring ruff's own directory-scan semantics.
_PYTHON_SUFFIXES = {".py", ".pyi"}

# "SKIP: <reason>" signals the reviewer that the tool was unavailable rather
# than that the code is broken.
_SKIP_MISSING_MODULE = "SKIP: {module} not installed in the runtime; review logic manually."


class _LintInput(BaseModel):
    path: str = Field(default=".", description="Relative path to lint")


class LintRunnerTool(BaseTool):
    name: str = "lint_runner"
    description: str = (
        "Run ruff check on a path in the workspace and return the output. "
        "Returns 'PASS' if clean, otherwise the ruff report."
    )
    args_schema: Type[BaseModel] = _LintInput
    workspace_dir: str

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
        )
        if code == 0:
            return "PASS"
        if "No module named ruff" in out:
            return _SKIP_MISSING_MODULE.format(module="ruff")
        return out or f"ruff exit {code}"


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
    # follow_imports = silent: dependencies are analyzed for type info but only
    # the files passed as targets report errors. This makes patch-mode scoping
    # real (a changed file is checked against unchanged deps without failing on
    # the deps' pre-existing debt) and is a no-op for new_project, where the whole
    # package is the target.
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

    def _run(self, path: str = ".") -> str:
        try:
            target = resolve_within(self.workspace_dir, path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if target.is_file() and target.suffix not in _PYTHON_SUFFIXES:
            return "PASS"
        # ensure the env first so pydantic-plugin detection sees the project venv.
        ensure_project_env(self.workspace_dir)
        config = _write_mypy_config(self.workspace_dir)
        try:
            code, out = _run_tool_module(
                "mypy",
                [
                    "--config-file",
                    config,
                    "--no-error-summary",
                    "--hide-error-context",
                    "--no-color-output",
                    "--no-pretty",
                    str(target),
                ],
                self.workspace_dir,
                timeout=180,
            )
        finally:
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
    description: str = (
        "Run pytest against a path in the workspace. Returns 'PASS' or the pytest output."
    )
    args_schema: Type[BaseModel] = _TestInput
    workspace_dir: str

    def _run(self, path: str = ".") -> str:
        try:
            target = resolve_within(self.workspace_dir, path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        # The runner owns its flags. A target project's pyproject may declare
        # addopts requiring plugins not installed here (e.g. --cov needs
        # pytest-cov), which crashes pytest at arg parsing before any test runs.
        code, out = _run_tool_module(
            "pytest",
            [
                "-q",
                "--no-header",
                "--override-ini=addopts=",
                str(target),
            ],
            self.workspace_dir,
            timeout=300,
        )
        if code == 0:
            return "PASS\n" + out
        if "No module named pytest" in out:
            return _SKIP_MISSING_MODULE.format(module="pytest")
        # pytest exits 5 when no tests are collected — not a failure.
        if code == 5:
            return "SKIP: no tests collected under this path."
        return out or f"pytest exit {code}"
