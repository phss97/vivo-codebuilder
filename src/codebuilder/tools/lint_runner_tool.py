import subprocess
import sys
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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


# Invoke tools through the active interpreter so the workspace cwd doesn't hide
# venv-installed packages. "SKIP: <reason>" signals the reviewer that the tool
# was unavailable rather than that the code is broken.
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
        target = (Path(self.workspace_dir) / path).resolve()
        code, out = _run(
            [sys.executable, "-m", "ruff", "check", str(target)],
            cwd=self.workspace_dir,
        )
        if code == 0:
            return "PASS"
        if "No module named ruff" in out:
            return _SKIP_MISSING_MODULE.format(module="ruff")
        return out or f"ruff exit {code}"


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
        target = (Path(self.workspace_dir) / path).resolve()
        code, out = _run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", str(target)],
            cwd=self.workspace_dir,
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
