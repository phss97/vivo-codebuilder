"""Tests for per-project QA environments, non-Python lint skips, RPA packaging
remediation, and git harness excludes — the fixes that let a skill-compliant
src-layout RPA project actually pass final QA."""

import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from codebuilder.runtime_qa import (
    rpa_packaging_remediation_subtask,
    run_final_qa,
)
from codebuilder.tools import git_tool
from codebuilder.tools.lint_runner_tool import LintRunnerTool, TestRunnerTool
from codebuilder.tools.project_env import ensure_project_env, project_python

needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")


# --- lint runner: non-Python files -----------------------------------------


def test_lint_runner_passes_explicit_non_python_files(tmp_path):
    # build.spec is Python syntax with PyInstaller-injected globals; explicit
    # ruff invocation flags F821 and previously failed every RPA build-kit
    # subtask review.
    (tmp_path / "build.spec").write_text("a = Analysis(['src/x/__main__.py'])\n")
    (tmp_path / "README.md").write_text("# docs\n")
    (tmp_path / "build.ps1").write_text("uv sync\n")
    tool = LintRunnerTool(workspace_dir=str(tmp_path))
    assert tool._run("build.spec") == "PASS"
    assert tool._run("README.md") == "PASS"
    assert tool._run("build.ps1") == "PASS"


def test_lint_runner_still_lints_python_files(tmp_path):
    (tmp_path / "bad.py").write_text("import os\n")  # F401 unused import
    out = LintRunnerTool(workspace_dir=str(tmp_path))._run("bad.py")
    assert out != "PASS"
    assert "F401" in out


def test_lint_runner_still_lints_directories(tmp_path):
    (tmp_path / "bad.py").write_text("import os\n")
    out = LintRunnerTool(workspace_dir=str(tmp_path))._run(".")
    assert out != "PASS"


# --- project env provisioning ----------------------------------------------


def test_project_python_falls_back_to_orchestrator(tmp_path):
    assert project_python(str(tmp_path)) == sys.executable


def test_project_python_returns_absolute_project_venv(tmp_path, monkeypatch):
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)

    assert project_python(tmp_path.name) == str(python.resolve())


def test_ensure_project_env_noop_without_pyproject(tmp_path):
    assert ensure_project_env(str(tmp_path)) == ""
    assert not (tmp_path / ".venv").exists()


def test_ensure_project_env_respects_disable_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
    )
    assert ensure_project_env(str(tmp_path)) == ""
    assert not (tmp_path / ".venv").exists()


def _write_src_layout_project(root: Path) -> None:
    """Minimal skill-style src-layout project whose tests import the package."""
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "meu-robo"
            version = "0.1.0"
            requires-python = ">=3.10"
            dependencies = []

            [dependency-groups]
            dev = ["pytest>=8", "ruff>=0.6"]

            [build-system]
            requires = ["hatchling"]
            build-backend = "hatchling.build"

            [tool.hatch.build.targets.wheel]
            packages = ["src/meu_robo"]
            """
        )
    )
    pkg = root / "src" / "meu_robo"
    (pkg / "domain").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "domain" / "__init__.py").write_text("")
    (pkg / "domain" / "item.py").write_text(
        "class Item:\n"
        "    def __init__(self, ref: str) -> None:\n"
        "        self.ref = ref\n"
    )
    tests = root / "tests" / "unit"
    tests.mkdir(parents=True)
    (tests / "test_item.py").write_text(
        "from meu_robo.domain.item import Item\n\n\n"
        "def test_item_keeps_ref():\n"
        "    assert Item('a').ref == 'a'\n"
    )


@needs_uv
def test_final_qa_passes_for_src_layout_project(tmp_path):
    """The original failure mode: without provisioning, pytest cannot import a
    src-layout package and every skill-compliant RPA project failed final QA."""
    _write_src_layout_project(tmp_path)
    report = run_final_qa(str(tmp_path), require_installable=True)
    assert report.passed, f"lint={report.lint_output!r} test={report.test_output!r}"
    # provisioned interpreter is now preferred for subsequent tool runs
    assert project_python(str(tmp_path)) != sys.executable
    # second provisioning call is a marker-hit no-op
    assert ensure_project_env(str(tmp_path)) == ""


@needs_uv
def test_final_qa_fails_when_new_project_not_installable(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["definitely-not-a-real-package-xyz==99.99.99"]\n'
    )
    report = run_final_qa(str(tmp_path), require_installable=True)
    assert not report.passed
    assert "uv sync" in report.integration_notes
    assert report.test_output  # carries the sync error for the repair writer


@needs_uv
def test_test_runner_uses_project_env_for_src_layout(tmp_path):
    _write_src_layout_project(tmp_path)
    out = TestRunnerTool(workspace_dir=str(tmp_path))._run(".")
    assert out.startswith("PASS"), out


# --- RPA packaging remediation ----------------------------------------------


def _write_complete_kit(root: Path) -> None:
    (root / ".env.example").write_text("APP_ENV=dev\n")
    (root / "build.spec").write_text("a = Analysis(['src/x/__main__.py'])\n")
    (root / "build.ps1").write_text("uv run pyinstaller build.spec\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        '[dependency-groups]\ndev = ["pyinstaller>=6"]\n'
    )


def test_rpa_remediation_returns_none_for_complete_kit(tmp_path):
    _write_complete_kit(tmp_path)
    assert rpa_packaging_remediation_subtask(str(tmp_path)) is None


def test_rpa_remediation_targets_missing_kit_files(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
    )
    stub = rpa_packaging_remediation_subtask(str(tmp_path))
    assert stub is not None
    assert stub.id == "rpa_packaging_kit"
    paths = stub.file_paths
    assert set(paths) == {".env.example", "build.spec", "build.ps1", "pyproject.toml"}
    by_path = {f.path: f for f in stub.files}
    assert by_path["pyproject.toml"].change_type == "modify"
    assert by_path["build.spec"].change_type == "create"


def test_rpa_remediation_accepts_bat_build_script(tmp_path):
    _write_complete_kit(tmp_path)
    (tmp_path / "build.ps1").unlink()
    (tmp_path / "build.bat").write_text("uv run pyinstaller build.spec\n")
    assert rpa_packaging_remediation_subtask(str(tmp_path)) is None


# --- git harness excludes ----------------------------------------------------


def test_git_diff_excludes_harness_artifacts(tmp_path):
    (tmp_path / "app.py").write_text("print('v1')\n")
    git_tool.init_and_commit(str(tmp_path), "baseline")

    # QA provisioning artifacts appear after the baseline commit
    venv_dir = tmp_path / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "junk.py").write_text("x = 1\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "app.py").write_text("print('v2')\n")

    patch = git_tool.diff(str(tmp_path))
    assert "app.py" in patch
    assert ".venv" not in patch
    assert "uv.lock" not in patch
