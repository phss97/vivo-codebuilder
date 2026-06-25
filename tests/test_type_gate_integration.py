"""End-to-end proof (real uv + mypy) that the type gate catches cross-file
symbol drift and lets a corrected project pass — the terra failure mode in
miniature: a container reading an attribute the settings class never declares.
"""

import shutil
import textwrap
from pathlib import Path

import pytest

from codebuilder.runtime_qa import run_final_qa

needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")


def _write_project(root: Path, *, container_body: str) -> None:
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "drift-demo"
            version = "0.1.0"
            requires-python = ">=3.10"
            dependencies = []

            [dependency-groups]
            dev = ["pytest>=8", "ruff>=0.6", "mypy>=1.11"]

            [build-system]
            requires = ["hatchling"]
            build-backend = "hatchling.build"

            [tool.hatch.build.targets.wheel]
            packages = ["src/drift_demo"]
            """
        )
    )
    pkg = root / "src" / "drift_demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "settings.py").write_text(
        'class Settings:\n    database_url: str = "sqlite://"\n'
    )
    (pkg / "container.py").write_text(
        "from drift_demo.settings import Settings\n\n\n"
        "def build(settings: Settings) -> str:\n"
        f"    {container_body}\n"
    )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_smoke.py").write_text("def test_ok() -> None:\n    assert True\n")


@needs_uv
def test_type_gate_catches_attribute_drift(tmp_path) -> None:
    # container reads settings.sap_host, which Settings never declares.
    _write_project(tmp_path, container_body="return settings.sap_host")
    report = run_final_qa(str(tmp_path), require_installable=True)
    assert not report.passed, f"notes={report.integration_notes!r}"
    assert "attr-defined" in report.type_output


@needs_uv
def test_type_gate_passes_when_drift_fixed(tmp_path) -> None:
    # corrected: read the real field.
    _write_project(tmp_path, container_body="return settings.database_url")
    report = run_final_qa(str(tmp_path), require_installable=True)
    assert report.passed, (
        f"lint={report.lint_output!r} test={report.test_output!r} "
        f"type={report.type_output!r} notes={report.integration_notes!r}"
    )


def _write_pydantic_project(root: Path, *, field: str) -> None:
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "cfg-demo"
            version = "0.1.0"
            requires-python = ">=3.10"
            dependencies = ["pydantic-settings>=2.0"]

            [dependency-groups]
            dev = ["pytest>=8", "ruff>=0.6", "mypy>=1.11"]

            [build-system]
            requires = ["hatchling"]
            build-backend = "hatchling.build"

            [tool.hatch.build.targets.wheel]
            packages = ["src/cfg_demo"]
            """
        )
    )
    pkg = root / "src" / "cfg_demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "settings.py").write_text(
        textwrap.dedent(
            """\
            from pydantic_settings import BaseSettings, SettingsConfigDict


            class Settings(BaseSettings):
                model_config = SettingsConfigDict(env_prefix="DEMO_")

                database_url: str
                sap_endpoint: str


            def get_settings() -> Settings:
                return Settings()
            """
        )
    )
    (pkg / "container.py").write_text(
        textwrap.dedent(
            f"""\
            from cfg_demo.settings import Settings, get_settings


            def endpoint(settings: Settings | None = None) -> str:
                settings = settings or get_settings()
                return settings.{field}
            """
        )
    )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_smoke.py").write_text("def test_ok() -> None:\n    assert True\n")


@needs_uv
def test_pydantic_settings_correct_passes_no_false_positive(tmp_path) -> None:
    """The BaseSettings false positive: without the pydantic mypy plugin, mypy
    flags `Settings()` as missing every field (call-arg) on CORRECT code. The
    gate must enable the plugin so a correct pydantic-settings project is clean.
    """
    _write_pydantic_project(tmp_path, field="sap_endpoint")
    report = run_final_qa(str(tmp_path), require_installable=True)
    assert report.passed, (
        f"FALSE POSITIVE — correct pydantic code failed the gate. "
        f"type={report.type_output!r} notes={report.integration_notes!r}"
    )


@needs_uv
def test_pydantic_settings_drift_fails(tmp_path) -> None:
    _write_pydantic_project(tmp_path, field="sap_host")  # not a declared field
    report = run_final_qa(str(tmp_path), require_installable=True)
    assert not report.passed
    assert "attr-defined" in report.type_output


class _FakeTool:
    def __init__(self, out: str):
        self.out = out

    def _run(self, path: str = ".") -> str:
        return self.out


def _write_scoped_project(root: Path) -> None:
    """legacy.py carries a real type error; consumer.py (the changed file) is clean."""
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "scoped-demo"
            version = "0.1.0"
            requires-python = ">=3.10"
            dependencies = []

            [dependency-groups]
            dev = ["mypy>=1.11"]

            [build-system]
            requires = ["hatchling"]
            build-backend = "hatchling.build"

            [tool.hatch.build.targets.wheel]
            packages = ["src/scoped_demo"]
            """
        )
    )
    pkg = root / "src" / "scoped_demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "legacy.py").write_text(
        "class Thing:\n    value: int = 0\n\n\ndef use() -> int:\n    return Thing().missing_attr\n"
    )
    (pkg / "consumer.py").write_text(
        "from scoped_demo.legacy import Thing\n\n\ndef run() -> int:\n    return Thing().value\n"
    )


@needs_uv
def test_type_gate_scopes_to_changed_files(tmp_path) -> None:
    """Whole-package check fails on legacy.py's pre-existing error; scoping the
    type gate to the changed file (consumer.py) passes — follow_imports=silent
    consults legacy.py for types without reporting its errors."""
    _write_scoped_project(tmp_path)
    fakes = dict(lint_runner=_FakeTool("PASS"), test_runner=_FakeTool("PASS\n1 passed"))

    whole = run_final_qa(str(tmp_path), require_installable=True, **fakes)
    assert not whole.passed, "expected legacy.py's pre-existing error to fail the whole-repo check"

    scoped = run_final_qa(
        str(tmp_path),
        require_installable=True,
        type_paths=["src/scoped_demo/consumer.py"],
        **fakes,
    )
    assert scoped.passed, f"scoped type check should ignore legacy debt: type={scoped.type_output!r}"
