"""Tests for the Claude Agent SDK engine: plan/executor wrappers, plan
validation, light QA, skills install, and async-flow correctness."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect

import pytest

import codebuilder.main as main
from codebuilder import cc_agent
from codebuilder.cc_agent import CCAgentError
from codebuilder.runtime_qa import run_final_qa, validate_plan
from codebuilder.schemas import Plan
from codebuilder.tools.git_tool import _HARNESS_EXCLUDES
from codebuilder.tools.s3_artifacts import SKIP_DIRS


# --- fake SDK messages / query --------------------------------------------


class _FakeText:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAssistant:
    def __init__(self, text: str) -> None:
        self.content = [_FakeText(text)]


class _FakeResult:
    def __init__(self, structured_output=None, subtype="success", is_error=False) -> None:
        self.structured_output = structured_output
        self.subtype = subtype
        self.is_error = is_error


def _make_query(messages):
    async def _q(**_kwargs):
        for m in messages:
            yield m

    return _q


VALID_PLAN = {
    "project_name": "demo",
    "mode": "new_project",
    "tech_stack": ["python"],
    "language": "English",
    "domain": "",
    "plan_markdown": "# Plan\n\nBuild the thing.",
    "open_questions": [],
    "assumptions": [],
}


# --- validate_plan ---------------------------------------------------------


def test_validate_plan_ok():
    assert validate_plan(Plan.model_validate(VALID_PLAN)).mode == "new_project"


def test_validate_plan_rejects_empty_markdown():
    with pytest.raises(ValueError):
        validate_plan(Plan.model_validate({**VALID_PLAN, "plan_markdown": "   "}))


def test_validate_plan_rejects_non_plan():
    with pytest.raises(ValueError):
        validate_plan(None)  # type: ignore[arg-type]


# --- run_planner -----------------------------------------------------------


def test_run_planner_returns_plan():
    q = _make_query([_FakeResult(structured_output=VALID_PLAN)])
    plan = asyncio.run(cc_agent.run_planner(cwd=".", prompt="x", query_fn=q))
    assert isinstance(plan, Plan)
    assert plan.plan_markdown.startswith("# Plan")


def test_run_planner_schema_retries_exhausted():
    q = _make_query([_FakeResult(subtype="error_max_structured_output_retries")])
    with pytest.raises(CCAgentError):
        asyncio.run(cc_agent.run_planner(cwd=".", prompt="x", query_fn=q))


def test_run_planner_no_output():
    q = _make_query([_FakeResult(structured_output=None)])
    with pytest.raises(CCAgentError):
        asyncio.run(cc_agent.run_planner(cwd=".", prompt="x", query_fn=q))


def test_run_planner_no_messages():
    q = _make_query([])
    with pytest.raises(CCAgentError):
        asyncio.run(cc_agent.run_planner(cwd=".", prompt="x", query_fn=q))


# --- run_executor ----------------------------------------------------------


def test_run_executor_transcript_and_progress():
    seen: list = []
    q = _make_query([_FakeAssistant("wrote file A"), _FakeAssistant("ran tests"), _FakeResult()])
    out = asyncio.run(
        cc_agent.run_executor(cwd=".", prompt="x", query_fn=q, on_message=seen.append)
    )
    assert "wrote file A" in out and "ran tests" in out
    assert len(seen) == 3


# --- async-flow correctness (the resume-path gap) --------------------------


def test_flow_methods_are_async():
    # build/finalize/revise_plan run inside AMP's already-running event loop on
    # the resume path; they must be coroutines (awaited), never asyncio.run().
    for name in ("plan", "revise_plan", "build", "finalize"):
        assert inspect.iscoroutinefunction(getattr(main.CodebuilderFlow, name)), name


def test_no_asyncio_run_in_main():
    import pathlib

    src = pathlib.Path(main.__file__).read_text(encoding="utf-8")
    assert "asyncio.run(" not in src


# --- skills install + exclusion --------------------------------------------


def test_install_skills_and_exclusions(tmp_path):
    main._install_skills(tmp_path)
    assert (tmp_path / ".claude" / "skills" / "rpa" / "SKILL.md").is_file()
    assert (tmp_path / ".claude" / "skills" / "code-review-gate" / "SKILL.md").is_file()
    # never shipped in artifacts / diffs
    assert ".claude" in SKIP_DIRS
    assert ".claude/" in _HARNESS_EXCLUDES


# --- light QA (integration; needs ruff) ------------------------------------

_ruff = importlib.util.find_spec("ruff") is not None


@pytest.mark.skipif(not _ruff, reason="ruff not available")
def test_run_final_qa_clean_project(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_mod.py").write_text(
        "from mod import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    report = run_final_qa(str(tmp_path))
    assert report.passed, report.integration_notes


@pytest.mark.skipif(not _ruff, reason="ruff not available")
def test_run_final_qa_failing_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert False\n")
    report = run_final_qa(str(tmp_path))
    assert not report.passed


@pytest.mark.skipif(not _ruff, reason="ruff not available")
def test_run_final_qa_patch_ignores_untouched_lint_debt(tmp_path, monkeypatch):
    # Patch mode: pre-existing lint debt in an untouched customer file must not
    # fail QA — only the changed file is linted.
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / "legacy.py").write_text("import os\nx=1\n")  # unused import + style: ruff would flag
    (tmp_path / "new_feature.py").write_text("def feature():\n    return 42\n")
    report = run_final_qa(
        str(tmp_path), changed_paths=["new_feature.py"], allow_no_tests=True
    )
    assert report.passed, report.integration_notes
    # Whole-dir lint would catch legacy.py and fail — prove that.
    whole = run_final_qa(str(tmp_path), allow_no_tests=True)
    assert not whole.passed
