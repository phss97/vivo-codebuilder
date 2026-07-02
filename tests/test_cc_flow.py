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
    def __init__(self, structured_output=None, subtype="success", is_error=False,
                 api_error_status=None) -> None:
        self.structured_output = structured_output
        self.subtype = subtype
        self.is_error = is_error
        self.api_error_status = api_error_status


def _make_query(messages):
    async def _q(**_kwargs):
        for m in messages:
            yield m

    return _q


class _FakeProcessError(Exception):
    """Stand-in for the SDK's ProcessError raised after an error result."""


def _make_api_error_query(status, *, succeed_after=None):
    """Query that yields an error ResultMessage (carrying api_error_status) then
    raises, mimicking the CLI exiting non-zero. If succeed_after is set, the Nth+
    call yields a good result instead (to test retry recovery)."""
    calls = {"n": 0}

    async def _q(**_kwargs):
        calls["n"] += 1
        if succeed_after is not None and calls["n"] > succeed_after:
            yield _FakeResult(structured_output=VALID_PLAN)
            return
        yield _FakeResult(is_error=True, subtype="success", api_error_status=status)
        raise _FakeProcessError(f"exit code 1 (status {status})")

    _q.calls = calls
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


# --- transient API error retry (the "error result: success" case) ----------


async def _no_sleep(_seconds):
    return None


def test_transient_api_error_surfaces_status(monkeypatch):
    # 529 exhausts retries → error message must carry the real HTTP status,
    # not the useless "success" subtype.
    monkeypatch.setattr(cc_agent.asyncio, "sleep", _no_sleep)
    monkeypatch.setenv("CODEBUILDER_AGENT_API_RETRIES", "1")
    q = _make_api_error_query(529)
    with pytest.raises(CCAgentError, match="HTTP 529"):
        asyncio.run(cc_agent.run_executor(cwd=".", prompt="x", query_fn=q))
    assert q.calls["n"] == 2  # 1 attempt + 1 retry


def test_transient_api_error_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(cc_agent.asyncio, "sleep", _no_sleep)
    monkeypatch.setenv("CODEBUILDER_AGENT_API_RETRIES", "3")
    q = _make_api_error_query(429, succeed_after=2)
    plan = asyncio.run(cc_agent.run_planner(cwd=".", prompt="x", query_fn=q))
    assert isinstance(plan, Plan)
    assert q.calls["n"] == 3  # failed twice, succeeded on the third


def test_non_transient_api_error_is_not_retried(monkeypatch):
    # A 400 (bad model / bad request) is a real bug — retrying just burns credits.
    monkeypatch.setenv("CODEBUILDER_AGENT_API_RETRIES", "3")
    q = _make_api_error_query(400)
    with pytest.raises(CCAgentError, match="HTTP 400"):
        asyncio.run(cc_agent.run_executor(cwd=".", prompt="x", query_fn=q))
    assert q.calls["n"] == 1  # no retry


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
