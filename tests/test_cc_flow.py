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
from codebuilder.schemas import Plan, QAReport
from codebuilder.tools.git_tool import _HARNESS_EXCLUDES
from codebuilder.tools.s3_artifacts import SKIP_DIRS


# --- fake SDK messages / query --------------------------------------------


class _FakeText:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAssistant:
    def __init__(self, text: str, usage=None) -> None:
        self.content = [_FakeText(text)]
        self.usage = usage  # per-turn token dict, or None


class _FakeResult:
    def __init__(self, structured_output=None, subtype="success", is_error=False,
                 api_error_status=None, usage=None, total_cost_usd=None,
                 num_turns=None, duration_ms=None) -> None:
        self.structured_output = structured_output
        self.subtype = subtype
        self.is_error = is_error
        self.api_error_status = api_error_status
        self.usage = usage
        self.total_cost_usd = total_cost_usd
        self.num_turns = num_turns
        self.duration_ms = duration_ms


def _make_query(messages):
    async def _q(**_kwargs):
        for m in messages:
            yield m

    return _q


def _make_capturing_query(messages):
    """Like _make_query but records the ClaudeAgentOptions it was called with,
    so tests can assert effort/model wiring."""
    async def _q(*, prompt, options):  # noqa: ARG001
        _q.captured["options"] = options
        for m in messages:
            yield m

    _q.captured = {}
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


# --- effort knob -----------------------------------------------------------


def test_executor_effort_default_is_medium():
    q = _make_capturing_query([_FakeResult()])
    asyncio.run(cc_agent.run_executor(cwd=".", prompt="x", query_fn=q))
    assert q.captured["options"].effort == "medium"


def test_planner_effort_default_is_high():
    q = _make_capturing_query([_FakeResult(structured_output=VALID_PLAN)])
    asyncio.run(cc_agent.run_planner(cwd=".", prompt="x", query_fn=q))
    assert q.captured["options"].effort == "high"


def test_executor_effort_override():
    q = _make_capturing_query([_FakeResult()])
    asyncio.run(cc_agent.run_executor(cwd=".", prompt="x", effort="xhigh", query_fn=q))
    assert q.captured["options"].effort == "xhigh"


def test_effort_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("CB_TEST_EFFORT", "turbo")
    assert cc_agent._effort("CB_TEST_EFFORT", "high") == "high"  # invalid → default
    monkeypatch.setenv("CB_TEST_EFFORT", "low")
    assert cc_agent._effort("CB_TEST_EFFORT", "high") == "low"   # valid → honored


# --- usage / cost logging --------------------------------------------------


def test_on_usage_fires_on_success():
    seen: list = []
    q = _make_query([_FakeResult(
        structured_output=VALID_PLAN, total_cost_usd=0.42, num_turns=3,
        usage={"input_tokens": 100, "output_tokens": 50,
               "cache_read_input_tokens": 10, "cache_creation_input_tokens": 5},
    )])
    asyncio.run(cc_agent.run_planner(cwd=".", prompt="x", query_fn=q, on_usage=seen.append))
    assert len(seen) == 1
    assert seen[0]["stage"] == "planner"
    assert seen[0]["cost_usd"] == 0.42
    assert seen[0]["input_tokens"] == 100 and seen[0]["output_tokens"] == 50


def test_on_usage_fires_on_failure():
    seen: list = []

    async def q(**_kwargs):
        yield _FakeResult(is_error=True, subtype="success", api_error_status=400,
                          total_cost_usd=1.23, usage={"input_tokens": 9})
        raise _FakeProcessError("exit 1")

    with pytest.raises(CCAgentError):
        asyncio.run(cc_agent.run_executor(cwd=".", prompt="x", query_fn=q, on_usage=seen.append))
    assert seen and seen[0]["cost_usd"] == 1.23  # wasted spend is surfaced


# --- cost budget cap -------------------------------------------------------


def test_budget_cap_trips_and_stops():
    turn = {"output_tokens": 1_000_000}  # ~$10/turn at default $10/MTok output
    q = _make_query([
        _FakeAssistant("turn 1", usage=turn),
        _FakeAssistant("turn 2", usage=turn),
        _FakeResult(),
    ])
    with pytest.raises(cc_agent.CCBudgetExceeded) as ei:
        asyncio.run(cc_agent.run_executor(cwd=".", prompt="x", budget_usd=5.0, query_fn=q))
    assert ei.value.cost_usd >= 5.0


def test_budget_not_tripped_when_under():
    q = _make_query([_FakeAssistant("t", usage={"output_tokens": 1000}), _FakeResult()])
    out = asyncio.run(cc_agent.run_executor(cwd=".", prompt="x", budget_usd=100.0, query_fn=q))
    assert "t" in out


def test_deterministic_changelog(tmp_path):
    flow = main.CodebuilderFlow()
    flow._write_deterministic_changelog(str(tmp_path), Plan.model_validate(VALID_PLAN), 12.5)
    md = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "cost budget" in md and "$12.50" in md and "# Plan" in md


def test_build_budget_exceeded_writes_changelog(tmp_path, monkeypatch):
    # build() catches CCBudgetExceeded, writes CHANGELOG.md, and marks failed —
    # so finalize can still deliver the partial package.
    monkeypatch.setenv("CODEBUILDER_BUDGET_CHANGELOG", "deterministic")
    monkeypatch.setenv("CODEBUILDER_MAX_RUN_COST_USD", "1")
    monkeypatch.setenv("CODEBUILDER_HISTORY_ENABLED", "false")

    async def _boom(**_kwargs):
        raise cc_agent.CCBudgetExceeded(1.5, "partial transcript")

    monkeypatch.setattr(cc_agent, "run_executor", _boom)

    flow = main.CodebuilderFlow()
    flow.state.plan = Plan.model_validate(VALID_PLAN)  # new_project
    flow.state.workspace_dir = str(tmp_path)
    (tmp_path / "output").mkdir()

    class _Prior:
        feedback = "approved"

    asyncio.run(flow.build(_Prior()))
    assert flow.state.status == "failed"
    assert (tmp_path / "output" / "CHANGELOG.md").is_file()
    assert "budget" in flow.state.qa_report.integration_notes.lower()


def test_finalize_zips_partial_on_failure(tmp_path, monkeypatch):
    # The critical fix: a failed/budget-stopped build must still deliver its
    # partial package (with CHANGELOG.md), not return empty-handed.
    monkeypatch.setenv("CODEBUILDER_HISTORY_ENABLED", "false")
    build_dir = tmp_path / "output"
    build_dir.mkdir()
    (build_dir / "partial.py").write_text("x = 1\n")
    (build_dir / "CHANGELOG.md").write_text("# stopped at budget\n")

    flow = main.CodebuilderFlow()
    flow.state.plan = Plan.model_validate(VALID_PLAN)
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "failed"
    flow.state.qa_report = QAReport(passed=False, integration_notes="stopped at budget")
    flow._build_dir = str(build_dir)

    payload = asyncio.run(flow.finalize())
    assert payload["status"] == "failed" and payload["qa_passed"] is False
    assert payload.get("zip_path"), "partial package should still be zipped on failure"
    assert (tmp_path / "demo.zip").is_file()


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
def test_run_final_qa_skips_tests_when_disabled(tmp_path, monkeypatch):
    # run_tests=False → the pytest sweep is skipped; ruff still gates. A failing
    # test that WOULD fail if run is proof the sweep didn't run.
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / "mod.py").write_text("x = 1\n")
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert False\n")
    report = run_final_qa(str(tmp_path), run_tests=False)
    assert report.passed, report.integration_notes
    assert "disabled" in report.test_output.lower()


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
