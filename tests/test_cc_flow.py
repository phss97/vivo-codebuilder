"""Tests for the Claude engine, deterministic package QA, and Flow contracts."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import zipfile

import pytest

import codebuilder.main as main
import codebuilder.runtime_qa as runtime_qa
from codebuilder import cc_agent
from codebuilder.cc_agent import CCAgentError
from codebuilder.runtime_qa import run_final_qa, validate_plan
from codebuilder.schemas import Attachment, Plan, ProductionReview, QAReport
from codebuilder.tools.git_tool import _HARNESS_EXCLUDES
from codebuilder.tools.s3_artifacts import SKIP_DIRS


@pytest.fixture(autouse=True)
def _disable_external_artifact_upload(monkeypatch):
    """Keep local .env settings from turning unit tests into network tests."""
    monkeypatch.delenv("CODEBUILDER_ARTIFACT_BUCKET", raising=False)


# --- fake SDK messages / query --------------------------------------------


class _FakeText:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAssistant:
    def __init__(self, text: str, usage=None) -> None:
        self.content = [_FakeText(text)]
        self.usage = usage  # per-turn token dict, or None


class _FakeResult:
    def __init__(
        self,
        structured_output=None,
        subtype="success",
        is_error=False,
        api_error_status=None,
        usage=None,
        total_cost_usd=None,
        num_turns=None,
        duration_ms=None,
    ) -> None:
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

VALID_REVIEW = {"passed": True, "issues": []}


# --- validate_plan ---------------------------------------------------------


def test_validate_plan_ok():
    assert validate_plan(Plan.model_validate(VALID_PLAN)).mode == "new_project"


def test_validate_plan_rejects_empty_markdown():
    with pytest.raises(ValueError):
        validate_plan(Plan.model_validate({**VALID_PLAN, "plan_markdown": "   "}))


def test_validate_plan_rejects_non_plan():
    with pytest.raises(ValueError):
        validate_plan(None)  # type: ignore[arg-type]


# --- attached-project preflight -------------------------------------------


def test_ingest_runs_nonterminal_preflight_before_planning(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "WORKSPACE_ROOT", tmp_path)
    calls: list[dict] = []

    def _materialize(_attachments, workspace_dir):
        project = main.Path(workspace_dir) / "inputs" / "project"
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text(
            "[project]\nname='demo'\nversion='0.1'\n"
        )
        return [{"kind": "zip", "name": "project.zip", "path": str(project)}]

    failed = QAReport(
        passed=False,
        lint_output="ruff check: F401 unused import",
        type_output="src/app.py:1: error: incompatible type",
        test_output="164 passed, 1 failed",
        integration_notes=".env.example is missing TERRA_DB_URL",
    )

    def _preflight(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return failed

    monkeypatch.setattr(main.attachment_tool, "materialize", _materialize)
    monkeypatch.setattr(main, "run_final_qa", _preflight)

    flow = main.CodebuilderFlow()
    flow.state.session_id = "preflight"
    flow.state.project_name = "terra-rpa"
    flow.state.brief = "Patch this RPA project"
    flow.state.attachments = [Attachment(kind="zip", name="project.zip")]
    flow.ingest()

    assert flow.state.status == "planning"
    assert flow.state.preflight_qa_report == failed
    assert calls and calls[0]["require_installable"] is True
    assert calls[0]["require_typecheck"] is True
    planner_prompt = main._planner_prompt(flow.state)
    executor_prompt = main._executor_prompt(
        flow.state,
        Plan.model_validate({**VALID_PLAN, "mode": "patch_existing", "domain": "rpa"}),
    )
    for prompt in (planner_prompt, executor_prompt):
        assert "F401 unused import" in prompt
        assert "TERRA_DB_URL" in prompt


def test_ingest_without_attached_project_skips_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "WORKSPACE_ROOT", tmp_path)

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("preflight should not run")

    monkeypatch.setattr(main, "run_final_qa", _unexpected)
    flow = main.CodebuilderFlow()
    flow.state.session_id = "new-project"
    flow.ingest()
    assert flow.state.preflight_qa_report is None


def test_preflight_prompt_is_bounded_per_category():
    prompt = runtime_qa.qa_report_for_prompt(
        QAReport(
            passed=False,
            lint_output="L" * 10_000,
            type_output="M" * 10_000,
            test_output="T" * 10_000,
            integration_notes="I" * 10_000,
        )
    )
    assert len(prompt) < 12_000
    assert prompt.count("[truncated ") == 4


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


def test_run_reviewer_is_structured_and_read_only():
    q = _make_capturing_query([_FakeResult(structured_output=VALID_REVIEW)])

    review = asyncio.run(cc_agent.run_reviewer(cwd=".", prompt="x", query_fn=q))

    assert review == ProductionReview(passed=True)
    options = q.captured["options"]
    assert "Read" in options.allowed_tools
    assert "Bash" in options.disallowed_tools
    assert options.permission_mode == "default"


# --- run_executor ----------------------------------------------------------


def test_run_executor_transcript_and_progress():
    seen: list = []
    q = _make_query(
        [_FakeAssistant("wrote file A"), _FakeAssistant("ran tests"), _FakeResult()]
    )
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
    assert cc_agent._effort("CB_TEST_EFFORT", "high") == "low"  # valid → honored


# --- usage / cost logging --------------------------------------------------


def test_on_usage_fires_on_success():
    seen: list = []
    q = _make_query(
        [
            _FakeResult(
                structured_output=VALID_PLAN,
                total_cost_usd=0.42,
                num_turns=3,
                usage={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 10,
                    "cache_creation_input_tokens": 5,
                },
            )
        ]
    )
    asyncio.run(
        cc_agent.run_planner(cwd=".", prompt="x", query_fn=q, on_usage=seen.append)
    )
    assert len(seen) == 1
    assert seen[0]["stage"] == "planner"
    assert seen[0]["cost_usd"] == 0.42
    assert seen[0]["input_tokens"] == 100 and seen[0]["output_tokens"] == 50


def test_on_usage_fires_on_failure():
    seen: list = []

    async def q(**_kwargs):
        yield _FakeResult(
            is_error=True,
            subtype="success",
            api_error_status=400,
            total_cost_usd=1.23,
            usage={"input_tokens": 9},
        )
        raise _FakeProcessError("exit 1")

    with pytest.raises(CCAgentError):
        asyncio.run(
            cc_agent.run_executor(cwd=".", prompt="x", query_fn=q, on_usage=seen.append)
        )
    assert seen and seen[0]["cost_usd"] == 1.23  # wasted spend is surfaced


# --- cost budget cap -------------------------------------------------------


def test_budget_cap_trips_and_stops():
    turn = {"output_tokens": 1_000_000}  # ~$10/turn at default $10/MTok output
    q = _make_query(
        [
            _FakeAssistant("turn 1", usage=turn),
            _FakeAssistant("turn 2", usage=turn),
            _FakeResult(),
        ]
    )
    with pytest.raises(cc_agent.CCBudgetExceeded) as ei:
        asyncio.run(
            cc_agent.run_executor(cwd=".", prompt="x", budget_usd=5.0, query_fn=q)
        )
    assert ei.value.cost_usd >= 5.0


def test_budget_not_tripped_when_under():
    q = _make_query([_FakeAssistant("t", usage={"output_tokens": 1000}), _FakeResult()])
    out = asyncio.run(
        cc_agent.run_executor(cwd=".", prompt="x", budget_usd=100.0, query_fn=q)
    )
    assert "t" in out


def test_build_budget_exceeded_uses_no_wrapup_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_MAX_RUN_COST_USD", "1")
    monkeypatch.setenv("CODEBUILDER_HISTORY_ENABLED", "false")
    calls = {"count": 0}

    async def _boom(**_kwargs):
        calls["count"] += 1
        raise cc_agent.CCBudgetExceeded(1.5, "partial transcript")

    monkeypatch.setattr(cc_agent, "run_executor", _boom)

    flow = main.CodebuilderFlow()
    flow.state.plan = Plan.model_validate(VALID_PLAN)  # new_project
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    (tmp_path / "output").mkdir()

    class _Prior:
        feedback = "approved"

    asyncio.run(flow.build(_Prior()))
    assert flow.state.status == "failed"
    assert calls["count"] == 1
    assert not (tmp_path / "output" / "CHANGELOG.md").exists()
    assert "budget" in flow.state.qa_report.integration_notes.lower()
    monkeypatch.setattr(
        main.CodebuilderFlow,
        "_run_final_qa",
        lambda *_: QAReport(passed=True, integration_notes="partial files checked"),
    )
    payload = asyncio.run(flow.finalize())
    assert payload["status"] == "failed" and payload["zip_path"]
    with zipfile.ZipFile(tmp_path / "demo.zip") as archive:
        report = archive.read("demo/CODEBUILDER_REPORT.md").decode()
    assert "cost budget" in report


def test_builder_crash_delivers_partial_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_HISTORY_ENABLED", "false")

    async def _crash(*, cwd, **_kwargs):
        (main.Path(cwd) / "partial.py").write_text("x = 1\n")
        raise RuntimeError("builder crashed")

    monkeypatch.setattr(cc_agent, "run_executor", _crash)
    flow = main.CodebuilderFlow()
    flow.state.plan = Plan.model_validate(VALID_PLAN)
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    (tmp_path / "output").mkdir()

    class _Prior:
        feedback = "approved"

    asyncio.run(flow.build(_Prior()))
    monkeypatch.setattr(
        main.CodebuilderFlow,
        "_run_final_qa",
        lambda *_: QAReport(passed=True, integration_notes="partial files checked"),
    )
    payload = asyncio.run(flow.finalize())
    assert payload["status"] == "failed"
    with zipfile.ZipFile(tmp_path / "demo.zip") as archive:
        report = archive.read("demo/CODEBUILDER_REPORT.md").decode()
    assert "builder crashed" in report


def test_finalize_zips_partial_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_HISTORY_ENABLED", "false")
    build_dir = tmp_path / "output"
    build_dir.mkdir()
    (build_dir / "partial.py").write_text("x = 1\n")

    flow = main.CodebuilderFlow()
    flow.state.plan = Plan.model_validate(VALID_PLAN)
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "failed"
    flow.state.qa_report = QAReport(passed=False, integration_notes="stopped at budget")
    flow._build_dir = str(build_dir)
    calls = {"qa": 0}

    def _qa(_flow, _path):
        calls["qa"] += 1
        return QAReport(passed=True, integration_notes="checks passed")

    monkeypatch.setattr(
        main.CodebuilderFlow,
        "_run_final_qa",
        _qa,
    )

    payload = asyncio.run(flow.finalize())
    assert payload["status"] == "failed" and payload["qa_passed"] is False
    assert calls["qa"] == 1, "final QA must run even after the builder failed"
    assert payload.get("zip_path"), "partial package should still be zipped on failure"
    assert (tmp_path / "demo.zip").is_file()
    with zipfile.ZipFile(tmp_path / "demo.zip") as archive:
        names = archive.namelist()
        assert "demo/CODEBUILDER_REPORT.md" in names
        report = archive.read("demo/CODEBUILDER_REPORT.md").decode()
    assert "stopped at budget" in report
    assert not (build_dir / "CODEBUILDER_REPORT.md").exists()


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


# --- strict package QA -----------------------------------------------------

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
def test_run_final_qa_requires_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / "mod.py").write_text("x = 1\n")
    report = run_final_qa(str(tmp_path))
    assert not report.passed
    assert "no tests collected" in report.test_output.lower()


@pytest.mark.skipif(not _ruff, reason="ruff not available")
def test_run_final_qa_checks_full_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / "legacy.py").write_text("import os\n\nx = 1\n")
    (tmp_path / "new_feature.py").write_text("def feature():\n    return 42\n")
    (tmp_path / "test_feature.py").write_text(
        "from new_feature import feature\n\n\ndef test_feature():\n    assert feature() == 42\n"
    )
    report = run_final_qa(str(tmp_path))
    assert not report.passed
    assert "F401" in report.lint_output


@pytest.mark.skipif(not _ruff, reason="ruff not available")
def test_run_final_qa_checks_ruff_format(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / "mod.py").write_text("items=[1,2,3]\n")
    (tmp_path / "test_mod.py").write_text(
        "from mod import items\n\n\ndef test_items():\n    assert len(items) == 3\n"
    )
    report = run_final_qa(str(tmp_path))
    assert not report.passed
    assert "ruff format --check" in report.lint_output


def test_run_final_qa_aggregates_checks_when_pytest_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    monkeypatch.setattr(runtime_qa.LintRunnerTool, "_run", lambda *_: "lint failed")
    monkeypatch.setattr(
        runtime_qa.TypeCheckRunnerTool, "_run", lambda *_: "mypy failed"
    )
    monkeypatch.setattr(runtime_qa.TestRunnerTool, "_run", lambda *_: "PASS\n1 passed")
    monkeypatch.setattr(runtime_qa, "check_env_example", lambda *_: "env failed")
    monkeypatch.setattr(runtime_qa, "check_runtime_contract", lambda *_: "deps failed")

    report = run_final_qa(str(tmp_path), require_typecheck=True)
    assert not report.passed
    assert report.test_output.startswith("PASS")
    assert report.lint_output == "lint failed"
    assert report.type_output == "mypy failed"
    assert "env failed" in report.integration_notes
    assert "deps failed" in report.integration_notes


def test_env_example_prefix_mismatch_is_blocking(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / ".env.example").write_text("DB_URL=sqlite:///demo.db\n")
    (tmp_path / "settings.py").write_text(
        "from pydantic_settings import BaseSettings, SettingsConfigDict\n\n"
        "class Settings(BaseSettings):\n"
        "    model_config = SettingsConfigDict(env_prefix='TERRA_')\n"
        "    db_url: str\n"
    )
    (tmp_path / "test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    report = run_final_qa(str(tmp_path))
    assert not report.passed
    assert "TERRA_DB_URL" in report.integration_notes


def test_rpa_runtime_dependencies_and_entrypoint_are_blocking(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0.1'\ndependencies=[]\n"
        "[project.scripts]\ndemo='missing_package.cli:main'\n"
    )
    (tmp_path / "app.py").write_text(
        "import win32com.client  # type: ignore[import-not-found]\n\n"
        "DATABASE_URL = 'mssql+pyodbc://host/db'\n"
    )
    (tmp_path / "test_smoke.py").write_text("def test_smoke():\n    assert True\n")
    report = run_final_qa(str(tmp_path), require_typecheck=True)
    assert not report.passed
    assert "`pyodbc`" in report.integration_notes
    assert "`pywin32`" in report.integration_notes
    assert "missing_package.cli" in report.integration_notes


def test_pywin32_must_be_windows_scoped(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0.1'\ndependencies=['pywin32>=306']\n"
    )
    (tmp_path / "app.py").write_text("import win32com.client\n")
    output = runtime_qa.check_runtime_contract(str(tmp_path))
    assert "must be scoped to Windows" in output


def test_rpa_production_contract_catches_dynamic_wiring_and_unused_lifecycle(
    tmp_path,
):
    source = tmp_path / "src" / "demo"
    source.mkdir(parents=True)
    (source / "settings.py").write_text(
        "from pydantic_settings import BaseSettings\n\n"
        "class Settings(BaseSettings):\n"
        "    sap_endpoint: str\n"
        "    sap_password_secret: str\n"
    )
    (source / "sap_client.py").write_text(
        "from typing import Any\n"
        "from .settings import Settings\n\n"
        "class SapClientImpl:\n"
        "    def __init__(self, settings: Settings, secret_provider: Any):\n"
        "        self._settings = settings\n"
        "        self._secret_provider = secret_provider\n"
        "    def login(self):\n"
        "        getattr(self._settings, 'sap_system_id', '')\n"
        "        return getattr(self._secret_provider, 'sap_password', '')\n"
        "    def logout(self):\n"
        "        return None\n"
    )

    output = runtime_qa.check_rpa_production_contract(str(tmp_path))

    assert "sap_system_id" in output
    assert "secret_provider" in output and "typed Any" in output
    assert "production source never calls: login, logout" in output


def test_rpa_production_contract_accepts_typed_and_orchestrated_client(tmp_path):
    source = tmp_path / "src" / "demo"
    source.mkdir(parents=True)
    (source / "settings.py").write_text(
        "from pydantic_settings import BaseSettings\n\n"
        "class Settings(BaseSettings):\n"
        "    sap_endpoint: str\n"
        "    sap_password_secret: str\n"
    )
    (source / "sap_client.py").write_text(
        "from typing import Protocol\n"
        "from .settings import Settings\n\n"
        "class SecretProvider(Protocol):\n"
        "    def get_secret(self, name: str) -> str: ...\n\n"
        "class SapClientImpl:\n"
        "    def __init__(self, settings: Settings, secret_provider: SecretProvider):\n"
        "        self._settings = settings\n"
        "        self._secret_provider = secret_provider\n"
        "    def login(self):\n"
        "        return self._secret_provider.get_secret(self._settings.sap_password_secret)\n"
        "    def logout(self):\n"
        "        return None\n"
    )
    (source / "orchestrator.py").write_text(
        "def run(client):\n"
        "    client.login()\n"
        "    try:\n"
        "        return 0\n"
        "    finally:\n"
        "        client.logout()\n"
    )

    assert runtime_qa.check_rpa_production_contract(str(tmp_path)) == "PASS"


def test_rpa_qa_reruns_tests_with_example_env_and_restores_workspace(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CODEBUILDER_PROVISION_PROJECT_ENV", "0")
    (tmp_path / ".env.example").write_text("TERRA_DATABASE_URL=example\n")
    calls: list[bool] = []

    def _tests(tool, _path):
        active = (tmp_path / ".env").exists()
        calls.append(active)
        return "settings leaked from .env.example" if active else "PASS\n1 passed"

    monkeypatch.setattr(runtime_qa.LintRunnerTool, "_run", lambda *_: "PASS")
    monkeypatch.setattr(runtime_qa.TypeCheckRunnerTool, "_run", lambda *_: "PASS")
    monkeypatch.setattr(runtime_qa.TestRunnerTool, "_run", _tests)
    monkeypatch.setattr(runtime_qa, "check_env_example", lambda *_: "PASS")
    monkeypatch.setattr(runtime_qa, "check_runtime_contract", lambda *_: "PASS")
    monkeypatch.setattr(runtime_qa, "check_rpa_production_contract", lambda *_: "PASS")

    report = run_final_qa(str(tmp_path), require_typecheck=True)

    assert not report.passed
    assert calls == [False, True]
    assert "settings leaked" in report.test_output
    assert "settings leaked" in report.integration_notes
    assert not (tmp_path / ".env").exists()


def test_rpa_entrypoint_help_smoke_is_blocking(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0.1'\ndependencies=[]\n"
        "[project.scripts]\ndemo='demo:main'\n"
    )
    (tmp_path / ".env").write_text("REAL_DATABASE_URL=do-not-use\n")
    (tmp_path / "demo.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n\n"
        "def main():\n"
        "    if Path('.env').exists():\n"
        "        return 3\n"
        "    argparse.ArgumentParser().parse_args()\n"
    )
    assert runtime_qa.check_runtime_contract(str(tmp_path), True) == "PASS"

    (tmp_path / "demo.py").write_text("def main():\n    return 2\n")
    output = runtime_qa.check_runtime_contract(str(tmp_path), True)
    assert "demo --help" in output


def test_final_qa_uses_rpa_heuristic_when_planner_domain_is_empty(
    tmp_path, monkeypatch
):
    source = tmp_path / "src" / "demo"
    source.mkdir(parents=True)
    for name in ("orchestrator.py", "producer.py", "consumer.py"):
        (source / name).write_text("x = 1\n")
    captured: dict = {}

    def _qa(path, **kwargs):
        captured.update(path=path, **kwargs)
        return QAReport(passed=True)

    monkeypatch.setattr(main, "run_final_qa", _qa)
    flow = main.CodebuilderFlow()
    flow.state.plan = Plan.model_validate(VALID_PLAN)

    flow._run_final_qa(str(tmp_path))

    assert captured["require_typecheck"] is True


def test_successful_archive_has_no_failure_report(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_HISTORY_ENABLED", "false")
    build_dir = tmp_path / "output"
    build_dir.mkdir()
    (build_dir / "app.py").write_text("x = 1\n")

    flow = main.CodebuilderFlow()
    flow.state.plan = Plan.model_validate(VALID_PLAN)
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "executing"
    flow._build_dir = str(build_dir)
    monkeypatch.setattr(
        main.CodebuilderFlow,
        "_run_final_qa",
        lambda *_: QAReport(passed=True, integration_notes="all checks passed"),
    )

    payload = asyncio.run(flow.finalize())
    assert payload["status"] == "done"
    with zipfile.ZipFile(tmp_path / "demo.zip") as archive:
        assert "demo/CODEBUILDER_REPORT.md" not in archive.namelist()


def test_rpa_semantic_failure_uses_existing_repair_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_HISTORY_ENABLED", "false")
    monkeypatch.setenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", "1")
    build_dir = tmp_path / "output"
    build_dir.mkdir()
    (build_dir / "app.py").write_text("x = 1\n")
    prompts: list[str] = []
    reviews = iter(
        [
            ProductionReview(
                passed=False,
                issues=["orchestrator.py never calls SapClient.login()"],
            ),
            ProductionReview(passed=True),
        ]
    )

    async def _reviewer(**_kwargs):
        return next(reviews)

    async def _executor(**kwargs):
        prompts.append(kwargs["prompt"])
        return "fixed"

    flow = main.CodebuilderFlow()
    flow.state.plan = Plan.model_validate({**VALID_PLAN, "domain": "rpa"})
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "executing"
    flow._build_dir = str(build_dir)
    monkeypatch.setattr(
        main.CodebuilderFlow,
        "_run_final_qa",
        lambda *_: QAReport(passed=True, integration_notes="deterministic PASS"),
    )
    monkeypatch.setattr(main.cc_agent, "run_reviewer", _reviewer)
    monkeypatch.setattr(main.cc_agent, "run_executor", _executor)

    payload = asyncio.run(flow.finalize())

    assert payload["status"] == "done"
    assert flow.state.final_qa_repair_attempts == 1
    assert len(prompts) == 1
    assert "never calls SapClient.login" in prompts[0]


def test_rpa_reviewer_error_fails_closed_without_spending_on_repair(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CODEBUILDER_HISTORY_ENABLED", "false")
    build_dir = tmp_path / "output"
    build_dir.mkdir()
    (build_dir / "app.py").write_text("x = 1\n")
    executor_called = False

    async def _reviewer(**_kwargs):
        raise CCAgentError("review service unavailable")

    async def _executor(**_kwargs):
        nonlocal executor_called
        executor_called = True
        return "unexpected"

    flow = main.CodebuilderFlow()
    flow.state.plan = Plan.model_validate({**VALID_PLAN, "domain": "rpa"})
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "executing"
    flow._build_dir = str(build_dir)
    monkeypatch.setattr(
        main.CodebuilderFlow,
        "_run_final_qa",
        lambda *_: QAReport(passed=True, integration_notes="deterministic PASS"),
    )
    monkeypatch.setattr(main.cc_agent, "run_reviewer", _reviewer)
    monkeypatch.setattr(main.cc_agent, "run_executor", _executor)

    payload = asyncio.run(flow.finalize())

    assert payload["status"] == "failed"
    assert not executor_called
    assert "review service unavailable" in payload["qa_report"]["integration_notes"]
    with zipfile.ZipFile(tmp_path / "demo.zip") as archive:
        assert "demo/CODEBUILDER_REPORT.md" in archive.namelist()


def test_final_qa_failure_is_archived_with_report(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_HISTORY_ENABLED", "false")
    monkeypatch.setenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", "0")
    build_dir = tmp_path / "output"
    build_dir.mkdir()
    (build_dir / "broken.py").write_text("x = 1\n")

    flow = main.CodebuilderFlow()
    flow.state.plan = Plan.model_validate(VALID_PLAN)
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "executing"
    flow._build_dir = str(build_dir)
    monkeypatch.setattr(
        main.CodebuilderFlow,
        "_run_final_qa",
        lambda *_: QAReport(
            passed=False, lint_output="F401", integration_notes="lint failed"
        ),
    )

    payload = asyncio.run(flow.finalize())
    assert payload["status"] == "failed"
    with zipfile.ZipFile(tmp_path / "demo.zip") as archive:
        report = archive.read("demo/CODEBUILDER_REPORT.md").decode()
    assert "F401" in report and "Original approved plan" in report
