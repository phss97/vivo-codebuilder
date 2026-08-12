"""A listener that runs during resume must never raise.

CrewAI's ``_resume_async_body`` clears the pending-feedback row *before* running
downstream listeners and only re-saves it when the escaping exception is
``HumanFeedbackPending``. So any ordinary exception from ``plan``, ``revise_plan``,
``retry_failed_qa`` or ``skip_failed_package`` leaves the job with no pending row:
permanently unresumable, and silent in the UI. Each of them must degrade into a
payload that re-gates instead.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import codebuilder.main as main
from codebuilder.runtime_qa import validate_plan
from codebuilder.schemas import (
    AuthoritativeAsset,
    IntakeAssessment,
    IntakeQuestion,
    Plan,
    QAReport,
)

_BOOM = RuntimeError("planner exploded")


def _structured_plan() -> Plan:
    criterion = "wp-1-criterion"
    return validate_plan(
        Plan.model_validate(
            {
                "project_name": "demo",
                "mode": "new_project",
                "tech_stack": ["python"],
                "package_name": "demo_pkg",
                "work_packages": [
                    {
                        "id": "wp-1",
                        "title": "Package wp-1",
                        "what_to_build": "Build the wp-1 behavior.",
                        "expected_behavior": "The wp-1 behavior works.",
                        "success_criteria": [
                            {"id": criterion, "description": "wp-1 is implemented."}
                        ],
                        "tests": [
                            {
                                "id": "wp-1-test",
                                "criterion_ids": [criterion],
                                "path": "tests/test_wp_1.py",
                                "test_name": "test_wp_1",
                                "expected_behavior": "Checks wp-1.",
                            }
                        ],
                        "files": [
                            {
                                "path": "src/demo_pkg/wp_1.py",
                                "purpose": "Implements wp-1.",
                                "kind": "source",
                                "public_api": ["build_wp_1() -> str"],
                            },
                            {
                                "path": "tests/test_wp_1.py",
                                "purpose": "Tests wp-1.",
                                "kind": "test",
                                "public_api": [],
                            },
                        ],
                    }
                ],
                "verification_commands": [
                    {
                        "id": "tests",
                        "category": "test",
                        "argv": ["python", "-m", "unittest", "discover"],
                    }
                ],
            }
        )
    )


@pytest.fixture
def flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> main.CodebuilderFlow:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(main.history, "record", lambda *_a, **_k: None)
    instance = main.CodebuilderFlow()
    instance.state.workspace_dir = str(workspace)
    instance.state.project_name = "demo"
    instance.state.brief = "Build the approved demo."
    return instance


def _feedback(text: str = "please change one detail"):
    return type("Feedback", (), {"feedback": text, "output": None})()


def _raising_planner(**_kwargs) -> Plan:
    raise _BOOM


def test_planner_failure_regates_instead_of_stranding_the_job(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main.cc_agent, "run_planner", _raising_planner)

    result = asyncio.run(main.CodebuilderFlow.plan.__wrapped__(flow))

    assert result["plan"] is None
    assert "planner exploded" in result["planner_error"]
    # No plan means the frontend's default action list is empty, so the payload
    # must name its own actions or the card renders with no buttons at all.
    assert result["actions"]
    assert flow.state.status == "awaiting_approval"


def test_revision_failure_without_a_prior_plan_regates(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main.cc_agent, "run_planner", _raising_planner)

    result = asyncio.run(
        main.CodebuilderFlow.revise_plan.__wrapped__(flow, _feedback())
    )

    assert result["plan"] is None
    assert result["actions"]


@pytest.mark.parametrize(
    ("method", "private"),
    [("retry_failed_qa", "_retry_failed_qa"), ("skip_failed_package", "_skip_failed_package")],
)
def test_failed_qa_action_regates_instead_of_stranding_the_job(
    flow: main.CodebuilderFlow,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    private: str,
) -> None:
    flow.state.current_failure = QAReport(passed=False)

    async def boom(*_args, **_kwargs) -> dict:
        raise _BOOM

    monkeypatch.setattr(flow, private, boom)

    result = asyncio.run(getattr(main.CodebuilderFlow, method)(flow, _feedback()))

    assert result == {"route": "qa_exhausted"}
    assert "planner exploded" in flow.state.current_failure.integration_notes


def test_build_without_a_plan_routes_to_the_failure_gate(
    flow: main.CodebuilderFlow,
) -> None:
    result = asyncio.run(main.CodebuilderFlow.build(flow, _feedback("approve")))

    # A missing "route" key would be coerced to "execution_complete" — i.e. a
    # planless job reported as a successful release.
    assert result["route"] == "qa_exhausted"


def test_planner_repair_feeds_the_rejection_back_into_the_next_prompt(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []
    good = _structured_plan()

    async def planner(*, prompt: str, **_kwargs) -> Plan:
        prompts.append(prompt)
        if len(prompts) == 1:
            return Plan(project_name="demo", mode="new_project", plan_markdown="# legacy")
        return good

    monkeypatch.setattr(main.cc_agent, "run_planner", planner)

    plan_obj = asyncio.run(flow._plan_with_repair("PLAN THIS", "plan"))

    assert plan_obj.is_structured
    assert len(prompts) == 2
    assert "Rejected specification" in prompts[1]
    assert "legacy plan" in prompts[1]


def test_planner_repair_is_bounded_and_logs_the_rejected_plan(
    flow: main.CodebuilderFlow,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[int] = []

    async def planner(**_kwargs) -> Plan:
        calls.append(1)
        return Plan(project_name="demo", mode="new_project", plan_markdown="# legacy")

    monkeypatch.setattr(main.cc_agent, "run_planner", planner)
    monkeypatch.setenv("CODEBUILDER_PLANNER_REPAIR_ATTEMPTS", "3")

    with caplog.at_level("WARNING"), pytest.raises(ValueError):
        asyncio.run(flow._plan_with_repair("PLAN THIS", "plan"))

    assert len(calls) == 3
    # One error string cannot tell a planner bug from a validator bug after a
    # 15-minute high-effort run, so the rejected plan itself must reach the log.
    assert '"plan_markdown":"# legacy"' in caplog.text


def test_missing_verification_commands_block_only_a_patch_job(
    flow: main.CodebuilderFlow,
) -> None:
    assessment = IntakeAssessment(ready=True, missing_verification_commands=["pytest?"])

    flow._apply_intake_gate(assessment)
    assert assessment.ready, "a new project has no existing test command to discover"

    flow.state.attachments = [main.Attachment(kind="zip", name="repo.zip")]
    flow._apply_intake_gate(assessment)
    assert not assessment.ready

    ready = IntakeAssessment(
        ready=True, blocking_questions=[IntakeQuestion(id="q1", question="Which API?")]
    )
    flow._apply_intake_gate(ready)
    assert not ready.ready, "a real blocking question always gates"


@pytest.mark.parametrize(
    ("method", "safe_outcome"),
    [
        ("plan", "spec_amend"),
        ("revise_plan", "spec_amend"),
        ("review_qa_failure", "qa_amend"),
    ],
)
def test_classifier_failure_falls_back_to_the_safe_outcome(
    method: str, safe_outcome: str
) -> None:
    # CrewAI collapses to emit[0] on any classifier error, ignoring default_outcome.
    config = getattr(main.CodebuilderFlow, method).__human_feedback_config__
    assert config.emit[0] == safe_outcome


def test_structured_build_failure_regates_instead_of_stranding_the_job(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # build() is the spec_approved resume listener, and the whole structured DAG
    # runs under it through staging helpers that raise WorkspaceSafetyError by
    # design. Unguarded, one symlink from an LLM kills the job for good.
    flow.state.plan = _structured_plan()

    async def boom(*_args, **_kwargs) -> dict:
        raise _BOOM

    monkeypatch.setattr(flow, "_build_structured", boom)

    result = asyncio.run(main.CodebuilderFlow.build(flow, _feedback("approve")))

    assert result == {"route": "qa_exhausted"}
    # "__setup__" clears can_skip — the gate must not offer to skip a package
    # the build never reached.
    assert flow.state.current_package_id == "__setup__"
    assert "planner exploded" in flow.state.current_failure.integration_notes
    # finalize() recomputes status from qa_report alone and reads None as "done",
    # so a degraded gate that set only current_failure would report this dead
    # job as a successful completion.
    assert flow.state.qa_report is not None and not flow.state.qa_report.passed


def test_terminate_survives_a_failed_quarantine_bundle(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # qa_terminate is review_qa_failure's default_outcome, so empty feedback
    # lands here. Losing the evidence bundle is survivable; losing the job isn't.
    def boom() -> None:
        raise _BOOM

    monkeypatch.setattr(flow, "_prepare_quarantine", boom)

    assert flow.terminate_failed_qa() == {"route": "quarantine_ready"}
    assert "planner exploded" in flow.state.current_failure.integration_notes
    assert flow.state.status == "failed"


@pytest.mark.parametrize("bad_path", ["dist/wp_1.py", "build/wp_1.py"])
def test_validate_plan_rejects_paths_promotion_would_refuse(bad_path: str) -> None:
    # Approval used to accept any relative path, while promote_files refuses
    # anything _is_excluded covers — so the plan detonated after the paid
    # test-author call instead of at the gate.
    plan = _structured_plan()
    plan.work_packages[0].files[0].path = bad_path

    with pytest.raises(ValueError, match="excluded from package staging"):
        validate_plan(plan)


def test_validate_plan_rejects_an_asset_the_executor_would_never_see() -> None:
    # copy_clean_tree drops excluded paths silently, so an "authoritative"
    # reference under one is invisible to the agent told to obey it.
    plan = _structured_plan()
    plan.authoritative_assets = [AuthoritativeAsset(path="node_modules/schema.json")]

    with pytest.raises(ValueError, match="invisible to the executor"):
        validate_plan(plan)
