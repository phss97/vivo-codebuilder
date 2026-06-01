"""Revise/amend path: a planner failure must re-gate with the prior plan, never
brick the job, and the targeted amend crew must reject empty plans.

Background: on resume, CrewAI clears the pending-feedback row *before* running the
"amend" listener and re-raises any non-HumanFeedbackPending exception. If
``revise_plan`` raised, the job would be permanently unresumable. So a revision
failure must fall back to the prior plan and let ``@human_feedback`` re-gate.
"""

from types import SimpleNamespace

from codebuilder.crews.planner_crew import PlannerCrew
from codebuilder.crews.planner_crew.planner_crew import _require_nonempty_plan
from codebuilder.main import CodebuilderFlow
from codebuilder.schemas import FileSkeleton, Plan, SubTask


# ``revise_plan`` is wrapped by @listen + @human_feedback; reach the raw body so a
# unit test doesn't trigger the feedback request (which would raise
# HumanFeedbackPending). ListenMethod.unwrap() -> human_feedback sync_wrapper;
# .__wrapped__ -> the original revise_plan function.
_REVISE_BODY = CodebuilderFlow.revise_plan.unwrap().__wrapped__


def _valid_plan(n: int = 3) -> Plan:
    return Plan(
        project_name="terra",
        mode="new_project",
        tech_stack=["python"],
        domain="rpa",
        subtasks=[
            SubTask(
                id=f"s{i:02d}",
                title=f"package {i}",
                description="d",
                files=[FileSkeleton(path=f"src/pkg/f{i}.py", purpose="Module.")],
                test_criteria="t",
            )
            for i in range(1, n + 1)
        ],
    )


def _flow_with_prior(prior_plan: Plan) -> CodebuilderFlow:
    flow = CodebuilderFlow()
    flow.state.plan = prior_plan
    return flow


def _fake_prior(plan: Plan, feedback: str = "please add a relatorio queue"):
    # Mirrors the HumanFeedbackResult passed to the "amend" listener.
    return SimpleNamespace(feedback=feedback, output=plan.model_dump(), outcome="amend")


class _FakeResult:
    def __init__(self, pydantic) -> None:
        self.pydantic = pydantic


class _FakeCrew:
    def __init__(self, result: _FakeResult) -> None:
        self._result = result

    def kickoff(self, inputs=None):
        return self._result


def test_revise_plan_recovers_when_amend_crew_raises(monkeypatch) -> None:
    prior = _valid_plan()
    flow = _flow_with_prior(prior)

    def boom(self):
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(PlannerCrew, "amend_crew", boom)

    result = _REVISE_BODY(flow, _fake_prior(prior))  # must NOT raise

    assert isinstance(result, dict)
    assert "Automatic plan revision failed" in result["open_questions"][0]
    # Prior work packages preserved verbatim so the job can be re-gated/approved.
    assert [s["id"] for s in result["subtasks"]] == [s.id for s in prior.subtasks]
    assert flow.state.status == "awaiting_approval"
    assert flow.state.amendments == "please add a relatorio queue"
    assert flow.state.amend_cycles == 1


def test_revise_plan_recovers_when_revised_plan_is_invalid(monkeypatch) -> None:
    prior = _valid_plan()
    flow = _flow_with_prior(prior)

    # Crew "succeeds" but returns an empty-subtasks Plan -> validate_plan raises.
    empty = Plan(project_name="terra", mode="new_project", tech_stack=["python"], subtasks=[])
    monkeypatch.setattr(PlannerCrew, "amend_crew", lambda self: _FakeCrew(_FakeResult(empty)))

    result = _REVISE_BODY(flow, _fake_prior(prior))

    assert isinstance(result, dict)
    assert "Automatic plan revision failed" in result["open_questions"][0]
    assert [s["id"] for s in result["subtasks"]] == [s.id for s in prior.subtasks]
    assert flow.state.status == "awaiting_approval"


def test_revise_plan_applies_valid_revision(monkeypatch) -> None:
    prior = _valid_plan(2)
    flow = _flow_with_prior(prior)

    revised = _valid_plan(4)  # a larger, valid revised plan
    monkeypatch.setattr(PlannerCrew, "amend_crew", lambda self: _FakeCrew(_FakeResult(revised)))

    result = _REVISE_BODY(flow, _fake_prior(prior))

    assert [s["id"] for s in result["subtasks"]] == [s.id for s in revised.subtasks]
    # No failure note injected on the happy path.
    assert not any("Automatic plan revision failed" in q for q in result["open_questions"])
    assert flow.state.plan.subtasks[-1].id == "s04"


def test_prior_plan_snapshot_prefers_state_plan() -> None:
    prior_plan = _valid_plan()
    flow = _flow_with_prior(prior_plan)

    snapshot = flow._prior_plan_snapshot(_fake_prior(prior_plan))

    assert snapshot is not None
    assert snapshot is not flow.state.plan  # deep copy, not the live object
    assert [s.id for s in snapshot.subtasks] == [s.id for s in prior_plan.subtasks]


def test_prior_plan_snapshot_falls_back_to_feedback_output() -> None:
    prior_plan = _valid_plan()
    flow = CodebuilderFlow()
    flow.state.plan = None

    snapshot = flow._prior_plan_snapshot(_fake_prior(prior_plan))

    assert snapshot is not None
    assert [s.id for s in snapshot.subtasks] == [s.id for s in prior_plan.subtasks]


def test_prior_plan_snapshot_returns_none_when_unrecoverable() -> None:
    flow = CodebuilderFlow()
    flow.state.plan = None

    assert flow._prior_plan_snapshot(SimpleNamespace(feedback="x", output=None)) is None
    assert flow._prior_plan_snapshot(SimpleNamespace(feedback="x", output={"garbage": 1})) is None


def test_amend_guardrail_rejects_empty_and_accepts_valid() -> None:
    empty = Plan(project_name="x", mode="new_project", tech_stack=["py"], subtasks=[])
    ok = _valid_plan(1)

    assert _require_nonempty_plan(SimpleNamespace(pydantic=empty))[0] is False
    assert _require_nonempty_plan(SimpleNamespace(pydantic=None))[0] is False
    passed, payload = _require_nonempty_plan(SimpleNamespace(pydantic=ok))
    assert passed is True
    assert payload is ok  # returns the Plan unchanged on success


def test_default_crew_excludes_amend_task() -> None:
    pc = PlannerCrew()
    assert [t.name for t in pc.crew().tasks] == ["skeleton_task", "expand_task"]
    amend = pc.amend_crew()
    assert len(amend.tasks) == 1
    assert amend.tasks[0].guardrail is not None
