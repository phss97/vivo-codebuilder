from pathlib import Path
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.skills import activate_skill, discover_skills
from crewai_tools import DirectoryReadTool, FileReadTool

from codebuilder.schemas import Plan, PlanSkeleton

_SKILLS = {
    s.name: activate_skill(s)
    for s in discover_skills(Path(__file__).resolve().parents[2] / "skills")
}


def _require_nonempty_plan(output) -> tuple[bool, Any]:
    """Task guardrail: reject empty/invalid plans so the crew retries (up to
    ``guardrail_max_retries``) instead of emitting a Plan the build can't use.

    Deterministic — no extra LLM call, so it adds no latency. On success it
    returns the Plan object as-is; CrewAI leaves the TaskOutput (and its
    ``.pydantic``) untouched for non-str/non-TaskOutput results, so the caller
    still receives the validated ``Plan`` in ``result.pydantic``.
    """
    plan = getattr(output, "pydantic", None)
    if not isinstance(plan, Plan):
        return (False, "Output must be a valid Plan JSON object matching the Plan schema.")
    if not plan.subtasks:
        return (
            False,
            "Plan.subtasks must not be empty — return the COMPLETE plan with every work package.",
        )
    return (True, plan)


@CrewBase
class PlannerCrew:
    """Plans a project from a brief into bundled Writer-ready work packages."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def planner(self) -> Agent:
        cfg = self.agents_config["planner"]  # type: ignore[index]
        # Build the LLM in Python so we can set max_tokens; the YAML value
        # would otherwise be passed as a bare string and stuck at the
        # AnthropicCompletion default (4096), which truncates the Plan JSON
        # for non-trivial briefs and drops required fields like `subtasks`.
        return Agent(
            config=cfg,
            tools=[FileReadTool(), DirectoryReadTool()],
            skills=[_SKILLS["rpa"]],
            llm=LLM(model=cfg["llm"], max_tokens=32768),
        )

    @task
    def skeleton_task(self) -> Task:
        return Task(
            config=self.tasks_config["skeleton_task"],  # type: ignore[index]
            output_pydantic=PlanSkeleton,
        )

    @task
    def expand_task(self) -> Task:
        return Task(
            config=self.tasks_config["expand_task"],  # type: ignore[index]
            context=[self.skeleton_task()],
            output_pydantic=Plan,
            guardrail=_require_nonempty_plan,
        )

    # Plain method (NOT @task) so it is excluded from the auto-collected
    # `self.tasks` used by `.crew()`. Mirrors WriterCrew.repair_task /
    # repair_crew. Used by `.amend_crew()` to patch an existing plan from
    # user amendments without re-deriving the whole project.
    def amend_task(self) -> Task:
        return Task(
            config=self.tasks_config["amend_task"],  # type: ignore[index]
            output_pydantic=Plan,
            guardrail=_require_nonempty_plan,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def amend_crew(self) -> Crew:
        """Single-task crew for plan revisions: edits only the work packages the
        amendments touch, instead of the full skeleton→expand re-plan. Faster and
        far less likely to blow the work-package cap than re-planning the brief."""
        return Crew(
            agents=[self.planner()],
            tasks=[self.amend_task()],
            process=Process.sequential,
            verbose=True,
        )
