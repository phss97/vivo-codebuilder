from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.skills import activate_skill, discover_skills

from codebuilder.schemas import QAReport, ReviewResult
from codebuilder.tools import (
    LintRunnerTool,
    TestRunnerTool,
    WorkspaceListTool,
    WorkspaceReadTool,
)

_SKILLS = {
    s.name: activate_skill(s)
    for s in discover_skills(Path(__file__).resolve().parents[2] / "skills")
}


@CrewBase
class ReviewerCrew:
    """Reviews one artifact and runs final QA over the whole workspace."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def _shared_tools(self):
        return [
            WorkspaceReadTool(workspace_dir=self.workspace_dir),
            WorkspaceListTool(workspace_dir=self.workspace_dir),
            LintRunnerTool(workspace_dir=self.workspace_dir),
            TestRunnerTool(workspace_dir=self.workspace_dir),
        ]

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer"],  # type: ignore[index]
            tools=self._shared_tools(),
            skills=[_SKILLS["rpa"], _SKILLS["code-review-gate"]],
        )

    @agent
    def qa_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["qa_agent"],  # type: ignore[index]
            tools=self._shared_tools(),
            skills=[_SKILLS["rpa"], _SKILLS["code-review-gate"]],
        )

    @task
    def review_task(self) -> Task:
        return Task(
            config=self.tasks_config["review_task"],  # type: ignore[index]
            output_pydantic=ReviewResult,
        )

    @task
    def qa_task(self) -> Task:
        return Task(
            config=self.tasks_config["qa_task"],  # type: ignore[index]
            output_pydantic=QAReport,
        )

    @task
    def architecture_gate_task(self) -> Task:
        return Task(
            config=self.tasks_config["architecture_gate_task"],  # type: ignore[index]
            output_pydantic=ReviewResult,
        )

    @crew
    def crew(self) -> Crew:
        """Fallback crew runs review_task only for ambiguous deterministic checks.

        Memory is intentionally disabled here because fallback reviews should
        be rare and per-file. Project-level learning lives on planner memory.
        """
        return Crew(
            agents=[self.reviewer()],
            tasks=[self.review_task()],
            process=Process.sequential,
            verbose=True,
        )

    def qa_crew(self) -> Crew:
        """Separate entry point used once at the end of the flow for integration QA."""
        return Crew(
            agents=[self.qa_agent()],
            tasks=[self.qa_task()],
            process=Process.sequential,
            verbose=True,
        )

    def architecture_gate_crew(self) -> Crew:
        """Final domain architecture acceptance gate (dispatched on plan.domain)."""
        return Crew(
            agents=[self.qa_agent()],
            tasks=[self.architecture_gate_task()],
            process=Process.sequential,
            verbose=True,
        )
