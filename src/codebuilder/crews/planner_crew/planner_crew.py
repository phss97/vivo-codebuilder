from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.skills import activate_skill, discover_skills
from crewai_tools import DirectoryReadTool, FileReadTool

from codebuilder.schemas import Plan, PlanSkeleton

_SKILLS = {
    s.name: activate_skill(s)
    for s in discover_skills(Path(__file__).resolve().parents[2] / "skills")
}


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
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
