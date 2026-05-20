from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.skills import activate_skill, discover_skills
from crewai_tools import DirectoryReadTool, FileReadTool

from codebuilder.schemas import Plan

_SKILLS = {
    s.name: activate_skill(s)
    for s in discover_skills(Path(__file__).resolve().parents[2] / "skills")
}


@CrewBase
class PlannerCrew:
    """Plans a project from a brief into atomic Writer-ready subtasks."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config["planner"],  # type: ignore[index]
            tools=[FileReadTool(), DirectoryReadTool()],
            skills=[_SKILLS["rpa"]],
        )

    @task
    def plan_task(self) -> Task:
        return Task(
            config=self.tasks_config["plan_task"],  # type: ignore[index]
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
