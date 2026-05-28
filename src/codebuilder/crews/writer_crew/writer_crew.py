from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.skills import activate_skill, discover_skills

from codebuilder.schemas import CodeArtifact, CodeBundleArtifact
from codebuilder.tools import (
    WorkspaceListTool,
    WorkspaceReadTool,
    WorkspaceWriteTool,
)

_SKILLS = {
    s.name: activate_skill(s)
    for s in discover_skills(Path(__file__).resolve().parents[2] / "skills")
}


@CrewBase
class WriterCrew:
    """Writes one bundle of planned files per subtask."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def _workspace_tools(self):
        return [
            WorkspaceReadTool(workspace_dir=self.workspace_dir),
            WorkspaceWriteTool(workspace_dir=self.workspace_dir),
            WorkspaceListTool(workspace_dir=self.workspace_dir),
        ]

    @agent
    def writer(self) -> Agent:
        cfg = self.agents_config["writer"]  # type: ignore[index]
        # max_tokens=32768 so CodeBundleArtifact content (full file bodies) isn't
        # truncated against AnthropicCompletion's 4096 default.
        return Agent(
            config=cfg,
            tools=self._workspace_tools(),
            skills=[_SKILLS["rpa"]],
            llm=LLM(model=cfg["llm"], max_tokens=32768),
        )

    @task
    def write_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_task"],  # type: ignore[index]
            output_pydantic=CodeBundleArtifact,
        )

    def repair_task(self) -> Task:
        return Task(
            config=self.tasks_config["repair_task"],  # type: ignore[index]
            output_pydantic=CodeArtifact,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def repair_crew(self) -> Crew:
        return Crew(
            agents=[self.writer()],
            tasks=[self.repair_task()],
            process=Process.sequential,
            verbose=True,
        )
