from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.project import CrewBase, agent, crew, task

from codebuilder.schemas import CodeArtifact
from codebuilder.tools import (
    WorkspaceListTool,
    WorkspaceReadTool,
    WorkspaceWriteTool,
)


KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"


def _load_knowledge(*names: str) -> list[StringKnowledgeSource]:
    sources = []
    for name in names:
        path = KNOWLEDGE_DIR / name
        if path.is_file():
            sources.append(StringKnowledgeSource(content=path.read_text(encoding="utf-8")))
    return sources


@CrewBase
class WriterCrew:
    """Writes one file per subtask. Verification happens downstream in the flow."""

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
        return Agent(
            config=self.agents_config["writer"],  # type: ignore[index]
            tools=self._workspace_tools(),
            knowledge_sources=_load_knowledge(
                "python_best_practices.md",
                "rpa_patterns.md",
            ),
        )

    @task
    def write_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_task"],  # type: ignore[index]
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
