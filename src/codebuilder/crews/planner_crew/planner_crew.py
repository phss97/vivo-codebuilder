from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectoryReadTool, FileReadTool

from codebuilder.schemas import Plan


KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"


def _load_knowledge(*names: str) -> list[StringKnowledgeSource]:
    sources = []
    for name in names:
        path = KNOWLEDGE_DIR / name
        if path.is_file():
            sources.append(StringKnowledgeSource(content=path.read_text(encoding="utf-8")))
    return sources


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
            knowledge_sources=_load_knowledge(
                "python_best_practices.md",
                "rpa_patterns.md",
            ),
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
            memory=True,
            embedder={"provider": "openai", "config": {"model_name": "text-embedding-3-small"}},
            verbose=True,
        )
