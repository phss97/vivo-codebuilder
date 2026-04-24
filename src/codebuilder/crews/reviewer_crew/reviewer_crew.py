from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.project import CrewBase, agent, crew, task

from codebuilder.schemas import QAReport, ReviewResult
from codebuilder.tools import (
    LintRunnerTool,
    TestRunnerTool,
    WorkspaceListTool,
    WorkspaceReadTool,
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
            knowledge_sources=_load_knowledge("review_checklist.md"),
        )

    @agent
    def qa_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["qa_agent"],  # type: ignore[index]
            tools=self._shared_tools(),
            knowledge_sources=_load_knowledge("review_checklist.md"),
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
            memory=True,
            embedder={"provider": "openai", "config": {"model_name": "text-embedding-3-small"}},
            verbose=True,
        )
