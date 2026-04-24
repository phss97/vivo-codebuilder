from typing import Literal

from crewai.flow.flow import FlowState
from pydantic import BaseModel, ConfigDict, Field


JobMode = Literal["new_project", "patch_existing"]
JobStatus = Literal[
    "pending",
    "planning",
    "awaiting_approval",
    "executing",
    "done",
    "failed",
]


class Attachment(BaseModel):
    kind: Literal["git", "pdf", "image", "zip"]
    name: str
    content_b64: str = ""
    uri: str = ""


class StrictOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubTask(StrictOutputModel):
    id: str
    title: str
    description: str
    file_path: str
    depends_on: list[str] = Field(default_factory=list)
    tech_notes: str = ""
    test_criteria: str = ""


class Plan(StrictOutputModel):
    project_name: str
    mode: JobMode
    tech_stack: list[str]
    subtasks: list[SubTask]
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class CodeArtifact(StrictOutputModel):
    subtask_id: str
    file_path: str
    content: str = ""
    language: str
    tests_included: bool = False


class ReviewResult(StrictOutputModel):
    subtask_id: str
    passed: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class ArtifactRef(StrictOutputModel):
    file_path: str
    size: int
    url: str


class QAReport(StrictOutputModel):
    passed: bool
    lint_output: str = ""
    test_output: str = ""
    integration_notes: str = ""
    artifact_urls: list[ArtifactRef] = Field(default_factory=list)


class CodebuilderState(FlowState):
    brief: str = ""
    project_name: str = ""
    project_key: str = ""
    goals: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    workspace_dir: str = ""
    plan: Plan | None = None
    amendments: str = ""
    amend_cycles: int = 0
    artifacts: list[CodeArtifact] = Field(default_factory=list)
    review_results: list[ReviewResult] = Field(default_factory=list)
    qa_report: QAReport | None = None
    final_qa_repair_attempts: int = 0
    patch: str = ""
    zip_path: str = ""
    zip_url: str = ""
    status: JobStatus = "pending"
