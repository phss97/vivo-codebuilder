from typing import Literal

from crewai.flow.flow import FlowState
from pydantic import BaseModel, Field


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


class SubTask(BaseModel):
    id: str
    title: str
    description: str
    file_path: str
    depends_on: list[str] = Field(default_factory=list)
    tech_notes: str = ""
    test_criteria: str = ""


class Plan(BaseModel):
    project_name: str
    mode: JobMode
    tech_stack: list[str]
    subtasks: list[SubTask]
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class CodeArtifact(BaseModel):
    subtask_id: str
    file_path: str
    content: str
    language: str
    tests_included: bool = False


class ReviewResult(BaseModel):
    subtask_id: str
    passed: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class QAReport(BaseModel):
    passed: bool
    lint_output: str = ""
    test_output: str = ""
    integration_notes: str = ""


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
    patch: str = ""
    status: JobStatus = "pending"
