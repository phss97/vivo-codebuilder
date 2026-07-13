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


class Plan(StrictOutputModel):
    """A CC-native plan: the body is freeform Markdown, with questions and
    assumptions pulled out so the UI can list them separately from the plan
    itself. This is the structured envelope the planner agent emits and the
    frontend renders on the HITL approval card."""

    project_name: str
    mode: JobMode
    tech_stack: list[str] = Field(default_factory=list)
    # Natural language for all comments/docstrings/narrative (e.g. "Portuguese").
    # The planner detects this from the brief unless the caller supplied an
    # explicit override.
    language: str = ""
    # Optional domain slug (e.g. "rpa") the planner picks. Informational only —
    # the executor loads the matching CC skill; there is no separate gate.
    domain: str = ""
    # The plan itself, as Markdown. Rendered verbatim on the approval card and
    # fed to the executor.
    plan_markdown: str = ""
    # Genuinely blocking questions for the human (empty when none).
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


ArtifactKind = Literal["file", "project_archive"]


class ArtifactRef(StrictOutputModel):
    file_path: str
    size: int
    url: str
    kind: ArtifactKind = "file"


class ProjectArchiveRef(StrictOutputModel):
    kind: Literal["project_archive"] = "project_archive"
    file_path: str
    size: int
    local_path: str
    url: str = ""


class QAReport(StrictOutputModel):
    passed: bool
    lint_output: str = ""
    test_output: str = ""
    type_output: str = ""
    integration_notes: str = ""
    artifact_urls: list[ArtifactRef] = Field(default_factory=list)


class CodebuilderState(FlowState):
    # Caller-supplied session identifier. Decoupled from `id` (= flow_id) on
    # purpose: passing `id` in kickoff inputs would override the auto-generated
    # flow_id used by AMP's OTel traces (see CON-101 / COR-48 — AMP can't fetch
    # traces from Wharf when execution_id and flow_id disagree). The frontend
    # uses session_id for its URL slug, in-memory registry, and to correlate
    # incoming progress / HITL webhooks back to the right session.
    session_id: str = ""
    brief: str = ""
    project_name: str = ""
    project_key: str = ""
    # Resolved output language for the agents: an explicit kickoff `language`
    # override if supplied (auto-merged before `ingest`), else the language the
    # planner detected from the brief, resolved onto state right after `plan`.
    language: str = ""
    goals: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    attachment_records: list[dict[str, str]] = Field(default_factory=list)
    workspace_dir: str = ""
    plan: Plan | None = None
    amendments: str = ""
    amend_cycles: int = 0
    preflight_qa_report: QAReport | None = None
    qa_report: QAReport | None = None
    final_qa_repair_attempts: int = 0
    patch: str = ""
    zip_path: str = ""
    zip_url: str = ""
    project_archive: ProjectArchiveRef | None = None
    status: JobStatus = "pending"
