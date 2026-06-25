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


ChangeType = Literal["create", "modify"]


class FileSkeleton(StrictOutputModel):
    path: str
    purpose: str
    change_type: ChangeType = "create"
    # Exact public symbols this file exports for OTHER files to import, with
    # signatures. e.g. ["create_registro(data: dict) -> Registro",
    # "class RegistroRepository(Protocol)"]. These names are binding: every
    # file importing from this one must use them verbatim (no synonyms). Empty
    # for files with no importable API (README, .env.example, pyproject.toml,
    # build.spec, build scripts).
    public_api: list[str] = Field(default_factory=list)


class SubTask(StrictOutputModel):
    id: str
    title: str
    description: str
    # A work package can contain multiple files. The writer returns a
    # CodeBundleArtifact with one CodeArtifact per planned file.
    files: list[FileSkeleton]
    depends_on: list[str] = Field(default_factory=list)
    tech_notes: str = ""
    test_criteria: str = ""

    @property
    def file_paths(self) -> list[str]:
        return [f.path for f in self.files]


class PlanSkeleton(StrictOutputModel):
    project_name: str
    mode: JobMode
    domain: str = ""
    # Natural language the agents must write all comments, docstrings, and
    # narrative output in (e.g. "Portuguese", "English"). The planner detects
    # this from the brief unless the caller supplied an explicit override.
    language: str = ""
    tech_stack: list[str]
    files: list[FileSkeleton]
    # Top-level Python package names belonging to reference libraries the user
    # attached for the writer to consume (not reimplement). The import
    # completeness gate uses this whitelist to skip "missing module" errors
    # for these packages.
    external_packages: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class Plan(StrictOutputModel):
    project_name: str
    mode: JobMode
    tech_stack: list[str]
    subtasks: list[SubTask]
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    # Optional slug matching an activated architecture-defining skill
    # (e.g. "rpa"). When set, finalize dispatches the matching
    # architecture gate; when empty, no domain gate runs.
    domain: str = ""
    # Carried over from PlanSkeleton — the language the planner detected (or
    # the caller's override). The flow resolves this onto CodebuilderState so
    # every downstream crew writes in the same language.
    language: str = ""
    # Carried over from PlanSkeleton — see FileSkeleton/PlanSkeleton docstrings.
    external_packages: list[str] = Field(default_factory=list)


class CodeArtifact(StrictOutputModel):
    subtask_id: str
    file_path: str
    content: str = ""
    language: str
    tests_included: bool = False


class CodeBundleArtifact(StrictOutputModel):
    subtask_id: str
    artifacts: list[CodeArtifact]


class ReviewResult(StrictOutputModel):
    subtask_id: str
    passed: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


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
    # Filtered mypy output: only the high-confidence cross-file drift errors
    # (attr-defined / call-arg / name-defined / arg-type). Empty when the type
    # gate passed or was unavailable. Surfaced to the repair writer.
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
    # Resolved output language for all agents: an explicit kickoff `language`
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
    artifacts: list[CodeArtifact] = Field(default_factory=list)
    review_results: list[ReviewResult] = Field(default_factory=list)
    qa_report: QAReport | None = None
    final_qa_repair_attempts: int = 0
    patch: str = ""
    zip_path: str = ""
    zip_url: str = ""
    project_archive: ProjectArchiveRef | None = None
    status: JobStatus = "pending"
