from typing import Any, Literal

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
FileKind = Literal["source", "test", "config", "docs"]
IssueOwner = Literal["code", "test", "spec", "environment"]


class AuthoritativeAsset(StrictOutputModel):
    """A caller-owned input whose exact name/content must survive generation."""

    path: str
    immutable: bool = True
    sha256: str = ""


class IntakeQuestion(StrictOutputModel):
    id: str
    question: str
    rationale: str = ""


class IntakeAssessment(StrictOutputModel):
    ready: bool
    understood_scope: str = ""
    evidence_inspected: list[str] = Field(default_factory=list)
    authoritative_assets: list[AuthoritativeAsset] = Field(default_factory=list)
    blocking_questions: list[IntakeQuestion] = Field(default_factory=list)
    detected_stack: list[str] = Field(default_factory=list)
    missing_verification_commands: list[str] = Field(default_factory=list)


class IdentifierContract(StrictOutputModel):
    """Machine identifiers copied verbatim by every downstream agent."""

    packages: list[str] = Field(default_factory=list)
    modules: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    environment_variables: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)


class TerminologyEntry(StrictOutputModel):
    """Translations for human-facing prose only, never code identifiers."""

    canonical: str
    translations: dict[str, str] = Field(default_factory=dict)


class VerificationCommand(StrictOutputModel):
    id: str
    category: Literal["lint", "typecheck", "test", "build", "integration"]
    argv: list[str]
    cwd: str = "."
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    required: bool = True
    network: bool = False


class CommandResult(StrictOutputModel):
    command_id: str
    passed: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    mutated_paths: list[str] = Field(default_factory=list)


class SuccessCriterion(StrictOutputModel):
    id: str
    description: str


class TestCaseSpec(StrictOutputModel):
    id: str
    criterion_ids: list[str]
    path: str
    test_name: str
    expected_behavior: str


class FileSpec(StrictOutputModel):
    path: str
    purpose: str
    change_type: ChangeType = "create"
    kind: FileKind = "source"
    # These declarations are binding. A generated synonym is a contract failure.
    public_api: list[str] = Field(default_factory=list)


class WorkPackageSpec(StrictOutputModel):
    id: str
    title: str
    what_to_build: str
    expected_behavior: str
    success_criteria: list[SuccessCriterion]
    tests: list[TestCaseSpec]
    files: list[FileSpec]
    depends_on: list[str] = Field(default_factory=list)

    @property
    def file_paths(self) -> list[str]:
        return [file.path for file in self.files]


class Plan(StrictOutputModel):
    """Revisioned specification with a legacy Markdown compatibility view."""

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
    revision: int = Field(default=1, ge=1)
    package_name: str = ""
    authoritative_assets: list[AuthoritativeAsset] = Field(default_factory=list)
    identifier_contract: IdentifierContract = Field(default_factory=IdentifierContract)
    terminology: list[TerminologyEntry] = Field(default_factory=list)
    verification_commands: list[VerificationCommand] = Field(default_factory=list)
    work_packages: list[WorkPackageSpec] = Field(default_factory=list)

    @property
    def is_structured(self) -> bool:
        return bool(
            self.package_name
            or self.work_packages
            or self.authoritative_assets
            or self.terminology
            or self.verification_commands
            or any(self.identifier_contract.model_dump().values())
        )

    def render_markdown(self) -> str:
        """Render the canonical structured fields for the existing approval UI."""
        if not self.is_structured:
            return self.plan_markdown
        lines = [
            f"# {self.project_name}",
            "",
            f"Specification revision: {self.revision}",
            f"Mode: `{self.mode}`",
            f"Package: `{self.package_name}`",
        ]
        identifiers = self.identifier_contract.model_dump()
        if any(identifiers.values()):
            lines.extend(["", "## Identifier contract"])
            for kind, values in identifiers.items():
                if values:
                    lines.append(
                        f"- {kind}: " + ", ".join(f"`{value}`" for value in values)
                    )
        if self.terminology:
            lines.extend(["", "## Terminology registry"])
            for entry in self.terminology:
                translations = ", ".join(
                    f"{language}: {value}"
                    for language, value in sorted(entry.translations.items())
                )
                lines.append(
                    f"- `{entry.canonical}`"
                    + (f" — {translations}" if translations else "")
                )
        if self.authoritative_assets:
            lines.extend(["", "## Authoritative assets"])
            for asset in self.authoritative_assets:
                lines.append(
                    f"- `{asset.path}` — immutable={str(asset.immutable).lower()}, "
                    f"sha256=`{asset.sha256 or '(unbound)'}`"
                )
        if self.assumptions:
            lines.extend(
                ["", "## Assumptions", *[f"- {item}" for item in self.assumptions]]
            )
        for package in self.work_packages:
            lines.extend(
                [
                    "",
                    f"## {package.id}: {package.title}",
                    "",
                    package.what_to_build,
                    "",
                    f"Expected behavior: {package.expected_behavior}",
                    f"Depends on: {', '.join(f'`{item}`' for item in package.depends_on) or '(none)'}",
                    "",
                    "### Success criteria",
                    *[
                        f"- `{criterion.id}`: {criterion.description}"
                        for criterion in package.success_criteria
                    ],
                    "",
                    "### Tests",
                    *[
                        f"- `{test.id}` ({', '.join(test.criterion_ids)}): "
                        f"`{test.path}::{test.test_name}` — {test.expected_behavior}"
                        for test in package.tests
                    ],
                    "",
                    "### Files",
                    *[
                        f"- `{file.path}` ({file.change_type}, {file.kind})"
                        + (
                            f" — exports: {', '.join(file.public_api)}"
                            if file.public_api
                            else ""
                        )
                        for file in package.files
                    ],
                ]
            )
        if self.verification_commands:
            lines.extend(
                [
                    "",
                    "## Verification",
                    *[
                        f"- `{command.id}` [{command.category}, cwd=`{command.cwd}`, "
                        f"required={str(command.required).lower()}, "
                        f"timeout={command.timeout_seconds}s, "
                        f"network={str(command.network).lower()}]: "
                        f"`{' '.join(command.argv)}`"
                        for command in self.verification_commands
                    ],
                ]
            )
        return "\n".join(lines).strip() + "\n"


ArtifactKind = Literal["file", "project_archive", "quarantine_archive"]


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


class QuarantineArchiveRef(StrictOutputModel):
    kind: Literal["quarantine_archive"] = "quarantine_archive"
    file_path: str
    size: int
    local_path: str
    url: str = ""


class QAIssue(StrictOutputModel):
    source: Literal["command", "contract", "review"]
    owner: IssueOwner
    message: str
    criterion_ids: list[str] = Field(default_factory=list)
    evidence: str = ""
    repair_instruction: str = ""
    blocking: bool = True


class QAReport(StrictOutputModel):
    passed: bool
    lint_output: str = ""
    test_output: str = ""
    type_output: str = ""
    integration_notes: str = ""
    artifact_urls: list[ArtifactRef] = Field(default_factory=list)
    spec_hash: str = ""
    package_id: str = ""
    command_results: list[CommandResult] = Field(default_factory=list)
    issues: list[QAIssue] = Field(default_factory=list)
    contract_issues: list[QAIssue] = Field(default_factory=list)
    review_issues: list[QAIssue] = Field(default_factory=list)
    repair_count: int = 0


class PackageResult(StrictOutputModel):
    package_id: str
    spec_hash: str
    status: Literal["pending", "passed", "failed", "skipped"]
    command_results: list[CommandResult] = Field(default_factory=list)
    issues: list[QAIssue] = Field(default_factory=list)
    repair_count: int = 0


class QuarantineReport(StrictOutputModel):
    spec_hash: str
    last_green_package_ids: list[str] = Field(default_factory=list)
    failed_package_id: str = ""
    skipped_package_ids: list[str] = Field(default_factory=list)
    issues: list[QAIssue] = Field(default_factory=list)
    package_results: list[PackageResult] = Field(default_factory=list)


class ProductionReview(StrictOutputModel):
    """Blocker-only semantic review of an RPA package's production wiring."""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    qa_issues: list[QAIssue] = Field(default_factory=list)


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
    canonical_build_dir: str = ""
    current_stage_dir: str = ""
    phase: str = "intake"
    intake_assessment: IntakeAssessment | None = None
    intake_answers: list[str] = Field(default_factory=list)
    plan: Plan | None = None
    approved_spec_hash: str = ""
    amendments: str = ""
    amend_cycles: int = 0
    preflight_qa_report: QAReport | None = None
    baseline_dependencies: list[str] = Field(default_factory=list)
    qa_report: QAReport | None = None
    production_review: ProductionReview | None = None
    final_qa_repair_attempts: int = 0
    package_cursor: int = 0
    current_package_id: str = ""
    package_results: list[PackageResult] = Field(default_factory=list)
    package_repair_attempts: dict[str, int] = Field(default_factory=dict)
    skipped_package_ids: list[str] = Field(default_factory=list)
    frozen_test_hashes: dict[str, str] = Field(default_factory=dict)
    current_failure: QAReport | None = None
    llm_usage: list[dict[str, Any]] = Field(default_factory=list)
    patch: str = ""
    zip_path: str = ""
    zip_url: str = ""
    project_archive: ProjectArchiveRef | None = None
    quarantine_report: QuarantineReport | None = None
    quarantine_archive: QuarantineArchiveRef | None = None
    status: JobStatus = "pending"
