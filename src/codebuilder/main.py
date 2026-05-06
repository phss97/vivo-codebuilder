"""Codebuilder Flow — plans, gates on human approval, then builds."""

from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from crewai.flow import Flow, listen, persist, start
from crewai.flow.human_feedback import human_feedback

from codebuilder import history
from codebuilder.crews.planner_crew import PlannerCrew
from codebuilder.crews.reviewer_crew import ReviewerCrew
from codebuilder.crews.writer_crew import WriterCrew
from codebuilder.schemas import (
    Attachment,
    ArtifactRef,
    CodeArtifact,
    CodebuilderState,
    Plan,
    QAReport,
    ReviewResult,
    SubTask,
)
from codebuilder.tools import LintRunnerTool, TestRunnerTool, attachment_tool, git_tool
from codebuilder.tools.s3_artifacts import SKIP_DIRS, SKIP_FILES, upload_file, upload_workspace
from codebuilder.tools.workspace_tool import WorkspaceListTool, resolve_within


log = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("CODEBUILDER_WORKSPACE_ROOT", "./workspaces")).resolve()
DEFAULT_MAX_SUBTASK_RETRIES = 1
DEFAULT_MAX_FINAL_QA_REPAIRS = 1
MAX_QA_OUTPUT_CHARS = 12000
PROGRESS_WEBHOOK_TIMEOUT_SECONDS = 5

GUARDRAIL_LLM = os.environ.get("CODEBUILDER_GUARDRAIL_LLM", "openai/gpt-5.4-mini")


_ZIP_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_TODO_TOKEN_RE = re.compile(r"\b(todo|fixme|placeholder|stub)\b", re.IGNORECASE)
_PLACEHOLDER_LINE_RE = re.compile(
    r"(pass|\.\.\.|raise\s+NotImplementedError(?:\([^)]*\))?)",
    re.IGNORECASE,
)


def _safe_zip_stem(name: str) -> str:
    stem = _ZIP_NAME_RE.sub("-", name).strip("-.") or "project"
    return stem[:80]


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default
    return max(minimum, value)


def _max_subtask_retries() -> int:
    return _env_int("CODEBUILDER_MAX_SUBTASK_RETRIES", DEFAULT_MAX_SUBTASK_RETRIES)


def _max_final_qa_repairs() -> int:
    return _env_int("CODEBUILDER_MAX_FINAL_QA_REPAIRS", DEFAULT_MAX_FINAL_QA_REPAIRS)


def _truncate(value: str, limit: int = MAX_QA_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n\n[truncated {omitted} chars]"


def _append_note(report: QAReport, note: str) -> None:
    report.integration_notes = " ".join(part for part in (report.integration_notes, note) if part)


def _emit_progress(state: CodebuilderState, event_type: str, **payload: Any) -> None:
    """Best-effort progress callback for UIs that need finer updates than AMP provides."""
    webhook = os.environ.get("CODEBUILDER_PROGRESS_WEBHOOK")
    if not webhook:
        return

    body = {
        "event_type": event_type,
        "job_id": state.id,
        "project_name": state.project_name,
        "project_key": state.project_key,
        **payload,
    }
    headers = {"Content-Type": "application/json"}
    secret = os.environ.get("CODEBUILDER_PROGRESS_WEBHOOK_SECRET")
    if secret:
        headers["X-Codebuilder-Progress-Secret"] = secret

    try:
        resp = requests.post(webhook, json=body, headers=headers, timeout=PROGRESS_WEBHOOK_TIMEOUT_SECONDS)
        if resp.status_code >= 400:
            log.warning("progress webhook POST for %s returned %s", event_type, resp.status_code)
    except requests.RequestException as exc:
        log.warning("progress webhook POST failed for %s: %s", event_type, exc)


def _zip_build(build_dir: str, out_dir: Path, project_name: str) -> Path:
    """Zip the built project into ``out_dir/<project>.zip``. Overwrites if present."""
    src = Path(build_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_zip_stem(project_name)}.zip"
    if out_path.exists():
        out_path.unlink()

    arcroot = out_path.stem  # wrap contents under a top-level folder in the archive
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            if path.name in SKIP_FILES:
                continue
            rel = path.relative_to(src)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            zf.write(path, arcname=f"{arcroot}/{rel.as_posix()}")
    return out_path


def _planner_inputs(state: CodebuilderState) -> dict:
    listing_tool = WorkspaceListTool(workspace_dir=state.workspace_dir) if state.workspace_dir else None
    listing = listing_tool._run(".") if listing_tool else ""
    prior_history = history.summarize_for_planner(state.project_key) if state.project_key else ""
    return {
        "brief": state.brief,
        "project_name": state.project_name or "(unspecified)",
        "goals": "\n".join(f"- {g}" for g in state.goals) or "(none)",
        "tech_stack": ", ".join(state.tech_stack) or "(unspecified)",
        "attachment_records": listing or "(no attachments)",
        "workspace_dir": state.workspace_dir,
        "prior_plan": state.plan.model_dump_json(indent=2) if state.plan else "",
        "prior_history": prior_history or "(no prior runs for this project)",
        "amendments": state.amendments,
    }


@dataclass(frozen=True)
class DeterministicReview:
    result: ReviewResult
    needs_fallback: bool = False


def _is_pass(output: str) -> bool:
    normalized = output.strip()
    return normalized == "PASS" or normalized.startswith("PASS\n")


def _is_skip(output: str) -> bool:
    return output.strip().startswith("SKIP:")


def _is_test_file(path: str) -> bool:
    p = Path(path)
    return "tests" in p.parts or p.name.startswith("test_") or p.name.endswith("_test.py")


def _looks_like_placeholder(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    if _TODO_TOKEN_RE.search(stripped):
        return True

    meaningful_lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not meaningful_lines:
        return True
    return all(_PLACEHOLDER_LINE_RE.fullmatch(line) for line in meaningful_lines)


def _artifact_refs(refs: list[dict] | list[ArtifactRef] | None) -> list[ArtifactRef]:
    converted: list[ArtifactRef] = []
    for ref in refs or []:
        converted.append(ref if isinstance(ref, ArtifactRef) else ArtifactRef(**ref))
    return converted


def _persist_artifact(artifact: CodeArtifact, build_dir: str) -> str:
    """Ensure ``artifact.content`` lives on disk under ``build_dir``.

    Returns "" on success, or an error message describing why the file
    could not be persisted. The orchestrator owns the write so we don't
    depend on the writer LLM remembering to call ``workspace_write``.
    If the writer DID already persist the file (artifact.content empty
    but file present), we backfill ``artifact.content`` from disk so
    downstream consumers see the same bytes.
    """
    try:
        target = resolve_within(build_dir, artifact.file_path)
    except ValueError as exc:
        return str(exc)

    if artifact.content:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact.content, encoding="utf-8")
        return ""

    if target.is_file():
        artifact.content = target.read_text(encoding="utf-8", errors="replace")
        return ""

    return (
        f"Writer returned empty content and did not write {artifact.file_path}; "
        "include the full file text in CodeArtifact.content."
    )


def _validate_plan(plan: Plan | None) -> Plan:
    if not isinstance(plan, Plan):
        raise ValueError("Planner did not return a valid Plan object.")

    issues: list[str] = []
    if not 1 <= len(plan.subtasks) <= 15:
        issues.append("plan must contain between 1 and 15 subtasks")
    for subtask in plan.subtasks:
        if not subtask.file_path.strip():
            issues.append(f"subtask {subtask.id} has an empty file_path")
        if not subtask.test_criteria.strip():
            issues.append(f"subtask {subtask.id} has empty test_criteria")
    if issues:
        raise ValueError("Invalid plan: " + "; ".join(issues))
    return plan


def run_deterministic_review(
    subtask: SubTask,
    artifact: CodeArtifact,
    build_dir: str,
    *,
    lint_runner: Any | None = None,
    test_runner: Any | None = None,
) -> DeterministicReview:
    """Review an artifact with local checks before falling back to an LLM."""
    issues: list[str] = []
    suggestions: list[str] = []
    fallback_reasons: list[str] = []

    if artifact.subtask_id != subtask.id:
        issues.append(
            f"Artifact subtask_id '{artifact.subtask_id}' does not match planned subtask '{subtask.id}'."
        )
    if artifact.file_path != subtask.file_path:
        issues.append(
            f"Artifact file_path '{artifact.file_path}' does not match planned path '{subtask.file_path}'."
        )

    try:
        target = resolve_within(build_dir, artifact.file_path)
    except ValueError as exc:
        issues.append(str(exc))
        return DeterministicReview(
            ReviewResult(subtask_id=subtask.id, passed=False, issues=issues)
        )

    actual = ""
    if not target.is_file():
        issues.append(f"Artifact file was not written to workspace: {artifact.file_path}")
    else:
        actual = target.read_text(encoding="utf-8", errors="replace")
        if artifact.content and actual != artifact.content:
            issues.append(
                f"Workspace file '{artifact.file_path}' does not match CodeArtifact.content."
            )

    content_to_check = actual or artifact.content
    if _looks_like_placeholder(content_to_check):
        issues.append("Artifact content is empty or contains placeholder/TODO-only output.")

    if not issues:
        lint_tool = lint_runner or LintRunnerTool(workspace_dir=build_dir)
        lint_output = lint_tool._run(artifact.file_path)
        if _is_skip(lint_output):
            fallback_reasons.append(f"Lint skipped for {artifact.file_path}: {lint_output}")
        elif not _is_pass(lint_output):
            issues.append(f"ruff failed for {artifact.file_path}:\n{lint_output}")

    if not issues and _is_test_file(artifact.file_path):
        test_tool = test_runner or TestRunnerTool(workspace_dir=build_dir)
        test_output = test_tool._run(artifact.file_path)
        if _is_skip(test_output):
            fallback_reasons.append(f"Pytest skipped for {artifact.file_path}: {test_output}")
        elif not _is_pass(test_output):
            issues.append(f"pytest failed for {artifact.file_path}:\n{test_output}")

    if issues:
        return DeterministicReview(
            ReviewResult(subtask_id=subtask.id, passed=False, issues=issues, suggestions=suggestions)
        )

    if fallback_reasons:
        return DeterministicReview(
            ReviewResult(
                subtask_id=subtask.id,
                passed=False,
                issues=fallback_reasons,
                suggestions=["Use the mini reviewer fallback to inspect the file manually."],
            ),
            needs_fallback=True,
        )

    return DeterministicReview(
        ReviewResult(
            subtask_id=subtask.id,
            passed=True,
            suggestions=["Deterministic path, content, lint, and test checks passed."],
        )
    )


def run_final_qa(
    build_dir: str,
    *,
    artifact_urls: list[dict] | list[ArtifactRef] | None = None,
    lint_runner: Any | None = None,
    test_runner: Any | None = None,
) -> QAReport:
    """Build the final QA report from deterministic workspace lint and tests."""
    lint_tool = lint_runner or LintRunnerTool(workspace_dir=build_dir)
    test_tool = test_runner or TestRunnerTool(workspace_dir=build_dir)

    lint_output = lint_tool._run(".")
    test_output = test_tool._run(".")

    lint_ok = _is_pass(lint_output) or _is_skip(lint_output)
    test_ok = _is_pass(test_output) or _is_skip(test_output)

    notes = ["Deterministic QA ran ruff check and pytest over the whole workspace."]
    if _is_skip(lint_output):
        notes.append(f"Lint diagnostic: {lint_output}")
    if _is_skip(test_output):
        notes.append(f"Test diagnostic: {test_output}")
    if not lint_ok:
        notes.append("Lint failed.")
    if not test_ok:
        notes.append("Tests failed.")

    return QAReport(
        passed=lint_ok and test_ok,
        lint_output=lint_output,
        test_output=test_output,
        integration_notes=" ".join(notes),
        artifact_urls=_artifact_refs(artifact_urls),
    )


@persist()
class CodebuilderFlow(Flow[CodebuilderState]):
    """Single flow: ingest → plan (HITL) → build → finalize."""

    @start()
    def ingest(self):
        # CrewAI Flow auto-merges `inputs={...}` keys into self.state before
        # this method runs. Each declared CodebuilderState field (id, brief,
        # project_name, goals, tech_stack, attachments) is already populated
        # by the time we get here. Only normalize attachments — they may
        # arrive as list[dict] (AMP / JSON inputs) instead of list[Attachment].
        self.state.attachments = [
            a if isinstance(a, Attachment) else Attachment(**a)
            for a in self.state.attachments
        ]

        workspace_dir = WORKSPACE_ROOT / self.state.id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "inputs").mkdir(exist_ok=True)
        (workspace_dir / "output").mkdir(exist_ok=True)
        self.state.workspace_dir = str(workspace_dir)

        if self.state.attachments:
            attachment_tool.materialize(
                [a.model_dump() for a in self.state.attachments],
                self.state.workspace_dir,
            )

        project_key = history.project_key_from(self.state)
        if not project_key:
            log.warning(
                "job %s has no project_name and no git attachment; "
                "falling back to flow_id for history keying",
                self.state.id,
            )
            project_key = self.state.id
        self.state.project_key = project_key

        # NOTE: do not mutate CREWAI_STORAGE_DIR here. crewai treats that env
        # var as the `app_name` input to `appdirs.user_data_dir(...)`, which
        # determines where `SQLiteFlowPersistence()` (with no db_path) stores
        # pending-feedback rows. `Flow.from_pending(flow_id)` always constructs
        # a default SQLiteFlowPersistence, so any drift in the env var between
        # save-time and resume-time breaks HITL resume with "No pending
        # feedback found for flow_id". Per-project memory scoping — if we want
        # it back — needs to go through Crew-level storage config, not this
        # env var.

        self.state.status = "planning"
        log.info(
            "job %s ingested; workspace=%s project_key=%s",
            self.state.id,
            self.state.workspace_dir,
            self.state.project_key,
        )

    @listen(ingest)
    @human_feedback(
        message="Review the generated plan. Reply 'approve' to start coding, describe changes to amend, or 'reject' to cancel.",
        emit=["approved", "amend", "rejected"],
        llm=GUARDRAIL_LLM,
        default_outcome="amend",
    )
    def plan(self) -> dict:
        result = PlannerCrew().crew().kickoff(inputs=_planner_inputs(self.state))
        plan_obj = _validate_plan(result.pydantic)
        self.state.plan = plan_obj
        self.state.status = "awaiting_approval"
        return plan_obj.model_dump()

    @listen("amend")
    @human_feedback(
        message="Revised plan — please review again. Approve, amend further, or reject.",
        emit=["approved", "amend", "rejected"],
        llm=GUARDRAIL_LLM,
        default_outcome="amend",
    )
    def revise_plan(self, prior) -> dict:
        self.state.amendments = getattr(prior, "feedback", "") or ""
        self.state.amend_cycles += 1
        result = PlannerCrew().crew().kickoff(inputs=_planner_inputs(self.state))
        plan_obj = _validate_plan(result.pydantic)
        self.state.plan = plan_obj
        self.state.status = "awaiting_approval"
        return plan_obj.model_dump()

    @listen("rejected")
    def on_rejected(self, prior):
        log.info("job %s rejected by human", self.state.id)
        self.state.status = "failed"
        try:
            history.record(self.state)
        except Exception as exc:  # noqa: BLE001 — history is observability, never fatal
            log.warning("history.record on rejection failed: %s", exc)
        return {"status": "failed", "reason": getattr(prior, "feedback", "")}

    @listen("approved")
    def build(self, prior):
        self.state.amendments = getattr(prior, "feedback", "") or self.state.amendments
        self.state.status = "executing"
        plan = self.state.plan
        if plan is None:
            self.state.status = "failed"
            return {"status": "failed", "reason": "no plan to execute"}

        if plan.mode == "patch_existing":
            repo_root = self._existing_repo_root()
            build_dir = repo_root or self.state.workspace_dir
        else:
            build_dir = str(Path(self.state.workspace_dir) / "output")
            Path(build_dir).mkdir(parents=True, exist_ok=True)
            git_tool.init_and_commit(build_dir)

        total_subtasks = len(plan.subtasks)
        for index, subtask in enumerate(plan.subtasks, start=1):
            self._build_subtask(subtask, build_dir, index=index, total=total_subtasks)

        self._build_dir = build_dir

    @listen(build)
    def finalize(self, _prior=None):
        build_dir = getattr(self, "_build_dir", self.state.workspace_dir)
        if self.state.status == "failed":
            try:
                history.record(self.state)
            except Exception as exc:  # noqa: BLE001 — history is observability, never fatal
                log.warning("history.record on build failure failed: %s", exc)
            return self._completion_payload(build_dir)

        _emit_progress(self.state, "final_qa_started")
        self.state.qa_report = run_final_qa(build_dir)
        self._repair_final_qa_if_needed(build_dir)
        _emit_progress(
            self.state,
            "final_qa_completed",
            passed=bool(self.state.qa_report and self.state.qa_report.passed),
            repair_attempts=self.state.final_qa_repair_attempts,
            integration_notes=self.state.qa_report.integration_notes if self.state.qa_report else "",
        )

        if self.state.plan and self.state.plan.mode == "patch_existing":
            try:
                self.state.patch = git_tool.diff(build_dir)
            except Exception as exc:
                log.warning("patch generation failed: %s", exc)
                self.state.patch = ""

        if self.state.plan and self.state.plan.mode == "new_project":
            try:
                zip_path = _zip_build(
                    build_dir,
                    Path(self.state.workspace_dir),
                    self.state.project_name or self.state.id,
                )
                self.state.zip_path = str(zip_path)
                log.info("job %s zipped to %s", self.state.id, zip_path)
            except Exception as exc:  # noqa: BLE001 — zip is convenience, not correctness
                log.warning("zip generation failed: %s", exc)

        try:
            prefix = f"{self.state.project_key or self.state.id}/{self.state.id}"
            self.state.qa_report.artifact_urls = _artifact_refs(upload_workspace(build_dir, prefix=prefix))
            if self.state.zip_path:
                zip_ref = upload_file(
                    self.state.zip_path,
                    key=f"{prefix}/{Path(self.state.zip_path).name}",
                )
                if zip_ref:
                    zip_artifact = ArtifactRef(**zip_ref)
                    self.state.zip_url = zip_artifact.url
                    self.state.qa_report.artifact_urls.append(zip_artifact)
        except Exception as exc:  # noqa: BLE001
            log.warning("artifact upload failed: %s", exc)

        self.state.status = "done"
        log.info("job %s complete", self.state.id)

        try:
            history.record(self.state)
        except Exception as exc:  # noqa: BLE001 — history is observability, never fatal
            log.warning("history.record on finalize failed: %s", exc)

        return self._completion_payload(build_dir)

    # --- helpers ---------------------------------------------------------

    def _existing_repo_root(self) -> str | None:
        inputs_dir = Path(self.state.workspace_dir) / "inputs"
        repo_dir = inputs_dir / "repo"
        if repo_dir.is_dir():
            return str(repo_dir)
        for child in inputs_dir.glob("repo*"):
            if child.is_dir():
                return str(child)
        return None

    def _plan_summary(self) -> str:
        if not self.state.plan:
            return "(no plan available)"
        subtasks = [
            {
                "id": subtask.id,
                "title": subtask.title,
                "file_path": subtask.file_path,
                "test_criteria": subtask.test_criteria,
            }
            for subtask in self.state.plan.subtasks
        ]
        return json.dumps(
            {
                "project_name": self.state.plan.project_name,
                "mode": self.state.plan.mode,
                "tech_stack": self.state.plan.tech_stack,
                "subtasks": subtasks,
            },
            indent=2,
        )

    def _qa_report_for_repair(self, report: QAReport) -> str:
        payload = report.model_dump()
        payload["lint_output"] = _truncate(payload.get("lint_output") or "")
        payload["test_output"] = _truncate(payload.get("test_output") or "")
        return json.dumps(payload, indent=2)

    def _validate_repair_artifact(self, artifact: CodeArtifact, build_dir: str) -> bool:
        persist_error = _persist_artifact(artifact, build_dir)
        if persist_error:
            log.warning("final QA repair could not be persisted: %s", persist_error)
            return False
        if _looks_like_placeholder(artifact.content):
            log.warning("final QA repair produced placeholder content: %s", artifact.file_path)
            return False
        return True

    def _repair_final_qa_once(self, build_dir: str, report: QAReport) -> CodeArtifact | None:
        writer = WriterCrew(workspace_dir=build_dir)
        listing_tool = WorkspaceListTool(workspace_dir=build_dir)
        result = writer.repair_crew().kickoff(
            inputs={
                "workspace_dir": build_dir,
                "workspace_listing": listing_tool._run("."),
                "plan_summary": self._plan_summary(),
                "qa_report": self._qa_report_for_repair(report),
            }
        )
        artifact = result.pydantic if isinstance(result.pydantic, CodeArtifact) else None
        if artifact is None:
            log.warning("final QA repair writer did not return a CodeArtifact")
            return None
        if artifact.subtask_id != "final_qa_repair":
            artifact.subtask_id = "final_qa_repair"
        if not self._validate_repair_artifact(artifact, build_dir):
            return None
        return artifact

    def _repair_final_qa_if_needed(self, build_dir: str) -> None:
        attempts = _max_final_qa_repairs()
        if attempts <= 0:
            return

        for attempt in range(1, attempts + 1):
            report = self.state.qa_report
            if report is None or report.passed:
                return

            log.info(
                "job %s final QA failed; starting writer repair attempt %s/%s",
                self.state.id,
                attempt,
                attempts,
            )
            _emit_progress(
                self.state,
                "final_qa_repair_started",
                repair_attempt=attempt,
                max_repair_attempts=attempts,
            )
            try:
                artifact = self._repair_final_qa_once(build_dir, report)
            except Exception as exc:  # noqa: BLE001 — repair is best-effort; still deliver artifacts
                log.warning("final QA repair attempt failed: %s", exc)
                artifact = None
            self.state.final_qa_repair_attempts += 1
            if artifact is not None:
                self.state.artifacts.append(artifact)
                self.state.qa_report = run_final_qa(build_dir)
                if self.state.qa_report.passed:
                    _append_note(
                        self.state.qa_report,
                        f"Final QA passed after {attempt} writer repair attempt(s).",
                    )
                    return
                continue

            _append_note(
                report,
                f"Final QA repair attempt {attempt}/{attempts} did not produce a valid patch.",
            )
            return

        if self.state.qa_report and not self.state.qa_report.passed:
            _append_note(
                self.state.qa_report,
                f"Final QA still failing after {attempts} writer repair attempt(s).",
            )

    def _completion_payload(self, build_dir: str | None = None) -> dict:
        payload: dict[str, Any] = {
            "status": self.state.status,
            "job_id": self.state.id,
            "project_name": self.state.project_name,
            "final_qa_repair_attempts": self.state.final_qa_repair_attempts,
        }
        if build_dir:
            payload["build_dir"] = build_dir
        if self.state.qa_report:
            qa = self.state.qa_report.model_dump(mode="json")
            payload["qa_report"] = qa
            payload["artifact_urls"] = qa.get("artifact_urls", [])
            payload["qa_passed"] = self.state.qa_report.passed
        if self.state.zip_path:
            payload["zip_path"] = self.state.zip_path
        if self.state.zip_url:
            payload["zip_url"] = self.state.zip_url
        if self.state.patch:
            payload["patch"] = self.state.patch
        return payload

    def _build_subtask(self, subtask: SubTask, build_dir: str, *, index: int, total: int) -> None:
        writer = WriterCrew(workspace_dir=build_dir)
        reviewer = ReviewerCrew(workspace_dir=build_dir)
        listing_tool = WorkspaceListTool(workspace_dir=build_dir)

        prior_issues = ""
        artifact: CodeArtifact | None = None
        review: ReviewResult | None = None
        attempts = 0

        _emit_progress(
            self.state,
            "subtask_started",
            subtask_id=subtask.id,
            title=subtask.title,
            file_path=subtask.file_path,
            index=index,
            total=total,
        )

        for attempt in range(_max_subtask_retries() + 1):
            attempts = attempt + 1
            write_result = writer.crew().kickoff(
                inputs={
                    "subtask": subtask.model_dump_json(indent=2),
                    "workspace_dir": build_dir,
                    "workspace_listing": listing_tool._run("."),
                    "amendments": self.state.amendments or "(none)",
                    "prior_review_issues": prior_issues or "(none)",
                }
            )
            artifact = write_result.pydantic if isinstance(write_result.pydantic, CodeArtifact) else None
            if artifact is None:
                prior_issues = "Writer did not return a valid CodeArtifact; try again and emit the schema exactly."
                continue

            persist_error = _persist_artifact(artifact, build_dir)
            if persist_error:
                prior_issues = persist_error
                continue

            deterministic = run_deterministic_review(subtask, artifact, build_dir)
            review = deterministic.result
            if deterministic.needs_fallback:
                review_result = reviewer.crew().kickoff(
                    inputs={
                        "subtask": subtask.model_dump_json(indent=2),
                        "artifact": artifact.model_dump_json(indent=2),
                        "workspace_dir": build_dir,
                    }
                )
                review = (
                    review_result.pydantic
                    if isinstance(review_result.pydantic, ReviewResult)
                    else deterministic.result
                )

            if review.passed:
                break
            next_issues = "\n".join(review.issues) if review.issues else "review failed without detail"
            if review.issues and all("SKIP:" in issue for issue in review.issues):
                log.warning("subtask %s: deterministic review skipped; not retrying writer", subtask.id)
                break
            # If the reviewer returns the same issues twice in a row, the writer
            # cannot fix them (systemic problem — missing tool, env, etc.).
            # Stop burning retries and let finalize/QA surface it instead.
            if attempt > 0 and next_issues == prior_issues:
                log.warning(
                    "subtask %s: identical review issues across retries, short-circuiting",
                    subtask.id,
                )
                break
            prior_issues = next_issues

        if artifact is not None:
            self.state.artifacts.append(artifact)
        if review is not None:
            self.state.review_results.append(review)

        if review and review.passed:
            _emit_progress(
                self.state,
                "subtask_completed",
                subtask_id=subtask.id,
                title=subtask.title,
                file_path=subtask.file_path,
                index=index,
                total=total,
                attempts=attempts,
            )
        else:
            _emit_progress(
                self.state,
                "subtask_failed",
                subtask_id=subtask.id,
                title=subtask.title,
                file_path=subtask.file_path,
                index=index,
                total=total,
                attempts=attempts,
                issues=review.issues if review else ["Writer did not return a valid artifact."],
            )


def kickoff() -> Any:
    return CodebuilderFlow().kickoff(
        inputs={
            "id": "local-dev-session",
            "project_name": "criador-de-piada",
            "brief": "Um projeto python extremamente simples que cria piadas usando OpenAI",
            "goals": ["Criar piadas"],
            "tech_stack": ["python", "openai"],
            "attachments": [],
        }
    )


def resume(job_id: str, feedback: str = "") -> Any:
    return CodebuilderFlow.from_pending(job_id).resume(feedback)


def plot():
    CodebuilderFlow().plot("codebuilder_flow")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kickoff()
