"""Codebuilder Flow — plans, gates on human approval, then builds."""

from __future__ import annotations

import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import requests
from crewai.flow import Flow, listen, start
from crewai.flow.human_feedback import human_feedback

from codebuilder import history
from codebuilder.crews.planner_crew import PlannerCrew
from codebuilder.crews.reviewer_crew import ReviewerCrew
from codebuilder.crews.writer_crew import WriterCrew
from codebuilder.runtime_qa import (
    artifact_refs,
    looks_like_placeholder,
    persist_artifact,
    persist_bundle_artifact,
    plan_summary,
    qa_report_for_repair,
    run_bundle_deterministic_review,
    run_final_qa,
    run_full_architecture_gate,
    run_import_completeness_gate,
    validate_plan,
)
from codebuilder.tools.workspace_tool import WorkspaceListTool, resolve_within
from codebuilder.schemas import (
    Attachment,
    ArtifactRef,
    CodeBundleArtifact,
    CodeArtifact,
    CodebuilderState,
    Plan,
    QAReport,
    ReviewResult,
    SubTask,
)
from codebuilder.tools import attachment_tool, git_tool
from codebuilder.tools.s3_artifacts import SKIP_DIRS, SKIP_FILES, upload_file, upload_workspace


log = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("CODEBUILDER_WORKSPACE_ROOT", "./workspaces")).resolve()
DEFAULT_MAX_SUBTASK_RETRIES = 1
DEFAULT_MAX_FINAL_QA_REPAIRS = 1
PROGRESS_WEBHOOK_TIMEOUT_SECONDS = 5

GUARDRAIL_LLM = os.environ.get("CODEBUILDER_GUARDRAIL_LLM", "openai/gpt-5.4-mini")


_ZIP_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
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


def _append_note(report: QAReport, note: str) -> None:
    report.integration_notes = " ".join(part for part in (report.integration_notes, note) if part)


def _emit_progress(state: CodebuilderState, event_type: str, **payload: Any) -> None:
    """Best-effort progress callback for UIs that need finer updates than AMP provides."""
    webhook = os.environ.get("CODEBUILDER_PROGRESS_WEBHOOK")
    if not webhook:
        return

    body = {
        "event_type": event_type,
        "session_id": state.session_id,
        "flow_id": state.id,
        "job_id": state.id,  # backward-compat alias for flow_id
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

class CodebuilderFlow(Flow[CodebuilderState]):
    """Single flow: ingest → plan (HITL) → build → finalize."""

    @start()
    def ingest(self):
        # CrewAI Flow auto-merges `inputs={...}` keys into self.state before
        # this method runs. Declared CodebuilderState fields (session_id,
        # brief, project_name, goals, tech_stack, attachments) are populated
        # by the time we get here. Do NOT pass `id` in inputs — overriding
        # state.id breaks AMP OTel trace correlation (CON-101 / COR-48):
        # OTel emits under the auto-generated flow_id, AMP looks up traces
        # under the overridden state.id, and the two disagree, so traces from
        # the pre-resume phase are stranded on Wharf. Use `session_id` for
        # the caller's identity and let `state.id` stay as the flow's UUID.
        self.state.attachments = [
            a if isinstance(a, Attachment) else Attachment(**a)
            for a in self.state.attachments
        ]

        # Caller's session_id keys workspace dir, project_key fallback, and
        # webhook payloads so the frontend can correlate. If unset (e.g. local
        # `uv run kickoff`), fall back to the flow_id so paths remain unique.
        session_key = self.state.session_id or self.state.id
        workspace_dir = WORKSPACE_ROOT / session_key
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
                "session %s has no project_name and no git attachment; "
                "falling back to session_id for history keying",
                session_key,
            )
            project_key = session_key
        self.state.project_key = project_key

        # NOTE: do not mutate CREWAI_STORAGE_DIR here. crewai treats that env
        # var as the `app_name` input to `appdirs.user_data_dir(...)`, which
        # determines where `SQLiteFlowPersistence()` (with no db_path) stores
        # pending-feedback rows. `Flow.from_pending(flow_id)` always constructs
        # a default SQLiteFlowPersistence, so any drift in the env var between
        # save-time and resume-time breaks HITL resume with "No pending
        # feedback found for flow_id".

        self.state.status = "planning"
        log.info(
            "session %s ingested (flow_id=%s); workspace=%s project_key=%s",
            self.state.session_id or "(unset)",
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
        plan_obj = validate_plan(result.pydantic)
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
        # revise_plan runs DURING resume, AFTER resume_async has already cleared
        # the pending-feedback row. If anything here raised, the exception would
        # propagate out of resume() with no pending row left, and any later
        # from_pending(job_id) would fail with "No pending feedback found" — the
        # job would be permanently unresumable. So a revision failure must NEVER
        # raise: fall back to the prior plan, surface the failure as an
        # open_question, and let @human_feedback re-gate (re-pause + re-persist)
        # so the user can retry or approve the prior plan as-is.
        try:
            result = PlannerCrew().amend_crew().kickoff(inputs=_planner_inputs(self.state))
            plan_obj = validate_plan(result.pydantic)
        except Exception as exc:  # noqa: BLE001 — a revise failure must never brick the job
            fallback = self._prior_plan_snapshot(prior)
            if fallback is None:
                # Pathological: no prior plan to show. A plan must have existed to
                # reach the amend gate, so this should not happen; re-raise rather
                # than fabricate one.
                raise
            log.warning("plan revision failed (%s); re-gating with the prior plan", exc)
            fallback.open_questions = [
                f"Automatic plan revision failed ({exc}). The previous plan is shown "
                "unchanged — re-state your changes to try again, or approve to build it as-is.",
                *fallback.open_questions,
            ]
            plan_obj = fallback
        self.state.plan = plan_obj
        self.state.status = "awaiting_approval"
        return plan_obj.model_dump()

    def _prior_plan_snapshot(self, prior) -> Plan | None:
        """Best-effort recovery of the plan the human last reviewed, for re-gating
        when a revision fails. Prefers the live state plan; falls back to the
        feedback result's `output` (the dict that was shown to the human)."""
        if self.state.plan is not None:
            return self.state.plan.model_copy(deep=True)  # StrictOutputModel is not frozen
        prior_output = getattr(prior, "output", None)
        if isinstance(prior_output, dict):
            try:
                return Plan.model_validate(prior_output)
            except Exception:  # noqa: BLE001 — fall through to None
                return None
        return None

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
            inputs_dir = Path(self.state.workspace_dir) / "inputs"
            repo_root = next(
                (str(c) for c in [inputs_dir / "repo", *inputs_dir.glob("repo*")] if c.is_dir()),
                None,
            )
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
        self._import_gate_overflow: list[str] = []
        self._run_import_completeness_pass(build_dir)
        self.state.qa_report = run_final_qa(build_dir)
        if self._import_gate_overflow and self.state.qa_report:
            self.state.qa_report.passed = False
            _append_note(
                self.state.qa_report,
                "Import completeness gate found more missing modules than the auto-stub cap "
                f"could cover. Unresolved paths: {', '.join(self._import_gate_overflow)}.",
            )
        self._repair_final_qa_if_needed(build_dir)
        if self.state.plan and self.state.plan.mode == "new_project":
            architecture_review = run_full_architecture_gate(build_dir, self.state.plan)
            self.state.review_results.append(architecture_review)
            if self.state.qa_report and not architecture_review.passed:
                self.state.qa_report.passed = False
                _append_note(
                    self.state.qa_report,
                    "Architecture gate failed: " + "; ".join(architecture_review.issues),
                )
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
            session_segment = self.state.project_key or self.state.session_id or self.state.id
            prefix = f"{session_segment}/{self.state.id}"
            self.state.qa_report.artifact_urls = artifact_refs(upload_workspace(build_dir, prefix=prefix))
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

        self.state.status = (
            "done" if self.state.qa_report is None or self.state.qa_report.passed else "failed"
        )
        log.info("job %s complete", self.state.id)

        try:
            history.record(self.state)
        except Exception as exc:  # noqa: BLE001 — history is observability, never fatal
            log.warning("history.record on finalize failed: %s", exc)

        return self._completion_payload(build_dir)

    # --- helpers ---------------------------------------------------------

    def _run_import_completeness_pass(self, build_dir: str) -> None:
        """Detect missing project-local modules and run stub subtasks for them.

        Runs before final QA so cascading ModuleNotFoundError failures during
        pytest collection get resolved by one writer pass per missing file
        (capped at 8 stubs). If more than the cap are missing, attaches a
        note to the QA report once it's generated.
        """
        plan = self.state.plan
        if plan is None:
            return
        try:
            missing_paths, stubs = run_import_completeness_gate(build_dir, plan)
        except Exception as exc:  # noqa: BLE001 — gate is best-effort, never fatal
            log.warning("import completeness gate failed: %s", exc)
            return

        if not missing_paths:
            return

        log.warning(
            "import completeness gate found %d missing modules; running %d stub subtask(s)",
            len(missing_paths),
            len(stubs),
        )
        _emit_progress(
            self.state,
            "import_completeness_started",
            missing_count=len(missing_paths),
            stub_count=len(stubs),
        )
        for i, stub in enumerate(stubs, start=1):
            self._build_subtask(stub, build_dir, index=i, total=len(stubs))

        overflow = missing_paths[len(stubs):]
        if overflow:
            self._import_gate_overflow = overflow
        _emit_progress(
            self.state,
            "import_completeness_completed",
            stub_count=len(stubs),
            overflow_count=len(overflow),
        )

    def _repair_final_qa_once(self, build_dir: str, report: QAReport) -> CodeArtifact | None:
        writer = WriterCrew(workspace_dir=build_dir)
        listing_tool = WorkspaceListTool(workspace_dir=build_dir)
        result = writer.repair_crew().kickoff(
            inputs={
                "workspace_dir": build_dir,
                "workspace_listing": listing_tool._run("."),
                "plan_summary": plan_summary(self.state.plan),
                "qa_report": qa_report_for_repair(report),
            }
        )
        artifact = result.pydantic if isinstance(result.pydantic, CodeArtifact) else None
        if artifact is None:
            log.warning("final QA repair writer did not return a CodeArtifact")
            return None
        if artifact.subtask_id != "final_qa_repair":
            artifact.subtask_id = "final_qa_repair"
        persist_error = persist_artifact(artifact, build_dir)
        if persist_error:
            log.warning("final QA repair could not be persisted: %s", persist_error)
            return None
        if looks_like_placeholder(artifact.content):
            log.warning("final QA repair produced placeholder content: %s", artifact.file_path)
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
            "session_id": self.state.session_id,
            "flow_id": self.state.id,
            "job_id": self.state.id,  # backward-compat alias for flow_id
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

        existing_snapshots: dict[str, str] = {}
        for planned_file in subtask.files:
            if planned_file.change_type != "modify":
                continue
            try:
                target = resolve_within(build_dir, planned_file.path)
            except ValueError:
                target = None
            if target is not None and target.is_file():
                raw = target.read_text(encoding="utf-8", errors="replace")
                limit = 12000
                if len(raw) > limit:
                    omitted = len(raw) - limit
                    existing_snapshots[planned_file.path] = (
                        f"{raw[:limit]}\n\n[truncated {omitted} chars — call workspace_read for full file]"
                    )
                else:
                    existing_snapshots[planned_file.path] = raw

        prior_issues = ""
        bundle: CodeBundleArtifact | None = None
        review: ReviewResult | None = None
        attempts = 0
        file_paths = subtask.file_paths
        primary_file_path = file_paths[0] if file_paths else ""
        existing_contents = (
            "\n\n".join(
                f"-----BEGIN EXISTING FILE: {path}-----\n{content}\n-----END EXISTING FILE: {path}-----"
                for path, content in existing_snapshots.items()
            )
            or "(no pre-existing planned files)"
        )

        _emit_progress(
            self.state,
            "subtask_started",
            subtask_id=subtask.id,
            title=subtask.title,
            file_path=primary_file_path,
            file_paths=file_paths,
            index=index,
            total=total,
        )

        for attempt in range(_max_subtask_retries() + 1):
            attempts = attempt + 1
            write_result = writer.crew().kickoff(
                inputs={
                    "subtask": subtask.model_dump_json(indent=2),
                    "change_type": "bundle",
                    "existing_contents": existing_contents,
                    "workspace_dir": build_dir,
                    "workspace_listing": listing_tool._run("."),
                    "amendments": self.state.amendments or "(none)",
                    "prior_review_issues": prior_issues or "(none)",
                }
            )
            bundle = (
                write_result.pydantic
                if isinstance(write_result.pydantic, CodeBundleArtifact)
                else None
            )
            if bundle is None:
                prior_issues = (
                    "Writer did not return a valid CodeBundleArtifact; "
                    "try again and emit the schema exactly."
                )
                continue

            persist_errors = persist_bundle_artifact(bundle, subtask, build_dir)
            if persist_errors:
                prior_issues = "\n".join(persist_errors)
                continue

            deterministic = run_bundle_deterministic_review(
                subtask,
                bundle,
                build_dir,
                existing_snapshots=existing_snapshots,
            )
            review = deterministic.result
            if deterministic.needs_fallback:
                review_result = reviewer.crew().kickoff(
                    inputs={
                        "subtask": subtask.model_dump_json(indent=2),
                        "artifact": bundle.model_dump_json(indent=2),
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

        if bundle is not None:
            self.state.artifacts.extend(bundle.artifacts)
        if review is not None:
            self.state.review_results.append(review)

        if review and review.passed:
            _emit_progress(
                self.state,
                "subtask_completed",
                subtask_id=subtask.id,
                title=subtask.title,
                file_path=primary_file_path,
                file_paths=file_paths,
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
                file_path=primary_file_path,
                file_paths=file_paths,
                index=index,
                total=total,
                attempts=attempts,
                issues=review.issues if review else ["Writer did not return a valid artifact."],
            )


def kickoff() -> Any:
    return CodebuilderFlow().kickoff(
        inputs={
            "session_id": "local-dev-session",
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
