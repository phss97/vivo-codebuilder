"""Codebuilder Flow — plans, gates on human approval, then builds."""

from __future__ import annotations

import logging
import json
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
    MODIFY_PREIMAGE_LIMIT,
    artifact_refs,
    build_symbol_index,
    looks_like_placeholder,
    persist_artifact,
    persist_bundle_artifact,
    plan_summary,
    qa_report_for_repair,
    rpa_packaging_remediation_subtask,
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
    FileSkeleton,
    Plan,
    ProjectArchiveRef,
    QAReport,
    ReviewResult,
    SubTask,
)
from codebuilder.tools import attachment_tool, git_tool
from codebuilder.tools.s3_artifacts import SKIP_DIRS, SKIP_FILES, upload_file, upload_workspace


log = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("CODEBUILDER_WORKSPACE_ROOT", "./workspaces")).resolve()
DEFAULT_MAX_SUBTASK_RETRIES = 1
DEFAULT_PATCH_FINAL_QA_REPAIRS = 1
DEFAULT_NEW_PROJECT_FINAL_QA_REPAIRS = 2
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
    return _env_int("CODEBUILDER_MAX_FINAL_QA_REPAIRS", DEFAULT_NEW_PROJECT_FINAL_QA_REPAIRS)


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


def _prompt_payload_stats(inputs: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(inputs, default=str, ensure_ascii=False)
    return {
        "input_chars": len(serialized),
        "input_keys": sorted(inputs.keys()),
    }


def _emit_prompt_inputs_prepared(
    state: CodebuilderState,
    event_type: str,
    inputs: dict[str, Any],
    **payload: Any,
) -> None:
    _emit_progress(state, event_type, **payload, **_prompt_payload_stats(inputs))


def _usage_payload(result: Any) -> dict[str, Any] | str | None:
    usage = getattr(result, "token_usage", None) or getattr(result, "usage_metrics", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return usage
    return str(usage)


def _emit_usage_metrics(state: CodebuilderState, stage: str, result: Any, **payload: Any) -> None:
    usage = _usage_payload(result)
    if usage is None:
        return
    _emit_progress(state, "llm_usage", stage=stage, usage=usage, **payload)


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
            if path.resolve() == out_path.resolve():
                continue
            if path.name in SKIP_FILES:
                continue
            rel = path.relative_to(src)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            zf.write(path, arcname=f"{arcroot}/{rel.as_posix()}")
    return out_path


_PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", ".git", "src")


def _has_project_markers(path: Path) -> bool:
    return any((path / marker).exists() for marker in _PROJECT_MARKERS)


def _descend_wrapper_dirs(path: Path, max_depth: int = 3) -> Path:
    """Step into single-child wrapper dirs (zip-of-a-folder) until markers appear."""
    current = path
    for _ in range(max_depth):
        if _has_project_markers(current):
            return current
        children = [c for c in current.iterdir() if not c.name.startswith(".")]
        if len(children) == 1 and children[0].is_dir():
            current = children[0]
            continue
        break
    return current


def _resolve_patch_root(workspace_dir: str) -> str | None:
    """Locate the attached project root for a ``patch_existing`` job.

    Prefers git clones (``inputs/repo*``), then any ``inputs/`` directory
    carrying project markers, then the only/first directory present. Zip
    attachments extract to ``inputs/<zip-stem>``, so they resolve here too —
    falling back to the workspace root (the old behavior for non-git inputs)
    made the writer duplicate the project tree at the root and final QA sweep
    the user's pre-existing code. Returns ``None`` when ``inputs/`` holds no
    directories at all.
    """
    inputs_dir = Path(workspace_dir) / "inputs"
    if not inputs_dir.is_dir():
        return None
    candidates = sorted((c for c in inputs_dir.iterdir() if c.is_dir()), key=lambda c: c.name)
    if not candidates:
        return None
    repos = [c for c in candidates if c.name.startswith("repo")]
    if repos:
        return str(repos[0])
    marked = [c for c in candidates if _has_project_markers(c)]
    if marked:
        chosen = marked[0]
    else:
        chosen = candidates[0]
        if len(candidates) > 1:
            log.warning(
                "patch_existing: %d candidate dirs under inputs/ and none has project markers; "
                "defaulting to %s",
                len(candidates),
                chosen,
            )
    return str(_descend_wrapper_dirs(chosen))


def _strip_patch_root_prefix(plan: Plan, prefix: str) -> None:
    """Rewrite planned file paths that carry the patch root's workspace prefix.

    The planner is told to emit paths relative to the attached project root,
    but plans sometimes arrive workspace-relative ('inputs/<dir>/src/x.py').
    Stripping the prefix keeps both conventions resolving inside build_dir
    instead of duplicating the tree one level up.
    """
    if not prefix.endswith("/"):
        prefix += "/"
    for subtask in plan.subtasks:
        for planned_file in subtask.files:
            if planned_file.path.startswith(prefix):
                planned_file.path = planned_file.path[len(prefix):]


def _format_attachment_records(records: list[dict[str, str]]) -> str:
    if not records:
        return "(no attachments)"
    lines: list[str] = []
    for record in records:
        kind = record.get("kind") or "attachment"
        name = record.get("name") or "(unnamed)"
        path = record.get("path") or "(no path)"
        summary = record.get("summary") or ""
        suffix = f": {summary}" if summary else ""
        lines.append(f"- {kind} {name} at {path}{suffix}")
    return "\n".join(lines)


def _workspace_context_for_files(build_dir: str, files: list[FileSkeleton]) -> str:
    root = Path(build_dir).resolve()
    listing_tool = WorkspaceListTool(workspace_dir=build_dir)
    planned_lines: list[str] = []
    parent_dirs: set[str] = set()

    for planned_file in files:
        try:
            target = resolve_within(build_dir, planned_file.path)
        except ValueError as exc:
            planned_lines.append(
                f"- {planned_file.path} ({planned_file.change_type}): invalid path ({exc})"
            )
            continue

        exists = "exists" if target.is_file() else "missing"
        planned_lines.append(
            f"- {planned_file.path} ({planned_file.change_type}, {exists}): "
            f"{planned_file.purpose}"
        )
        parent = target.parent
        try:
            parent_rel = parent.relative_to(root).as_posix()
        except ValueError:
            continue
        parent_dirs.add(parent_rel if parent_rel != "." else ".")

    listing_sections: list[str] = []
    for parent in sorted(parent_dirs):
        listing_sections.append(
            f"-----BEGIN DIRECTORY LISTING: {parent}-----\n"
            f"{listing_tool._run(parent)}\n"
            f"-----END DIRECTORY LISTING: {parent}-----"
        )

    planned = "\n".join(planned_lines) or "(no planned files)"
    listings = "\n\n".join(listing_sections) or "(no parent directories to list)"
    return (
        "Planned files:\n"
        f"{planned}\n\n"
        "Scoped directory listings:\n"
        f"{listings}"
    )


def _subtask_workspace_context(build_dir: str, subtask: SubTask) -> str:
    return _workspace_context_for_files(build_dir, subtask.files)


MAX_REPAIR_CONTEXT_FILES = 40


def _repair_workspace_context(
    build_dir: str,
    plan: Plan | None,
    artifacts: list[CodeArtifact],
) -> str:
    planned_by_path: dict[str, FileSkeleton] = {}
    if plan:
        for subtask in plan.subtasks:
            for planned_file in subtask.files:
                planned_by_path[planned_file.path] = planned_file

    files: list[FileSkeleton] = []
    seen: set[str] = set()
    for artifact in artifacts:
        if not artifact.file_path or artifact.file_path in seen:
            continue
        files.append(
            planned_by_path.get(artifact.file_path)
            or FileSkeleton(
                path=artifact.file_path,
                purpose="File changed during this run.",
                change_type="modify",
            )
        )
        seen.add(artifact.file_path)
        # Cap always: a large job touches dozens of files and each adds a
        # directory listing — uncapped this dominated the repair prompt.
        if len(files) >= MAX_REPAIR_CONTEXT_FILES:
            break

    if not files and plan:
        for subtask in plan.subtasks:
            for planned_file in subtask.files:
                if planned_file.path not in seen:
                    files.append(planned_file)
                    seen.add(planned_file.path)
                if len(files) >= MAX_REPAIR_CONTEXT_FILES:
                    break
            if len(files) >= MAX_REPAIR_CONTEXT_FILES:
                break

    return _workspace_context_for_files(build_dir, files)


def _preimage_for_prompt(content: str) -> str:
    """Writer-facing view of a modify target's preimage, truncated only for the
    rare file larger than ``MODIFY_PREIMAGE_LIMIT``. The marker tells the writer
    not to rewrite wholesale; the review guard rejects a shorter result so the
    unshown tail can't be silently dropped."""
    if len(content) <= MODIFY_PREIMAGE_LIMIT:
        return content
    omitted = len(content) - MODIFY_PREIMAGE_LIMIT
    return (
        f"{content[:MODIFY_PREIMAGE_LIMIT]}\n\n[truncated {omitted} chars — file too large to "
        "show in full; make targeted edits and preserve the omitted tail]"
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _upload_file_artifacts_enabled(plan: Plan | None) -> bool:
    default = not (plan and plan.mode == "patch_existing")
    return _env_bool("CODEBUILDER_UPLOAD_FILE_ARTIFACTS", default)


def _planner_inputs(state: CodebuilderState) -> dict:
    records = _format_attachment_records(state.attachment_records)
    if records == "(no attachments)" and state.workspace_dir:
        inputs_dir = Path(state.workspace_dir) / "inputs"
        if inputs_dir.exists():
            listing = WorkspaceListTool(workspace_dir=state.workspace_dir)._run("inputs")
            if listing != "(empty)":
                records = f"Materialized inputs:\n{listing}"
    prior_history = history.summarize_for_planner(state.project_key) if state.project_key else ""
    return {
        "brief": state.brief,
        "project_name": state.project_name or "(unspecified)",
        "goals": "\n".join(f"- {g}" for g in state.goals) or "(none)",
        "tech_stack": ", ".join(state.tech_stack) or "(unspecified)",
        "attachment_records": records,
        "workspace_dir": state.workspace_dir,
        "prior_plan": state.plan.model_dump_json(indent=2) if state.plan else "",
        "prior_history": prior_history or "(no prior runs for this project)",
        "amendments": state.amendments,
        # Override-or-detect hint: a concrete language name when the caller
        # supplied one (the planner must honor it), else an instruction to
        # infer the language from the brief. The planner is the only crew that
        # runs before state.language is resolved, so it gets the hint form.
        "language": state.language or "(detect the language from the brief and goals and use it)",
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
            self.state.attachment_records = attachment_tool.materialize(
                [a.model_dump() for a in self.state.attachments],
                self.state.workspace_dir,
            )
        else:
            self.state.attachment_records = []

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
        planner_inputs = _planner_inputs(self.state)
        _emit_prompt_inputs_prepared(
            self.state,
            "planner_inputs_prepared",
            planner_inputs,
            stage="plan",
        )
        result = (
            PlannerCrew(workspace_dir=self.state.workspace_dir)
            .crew()
            .kickoff(inputs=planner_inputs)
        )
        _emit_usage_metrics(self.state, "plan", result)
        plan_obj = validate_plan(result.pydantic)
        self.state.plan = plan_obj
        # Resolve the output language for every downstream crew: caller override
        # wins, else the planner's detection, else English.
        self.state.language = self.state.language or plan_obj.language or "English"
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
            planner_inputs = _planner_inputs(self.state)
            _emit_prompt_inputs_prepared(
                self.state,
                "planner_inputs_prepared",
                planner_inputs,
                stage="revise_plan",
                amend_cycle=self.state.amend_cycles,
            )
            result = (
                PlannerCrew(workspace_dir=self.state.workspace_dir)
                .amend_crew()
                .kickoff(inputs=planner_inputs)
            )
            _emit_usage_metrics(
                self.state,
                "revise_plan",
                result,
                amend_cycle=self.state.amend_cycles,
            )
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
        # Normally a no-op (amend carries plan.language forward), but guards the
        # edge case where the first plan emitted an empty language.
        self.state.language = self.state.language or plan_obj.language or "English"
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
            patch_root = _resolve_patch_root(self.state.workspace_dir)
            if patch_root is None:
                log.warning(
                    "patch_existing job %s has no attached project under inputs/; "
                    "building at the workspace root",
                    self.state.id,
                )
                build_dir = self.state.workspace_dir
            else:
                build_dir = patch_root
                rel_prefix = Path(build_dir).relative_to(self.state.workspace_dir).as_posix()
                _strip_patch_root_prefix(plan, rel_prefix)
                # Extracted zips aren't git repos; commit the pristine attachment
                # as a baseline so finalize's git diff captures exactly the repair.
                if not (Path(build_dir) / ".git").exists():
                    git_tool.init_and_commit(build_dir, "codebuilder baseline (pre-patch)")
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
        self._run_rpa_packaging_pass(build_dir)
        self.state.qa_report = self._run_final_qa(build_dir)
        if self._import_gate_overflow and self.state.qa_report:
            self.state.qa_report.passed = False
            _append_note(
                self.state.qa_report,
                "Import completeness gate found more missing modules than the auto-stub cap "
                f"could cover. Unresolved paths: {', '.join(self._import_gate_overflow)}.",
            )
        self._repair_final_qa_if_needed(build_dir)
        if self.state.plan and self.state.plan.mode == "new_project":
            architecture_review = run_full_architecture_gate(
                build_dir, self.state.plan, self.state.language or "English"
            )
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

        # Zip the build for both modes: a repair job's deliverable is the
        # repaired project, not just the diff. SKIP_DIRS keeps .git out.
        if self.state.plan:
            try:
                zip_path = _zip_build(
                    build_dir,
                    Path(self.state.workspace_dir),
                    self.state.project_name or self.state.id,
                )
                self.state.zip_path = str(zip_path)
                self.state.project_archive = ProjectArchiveRef(
                    file_path=zip_path.name,
                    size=zip_path.stat().st_size,
                    local_path=str(zip_path),
                )
                log.info("job %s zipped to %s", self.state.id, zip_path)
            except Exception as exc:  # noqa: BLE001 — archive generation failure is reported in QA
                log.warning("zip generation failed: %s", exc)
                if self.state.qa_report:
                    self.state.qa_report.passed = False
                    _append_note(
                        self.state.qa_report,
                        f"Project archive generation failed: {exc}",
                    )

        if self.state.qa_report:
            session_segment = self.state.project_key or self.state.session_id or self.state.id
            prefix = f"{session_segment}/{self.state.id}"
            uploaded_refs: list[ArtifactRef] = []

            if self.state.zip_path:
                zip_ref = upload_file(
                    self.state.zip_path,
                    key=f"{prefix}/{Path(self.state.zip_path).name}",
                )
                if zip_ref:
                    zip_artifact = ArtifactRef(**{**zip_ref, "kind": "project_archive"})
                    self.state.zip_url = zip_artifact.url
                    if self.state.project_archive:
                        self.state.project_archive.url = zip_artifact.url
                    uploaded_refs.append(zip_artifact)
                elif os.environ.get("CODEBUILDER_ARTIFACT_BUCKET"):
                    self.state.qa_report.passed = False
                    _append_note(
                        self.state.qa_report,
                        "Project archive upload failed: CODEBUILDER_ARTIFACT_BUCKET is set "
                        "but no downloadable archive URL was returned.",
                    )

            if _upload_file_artifacts_enabled(self.state.plan):
                try:
                    uploaded_refs.extend(artifact_refs(upload_workspace(build_dir, prefix=prefix)))
                except Exception as exc:  # noqa: BLE001
                    log.warning("workspace artifact upload failed: %s", exc)
            else:
                _emit_progress(
                    self.state,
                    "file_artifact_upload_skipped",
                    reason="disabled_for_patch_existing",
                )

            self.state.qa_report.artifact_urls = uploaded_refs

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

    def _run_final_qa(self, build_dir: str) -> QAReport:
        """Final QA with mode-aware strictness: a new project that fails to
        install (`uv sync`) fails QA outright; patch jobs degrade to the
        orchestrator's interpreter because the user's project may not be
        uv-installable."""
        plan = self.state.plan
        return run_final_qa(
            build_dir,
            lint_paths=self._final_qa_lint_paths(),
            test_paths=self._final_qa_test_paths(build_dir),
            type_paths=self._final_qa_type_paths(),
            require_installable=bool(plan and plan.mode == "new_project"),
            allow_no_tests=bool(plan and plan.mode == "patch_existing"),
        )

    def _max_final_qa_repairs(self) -> int:
        plan = self.state.plan
        default = (
            DEFAULT_PATCH_FINAL_QA_REPAIRS
            if plan and plan.mode == "patch_existing"
            else DEFAULT_NEW_PROJECT_FINAL_QA_REPAIRS
        )
        return _env_int("CODEBUILDER_MAX_FINAL_QA_REPAIRS", default)

    def _run_rpa_packaging_pass(self, build_dir: str) -> None:
        """Fill in missing Windows build-kit files before final QA.

        The plan mandates the kit, but a writer bundle can drop a file; the
        deterministic RPA gate would then fail the job at finalize with no
        remediation path. Best-effort: failures surface later in the gate.
        """
        plan = self.state.plan
        if plan is None or plan.mode != "new_project" or plan.domain != "rpa":
            return
        try:
            stub = rpa_packaging_remediation_subtask(build_dir)
        except Exception as exc:  # noqa: BLE001 — remediation is best-effort
            log.warning("RPA packaging remediation check failed: %s", exc)
            return
        if stub is None:
            return
        log.warning(
            "RPA packaging kit incomplete; running remediation subtask for: %s",
            ", ".join(stub.file_paths),
        )
        _emit_progress(
            self.state,
            "rpa_packaging_remediation_started",
            file_paths=stub.file_paths,
        )
        self._build_subtask(stub, build_dir, index=1, total=1)

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

    def _final_qa_lint_paths(self) -> list[str] | None:
        """Lint scope for final QA: changed files only for patch jobs.

        Pre-existing lint debt in user files the writer never touched must not
        fail a repair job. ``None`` means lint the whole build dir (new_project,
        or a patch job that somehow produced no artifacts — QA should fail then
        anyway).
        """
        plan = self.state.plan
        if plan is None or plan.mode != "patch_existing":
            return None
        paths = sorted({a.file_path for a in self.state.artifacts if a.file_path})
        return paths or None

    def _final_qa_type_paths(self) -> list[str] | None:
        """Type-check scope for final QA: changed Python files only for patch
        jobs. Without this, mypy runs over the whole repo and a patch fails on
        pre-existing type debt in files the writer never touched (the gate uses
        ``follow_imports = silent`` so dependencies are still consulted for types
        but only these target files report errors). ``None`` = whole package
        (new_project, where every file is in scope)."""
        plan = self.state.plan
        if plan is None or plan.mode != "patch_existing":
            return None
        paths = sorted(
            {
                a.file_path
                for a in self.state.artifacts
                if a.file_path and a.file_path.endswith(".py")
            }
        )
        return paths or None

    def _final_qa_test_paths(self, build_dir: str) -> list[str] | None:
        """Pytest scope for final QA.

        Patch jobs default to relevant tests only: changed test files plus tests
        whose filename matches a changed Python module. ``None`` means run the
        whole suite; ``[]`` means no related tests were found.
        """
        plan = self.state.plan
        if plan is None or plan.mode != "patch_existing":
            return None
        if os.environ.get("CODEBUILDER_PATCH_TEST_SCOPE", "").strip().lower() in {
            "all",
            "full",
            "whole",
        }:
            return None

        root = Path(build_dir)
        paths: set[str] = set()
        module_stems: set[str] = set()
        for artifact in self.state.artifacts:
            rel_path = artifact.file_path or ""
            if not rel_path:
                continue
            rel = Path(rel_path)
            is_test = (
                "tests" in rel.parts
                or rel.name.startswith("test_")
                or rel.name.endswith("_test.py")
            )
            if is_test:
                paths.add(rel.as_posix())
            elif rel.suffix == ".py":
                module_stems.add(rel.stem)

        tests_root = root / "tests"
        if tests_root.is_dir():
            wanted_names = {
                name
                for stem in module_stems
                for name in (f"test_{stem}.py", f"{stem}_test.py")
            }
            for test_file in tests_root.rglob("*.py"):
                if test_file.name in wanted_names:
                    paths.add(test_file.relative_to(root).as_posix())

        return sorted(paths)

    @staticmethod
    def _qa_failure_signature(report: QAReport | None) -> str:
        """A stable string fingerprint of a QA report's failures, used to detect
        a repair pass that changed nothing so we stop burning attempts."""
        if report is None:
            return ""
        return " ".join(
            (report.lint_output or "", report.test_output or "", report.type_output or "")
        )

    def _repair_final_qa_once(self, build_dir: str, report: QAReport) -> list[CodeArtifact]:
        """Run one repair pass, persisting EVERY file the writer returns.

        The writer returns a CodeBundleArtifact so a single pass can fix all the
        files implicated by the failures (systemic drift spans several). A bare
        CodeArtifact is still accepted for backward compatibility.
        """
        writer = WriterCrew(workspace_dir=build_dir)
        repair_inputs = {
            "workspace_dir": build_dir,
            "workspace_listing": _repair_workspace_context(
                build_dir,
                self.state.plan,
                self.state.artifacts,
            ),
            "plan_summary": plan_summary(self.state.plan),
            "qa_report": qa_report_for_repair(report),
            "dependency_contracts": self._symbol_contract(build_dir),
            "language": self.state.language or "English",
        }
        _emit_prompt_inputs_prepared(
            self.state,
            "final_qa_repair_inputs_prepared",
            repair_inputs,
            stage="final_qa_repair",
        )
        result = writer.repair_crew().kickoff(inputs=repair_inputs)
        _emit_usage_metrics(self.state, "final_qa_repair", result)

        pydantic = result.pydantic
        if isinstance(pydantic, CodeBundleArtifact):
            candidates = list(pydantic.artifacts)
        elif isinstance(pydantic, CodeArtifact):
            candidates = [pydantic]
        else:
            log.warning("final QA repair writer did not return a CodeArtifact/CodeBundleArtifact")
            return []

        repaired: list[CodeArtifact] = []
        for artifact in candidates:
            if not artifact.file_path:
                continue
            if artifact.subtask_id != "final_qa_repair":
                artifact.subtask_id = "final_qa_repair"
            persist_error = persist_artifact(artifact, build_dir)
            if persist_error:
                log.warning(
                    "final QA repair could not persist %s: %s", artifact.file_path, persist_error
                )
                continue
            if looks_like_placeholder(artifact.content):
                log.warning("final QA repair produced placeholder content: %s", artifact.file_path)
                continue
            repaired.append(artifact)
        return repaired

    def _repair_final_qa_if_needed(self, build_dir: str) -> None:
        attempts = self._max_final_qa_repairs()
        if attempts <= 0:
            return

        last_signature = self._qa_failure_signature(self.state.qa_report)
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
                repaired = self._repair_final_qa_once(build_dir, report)
            except Exception as exc:  # noqa: BLE001 — repair is best-effort; still deliver artifacts
                log.warning("final QA repair attempt failed: %s", exc)
                repaired = []
            self.state.final_qa_repair_attempts += 1
            if repaired:
                self.state.artifacts.extend(repaired)
                self.state.qa_report = self._run_final_qa(build_dir)
                if self.state.qa_report.passed:
                    _append_note(
                        self.state.qa_report,
                        f"Final QA passed after {attempt} writer repair attempt(s).",
                    )
                    return
                new_signature = self._qa_failure_signature(self.state.qa_report)
                # A pass that changed nothing won't converge — stop early, but only
                # when attempts remain (the last attempt falls through to the
                # post-loop "still failing" summary the callers/tests expect).
                if new_signature == last_signature and attempt < attempts:
                    _append_note(
                        self.state.qa_report,
                        f"Final QA repair attempt {attempt} did not change the failures; "
                        "stopping early.",
                    )
                    return
                last_signature = new_signature
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
        if self.state.project_archive:
            payload["project_archive"] = self.state.project_archive.model_dump(mode="json")
        if self.state.patch:
            payload["patch"] = self.state.patch
        return payload

    def _symbol_contract(self, build_dir: str | None = None) -> str:
        """The name map every writer must import/call against verbatim.

        Two sources, real-first: (1) the *actual* public API extracted from the
        files already written this run (real class fields, constructor and
        function signatures — so a later file imports the true `Settings` fields
        instead of inventing `settings.sap_host`); (2) the planner-declared
        ``public_api`` for files not yet written. Building the index from the
        written artifacts (not the whole tree) keeps it bounded in both modes.
        """
        real_lines: list[str] = []
        if build_dir and self.state.artifacts:
            written = sorted({a.file_path for a in self.state.artifacts if a.file_path})
            try:
                index = build_symbol_index(build_dir, paths=written)
            except Exception as exc:  # noqa: BLE001 — prevention is best-effort, never fatal
                log.warning("symbol index build failed: %s", exc)
                index = {}
            for module in sorted(index):
                compact = index[module].replace("\n", " | ")
                real_lines.append(f"- {module}: {compact}")

        planned_lines: list[str] = []
        plan = self.state.plan
        if plan:
            for subtask in plan.subtasks:
                for planned_file in subtask.files:
                    if planned_file.public_api:
                        symbols = "; ".join(planned_file.public_api)
                        planned_lines.append(f"- {planned_file.path} → [{symbols}]")

        sections: list[str] = []
        if real_lines:
            sections.append(
                "REAL APIs already written this run (import these EXACT names, "
                "fields, and signatures — extracted from the actual code):\n"
                + "\n".join(real_lines)
            )
        if planned_lines:
            sections.append(
                "Planned file APIs from the plan (files not yet written):\n"
                + "\n".join(planned_lines)
            )
        return "\n\n".join(sections) if sections else "(no symbols available)"

    def _build_subtask(self, subtask: SubTask, build_dir: str, *, index: int, total: int) -> None:
        writer = WriterCrew(workspace_dir=build_dir)

        # Full preimage of each modify target — review needs it to detect
        # silent content loss. The writer prompt gets a generously-truncated
        # view (built below); it must never be the only copy of a file.
        existing_snapshots: dict[str, str] = {}
        for planned_file in subtask.files:
            if planned_file.change_type != "modify":
                continue
            try:
                target = resolve_within(build_dir, planned_file.path)
            except ValueError:
                target = None
            if target is not None and target.is_file():
                existing_snapshots[planned_file.path] = target.read_text(
                    encoding="utf-8", errors="replace"
                )

        prior_issues = ""
        bundle: CodeBundleArtifact | None = None
        review: ReviewResult | None = None
        attempts = 0
        file_paths = subtask.file_paths
        primary_file_path = file_paths[0] if file_paths else ""
        existing_contents = (
            "\n\n".join(
                f"-----BEGIN EXISTING FILE: {path}-----\n{_preimage_for_prompt(content)}\n-----END EXISTING FILE: {path}-----"
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
            writer_inputs = {
                "subtask": subtask.model_dump_json(indent=2),
                "change_type": "bundle",
                "existing_contents": existing_contents,
                "workspace_dir": build_dir,
                "workspace_listing": _subtask_workspace_context(build_dir, subtask),
                "amendments": self.state.amendments or "(none)",
                "prior_review_issues": prior_issues or "(none)",
                "dependency_contracts": self._symbol_contract(build_dir),
                "language": self.state.language or "English",
            }
            _emit_prompt_inputs_prepared(
                self.state,
                "writer_inputs_prepared",
                writer_inputs,
                stage="subtask",
                subtask_id=subtask.id,
                attempt=attempts,
            )
            write_result = writer.crew().kickoff(inputs=writer_inputs)
            _emit_usage_metrics(
                self.state,
                "subtask",
                write_result,
                subtask_id=subtask.id,
                attempt=attempts,
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
                reviewer = ReviewerCrew(workspace_dir=build_dir)
                review_result = reviewer.crew().kickoff(
                    inputs={
                        "subtask": subtask.model_dump_json(indent=2),
                        "artifact": bundle.model_dump_json(indent=2),
                        "workspace_dir": build_dir,
                        "language": self.state.language or "English",
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
