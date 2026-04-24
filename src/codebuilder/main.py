"""Codebuilder Flow — plans, gates on human approval, then builds."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

from crewai.flow import Flow, HumanFeedbackPending, listen, persist, start
from crewai.flow.human_feedback import human_feedback

from codebuilder import history
from codebuilder.crews.planner_crew import PlannerCrew
from codebuilder.crews.reviewer_crew import ReviewerCrew
from codebuilder.crews.writer_crew import WriterCrew
from codebuilder.feedback_provider import WebhookFeedbackProvider
from codebuilder.schemas import (
    Attachment,
    CodeArtifact,
    CodebuilderState,
    Plan,
    QAReport,
    ReviewResult,
    SubTask,
)
from codebuilder.tools import attachment_tool, git_tool
from codebuilder.tools.s3_artifacts import SKIP_DIRS, SKIP_FILES, upload_file, upload_workspace
from codebuilder.tools.workspace_tool import WorkspaceListTool


log = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("CODEBUILDER_WORKSPACE_ROOT", "./workspaces")).resolve()
MAX_SUBTASK_RETRIES = 2
AMEND_FEEDBACK_PROVIDER = WebhookFeedbackProvider()


_ZIP_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_zip_stem(name: str) -> str:
    stem = _ZIP_NAME_RE.sub("-", name).strip("-.") or "project"
    return stem[:80]


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


@persist()
class CodebuilderFlow(Flow[CodebuilderState]):
    """Single flow: ingest → plan (HITL) → build → finalize."""

    @start()
    def ingest(self, crewai_trigger_payload: dict | str | None = None):
        # AMP sends `inputs` as a flat KV of strings; accept a JSON string
        # here so both local CLI (dict) and AMP (str) paths work.
        if isinstance(crewai_trigger_payload, str):
            try:
                crewai_trigger_payload = json.loads(crewai_trigger_payload)
            except json.JSONDecodeError:
                log.warning("crewai_trigger_payload was a non-JSON string; ignoring")
                crewai_trigger_payload = None
        payload = crewai_trigger_payload or {}
        self.state.brief = payload.get("brief", "") or self.state.brief
        self.state.project_name = payload.get("project_name", "") or self.state.project_name
        self.state.goals = payload.get("goals") or self.state.goals
        self.state.tech_stack = payload.get("tech_stack") or self.state.tech_stack
        self.state.attachments = [
            Attachment(**a) if not isinstance(a, Attachment) else a
            for a in (payload.get("attachments") or [])
        ] or self.state.attachments

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
                "falling back to flow_id for memory/history keying",
                self.state.id,
            )
            project_key = self.state.id
        self.state.project_key = project_key

        memory_dir = WORKSPACE_ROOT / "_memory" / project_key
        memory_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CREWAI_STORAGE_DIR"] = str(memory_dir)

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
        llm="openai/gpt-5.4",
        default_outcome="amend",
        provider=AMEND_FEEDBACK_PROVIDER,
    )
    def plan(self) -> dict:
        result = PlannerCrew().crew().kickoff(inputs=_planner_inputs(self.state))
        plan_obj: Plan = result.pydantic
        self.state.plan = plan_obj
        self.state.status = "awaiting_approval"
        return plan_obj.model_dump()

    @listen("amend")
    @human_feedback(
        message="Revised plan — please review again. Approve, amend further, or reject.",
        emit=["approved", "amend", "rejected"],
        llm="openai/gpt-5.4",
        default_outcome="amend",
        provider=AMEND_FEEDBACK_PROVIDER,
    )
    def revise_plan(self, prior) -> dict:
        self.state.amendments = getattr(prior, "feedback", "") or ""
        self.state.amend_cycles += 1
        result = PlannerCrew().crew().kickoff(inputs=_planner_inputs(self.state))
        plan_obj: Plan = result.pydantic
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

        for subtask in plan.subtasks:
            self._build_subtask(subtask, build_dir)

        self._build_dir = build_dir

    @listen(build)
    def finalize(self, _prior=None):
        build_dir = getattr(self, "_build_dir", self.state.workspace_dir)
        if self.state.status == "failed":
            try:
                history.record(self.state)
            except Exception as exc:  # noqa: BLE001 — history is observability, never fatal
                log.warning("history.record on build failure failed: %s", exc)
            return

        listing_tool = WorkspaceListTool(workspace_dir=self.state.workspace_dir)
        qa_result = (
            ReviewerCrew(workspace_dir=build_dir)
            .qa_crew()
            .kickoff(
                inputs={
                    "workspace_dir": build_dir,
                    "plan_summary": self.state.plan.model_dump_json(indent=2) if self.state.plan else "",
                    "workspace_listing": listing_tool._run("."),
                }
            )
        )
        self.state.qa_report = qa_result.pydantic if isinstance(qa_result.pydantic, QAReport) else QAReport(
            passed=False,
            integration_notes="QA agent returned non-schema output",
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
            self.state.qa_report.artifact_urls = upload_workspace(build_dir, prefix=prefix)
            if self.state.zip_path:
                zip_ref = upload_file(
                    self.state.zip_path,
                    key=f"{prefix}/{Path(self.state.zip_path).name}",
                )
                if zip_ref:
                    self.state.zip_url = zip_ref["url"]
                    self.state.qa_report.artifact_urls.append(zip_ref)
        except Exception as exc:  # noqa: BLE001
            log.warning("artifact upload failed: %s", exc)

        self.state.status = "done"
        log.info("job %s complete", self.state.id)

        try:
            history.record(self.state)
        except Exception as exc:  # noqa: BLE001 — history is observability, never fatal
            log.warning("history.record on finalize failed: %s", exc)

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

    def _build_subtask(self, subtask: SubTask, build_dir: str) -> None:
        writer = WriterCrew(workspace_dir=build_dir)
        reviewer = ReviewerCrew(workspace_dir=build_dir)
        listing_tool = WorkspaceListTool(workspace_dir=build_dir)

        prior_issues = ""
        artifact: CodeArtifact | None = None
        review: ReviewResult | None = None

        for attempt in range(MAX_SUBTASK_RETRIES + 1):
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

            review_result = reviewer.crew().kickoff(
                inputs={
                    "subtask": subtask.model_dump_json(indent=2),
                    "artifact": artifact.model_dump_json(indent=2),
                    "workspace_dir": build_dir,
                }
            )
            review = review_result.pydantic if isinstance(review_result.pydantic, ReviewResult) else ReviewResult(
                subtask_id=subtask.id, passed=False, issues=["Reviewer output not parseable"]
            )

            if review.passed:
                break
            next_issues = "\n".join(review.issues) if review.issues else "review failed without detail"
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


# --- Entrypoints for CLI + Flask ---------------------------------------------


def _parse_payload() -> dict:
    if len(sys.argv) < 2:
        return {}
    raw = sys.argv[1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        path = Path(raw)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        raise


def kickoff(payload: dict | None = None) -> Any:
    payload = payload if payload is not None else _parse_payload()
    flow = CodebuilderFlow()
    result = flow.kickoff(inputs={"crewai_trigger_payload": payload})
    if isinstance(result, HumanFeedbackPending):
        print(json.dumps({"status": "pending", "job_id": result.context.flow_id}))
    else:
        print(json.dumps(_cli_summary(flow)))
    return result


def kickoff_cli() -> None:
    kickoff()


def _cli_summary(flow: "CodebuilderFlow") -> dict:
    out = {"status": flow.state.status, "job_id": flow.state.id}
    if flow.state.zip_path:
        out["zip_path"] = flow.state.zip_path
    if flow.state.zip_url:
        out["zip_url"] = flow.state.zip_url
    return out


def resume(job_id: str | None = None, feedback: str | None = None) -> Any:
    if job_id is None:
        if len(sys.argv) < 2:
            raise SystemExit("Usage: resume <job_id> [feedback]")
        job_id = sys.argv[1]
        feedback = sys.argv[2] if len(sys.argv) > 2 else ""
    flow = CodebuilderFlow.from_pending(job_id)
    result = flow.resume(feedback or "")
    if isinstance(result, HumanFeedbackPending):
        print(json.dumps({"status": "pending", "job_id": result.context.flow_id}))
    else:
        print(json.dumps(_cli_summary(flow)))
    return result


def resume_cli() -> None:
    resume()


def plot():
    CodebuilderFlow().plot("codebuilder_flow")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kickoff_cli()
