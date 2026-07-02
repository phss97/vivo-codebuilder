"""Codebuilder Flow — plans, gates on human approval, then builds.

The planning and building are done by Claude Agent SDK agents (see cc_agent):
an Opus planner and a Sonnet executor. The CrewAI Flow shell is kept only for
what AMP + the frontend depend on — kickoff, the @human_feedback HITL gate,
progress/completion webhooks, S3 upload, and per-project history.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import requests
from crewai.flow import Flow, listen, start
from crewai.flow.human_feedback import human_feedback

from codebuilder import cc_agent, history
from codebuilder.runtime_qa import (
    artifact_refs,
    has_pytest_files,
    qa_report_for_repair,
    run_final_qa,
    validate_plan,
)
from codebuilder.schemas import (
    ArtifactRef,
    Attachment,
    CodebuilderState,
    Plan,
    ProjectArchiveRef,
    QAReport,
)
from codebuilder.tools import attachment_tool, git_tool
from codebuilder.tools.s3_artifacts import SKIP_DIRS, SKIP_FILES, upload_file, upload_workspace


log = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("CODEBUILDER_WORKSPACE_ROOT", "./workspaces")).resolve()
SKILLS_SRC = Path(__file__).parent / "skills"
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _append_note(report: QAReport, note: str) -> None:
    report.integration_notes = " ".join(part for part in (report.integration_notes, note) if part)


def _markdown_excerpt(value: str, limit: int = 6000) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n\n[truncated {omitted} chars]"


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


def _emit_prompt_prepared(state: CodebuilderState, stage: str, prompt: str, **payload: Any) -> None:
    _emit_progress(state, "planner_inputs_prepared", stage=stage, prompt_chars=len(prompt), **payload)


def _emit_usage(state: CodebuilderState, summary: dict) -> None:
    """Surface per-agent-call token/cost to logs + the progress webhook. Fires on
    success and failure so wasted spend on a crashed run is visible."""
    _emit_progress(state, "llm_usage", **summary)


def _run_cost_budget_usd() -> float | None:
    """Cost cap for the build phase (executor + repairs). Unset = no cap."""
    raw = os.environ.get("CODEBUILDER_MAX_RUN_COST_USD")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        log.warning("CODEBUILDER_MAX_RUN_COST_USD=%r is not a number; no cap applied", raw)
        return None
    return value if value > 0 else None


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
    attachments extract to ``inputs/<zip-stem>``, so they resolve here too.
    Returns ``None`` when ``inputs/`` holds no directories at all.
    """
    inputs_dir = Path(workspace_dir) / "inputs"
    if not inputs_dir.is_dir():
        return None
    candidates = sorted((c for c in inputs_dir.iterdir() if c.is_dir()), key=lambda c: c.name)
    if not candidates:
        return None
    repos = [c for c in candidates if c.name.startswith("repo")]
    if repos:
        return str(_descend_wrapper_dirs(repos[0]))
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


def _upload_file_artifacts_enabled(plan: Plan | None) -> bool:
    default = not (plan and plan.mode == "patch_existing")
    return _env_bool("CODEBUILDER_UPLOAD_FILE_ARTIFACTS", default)


def _install_skills(target_dir: Path) -> None:
    """Copy the packaged CC skills into ``<target_dir>/.claude/skills/`` so the
    SDK discovers them from the agent's cwd. The ``.claude`` dir is excluded
    from git diffs, zips, and S3 uploads (see git_tool/s3_artifacts)."""
    if not SKILLS_SRC.is_dir():
        return
    dest = target_dir / ".claude" / "skills"
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for skill_dir in SKILLS_SRC.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith("__"):
                shutil.copytree(skill_dir, dest / skill_dir.name, dirs_exist_ok=True)
    except OSError as exc:  # noqa: BLE001 — skills are best-effort context, never fatal
        log.warning("failed to install CC skills into %s: %s", target_dir, exc)


def _language_hint(state: CodebuilderState) -> str:
    return state.language or "(detect the language from the brief and goals, and write all comments/docstrings in it)"


def _planner_prompt(state: CodebuilderState) -> str:
    records = _format_attachment_records(state.attachment_records)
    prior_history = history.summarize_for_planner(state.project_key) if state.project_key else ""
    goals = "\n".join(f"- {g}" for g in state.goals) or "(none)"
    tech_stack = ", ".join(state.tech_stack) or "(unspecified)"

    sections = [
        "You are the planning agent for CodeBuilder. Produce a plan a human will "
        "review and approve before any code is written. Explore the workspace "
        "with Read/Grep/Glob before planning; attachments (a repo to patch, PDFs, "
        "reference code) are under `inputs/`. Use the `rpa` and `code-review-gate` "
        "skills for standards when they apply.",
        f"## Brief\n{state.brief or '(none)'}",
        f"## Project name\n{state.project_name or '(unspecified)'}",
        f"## Goals\n{goals}",
        f"## Tech stack (hint)\n{tech_stack}",
        f"## Attachments (under inputs/)\n{records}",
        f"## Output language\n{_language_hint(state)}",
    ]
    if prior_history:
        sections.append(f"## Prior runs for this project\n{prior_history}")

    if state.amendments and state.plan is not None:
        sections.append(
            "## Revision requested\n"
            "The human reviewed the plan below and asked for these changes. "
            "Revise ONLY what the feedback touches; keep the rest intact.\n\n"
            f"### Human feedback\n{state.amendments}\n\n"
            f"### Previous plan\n{state.plan.plan_markdown}"
        )

    sections.append(
        "## Instructions\n"
        "- Decide `mode`: `patch_existing` if `inputs/` contains a project to "
        "modify, else `new_project`.\n"
        "- Set `domain` to `rpa` when this is a Python RPA project (so the "
        "executor loads the RPA skill), else leave it empty.\n"
        "- Write the plan in `plan_markdown` as clear Markdown: overview, "
        "approach, the files/structure to create or change, and how it will be "
        "tested. This is shown verbatim to the human and handed to the builder.\n"
        "- Put only genuinely blocking decisions in `open_questions` (max 3, "
        "empty when possible — prefer stating `assumptions` instead).\n"
        "- Do NOT write any files; you are read-only."
    )
    return "\n\n".join(sections)


def _executor_prompt(state: CodebuilderState, plan: Plan, build_dir: str) -> str:
    mode_note = (
        "You are modifying an EXISTING project in place at the current directory. "
        "Make targeted changes; do not rewrite unrelated files."
        if plan.mode == "patch_existing"
        else "You are creating a NEW project in the current directory (empty)."
    )
    installable = (
        "Ensure the project installs cleanly (`uv sync`) and that `ruff check` and "
        "`pytest` pass before you finish."
        if plan.mode == "new_project"
        else "Run the project's tests and `ruff` on what you changed before you finish."
    )
    return "\n\n".join(
        [
            "You are the build agent for CodeBuilder. Implement the approved plan "
            "below in the current working directory. Write complete, working code — "
            "no placeholders, TODOs, or stubbed functions. Use the `rpa` and "
            "`code-review-gate` skills for standards when they apply.",
            mode_note,
            f"## Output language\nWrite all comments and docstrings in: {state.language or 'English'}.",
            f"## Original brief\n{state.brief or '(none)'}",
            f"## Approved plan\n{plan.plan_markdown}",
            f"## Definition of done\n{installable}",
        ]
    )


def _repair_prompt(state: CodebuilderState, plan: Plan, report: QAReport) -> str:
    return "\n\n".join(
        [
            "The project you built failed QA. Fix the failures below in the current "
            "working directory, then re-run the relevant checks (`ruff`, `pytest`, "
            "and `uv sync` for a new project) to confirm they pass.",
            f"## Output language\nWrite all comments and docstrings in: {state.language or 'English'}.",
            f"## QA report\n{qa_report_for_repair(report)}",
            f"## Original plan\n{plan.plan_markdown}",
        ]
    )


def _changelog_prompt(state: CodebuilderState, plan: Plan) -> str:
    lang = state.language or "English"
    return "\n\n".join(
        [
            "The build was stopped because it reached its cost budget. Do NOT "
            "implement, fix, or write any more code — there is no budget left for that.",
            "Your only task: inspect what is already on disk in the current directory "
            f"and write a single file named CHANGELOG.md (in {lang}) with exactly these "
            "sections:\n## What's done\n## What's left\n## What to do on the next run",
            f"## Original plan (for reference)\n{plan.plan_markdown}",
            "Write only CHANGELOG.md, then stop.",
        ]
    )


class CodebuilderFlow(Flow[CodebuilderState]):
    """Single flow: ingest → plan (HITL) → build → finalize."""

    @start()
    def ingest(self):
        # CrewAI Flow auto-merges `inputs={...}` keys into self.state before this
        # method runs. Do NOT pass `id` in inputs — overriding state.id breaks AMP
        # OTel trace correlation (CON-101 / COR-48). Use `session_id` for the
        # caller's identity and let `state.id` stay as the flow's UUID.
        self.state.attachments = [
            a if isinstance(a, Attachment) else Attachment(**a)
            for a in self.state.attachments
        ]

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

        # Skills for the planner (cwd = workspace root).
        _install_skills(workspace_dir)

        project_key = history.project_key_from(self.state)
        if not project_key:
            log.warning(
                "session %s has no project_name and no git attachment; "
                "falling back to session_id for history keying",
                session_key,
            )
            project_key = session_key
        self.state.project_key = project_key

        # NOTE: do not mutate CREWAI_STORAGE_DIR here — HITL resume depends on the
        # default SQLiteFlowPersistence location staying stable.

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
    async def plan(self) -> dict:
        prompt = _planner_prompt(self.state)
        _emit_prompt_prepared(self.state, "plan", prompt)
        plan_obj = validate_plan(
            await cc_agent.run_planner(
                cwd=self.state.workspace_dir,
                prompt=prompt,
                on_usage=lambda s: _emit_usage(self.state, s),
            )
        )
        self.state.plan = plan_obj
        # Resolve the output language: caller override wins, else planner's
        # detection, else English.
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
    async def revise_plan(self, prior) -> dict:
        self.state.amendments = getattr(prior, "feedback", "") or ""
        self.state.amend_cycles += 1
        # revise_plan runs DURING resume, AFTER resume_async cleared the
        # pending-feedback row. It must NEVER raise, or the job becomes
        # unresumable ("No pending feedback found"). On any failure, fall back to
        # the prior plan (annotated) and let @human_feedback re-gate.
        try:
            prompt = _planner_prompt(self.state)
            _emit_prompt_prepared(self.state, "revise_plan", prompt, amend_cycle=self.state.amend_cycles)
            plan_obj = validate_plan(
                await cc_agent.run_planner(
                    cwd=self.state.workspace_dir,
                    prompt=prompt,
                    on_usage=lambda s: _emit_usage(self.state, s),
                )
            )
        except Exception as exc:  # noqa: BLE001 — a revise failure must never brick the job
            fallback = self._prior_plan_snapshot(prior)
            if fallback is None:
                raise
            log.warning("plan revision failed (%s); re-gating with the prior plan", exc)
            fallback.open_questions = [
                f"Automatic plan revision failed ({exc}). The previous plan is shown "
                "unchanged — re-state your changes to try again, or approve to build it as-is.",
                *fallback.open_questions,
            ]
            plan_obj = fallback
        self.state.plan = plan_obj
        self.state.language = self.state.language or plan_obj.language or "English"
        self.state.status = "awaiting_approval"
        return plan_obj.model_dump()

    def _prior_plan_snapshot(self, prior) -> Plan | None:
        """Best-effort recovery of the last-reviewed plan, for re-gating when a
        revision fails."""
        if self.state.plan is not None:
            return self.state.plan.model_copy(deep=True)
        prior_output = getattr(prior, "output", None)
        if isinstance(prior_output, dict):
            try:
                return Plan.model_validate(prior_output)
            except Exception:  # noqa: BLE001
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
    async def build(self, prior):
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
                # Extracted zips aren't git repos; commit a pristine baseline so
                # finalize's git diff captures exactly the repair.
                if not (Path(build_dir) / ".git").exists():
                    git_tool.init_and_commit(build_dir, "codebuilder baseline (pre-patch)")
        else:
            build_dir = str(Path(self.state.workspace_dir) / "output")
            Path(build_dir).mkdir(parents=True, exist_ok=True)
            git_tool.init_and_commit(build_dir)

        self._build_dir = build_dir
        self._build_cost_usd = 0.0
        _install_skills(Path(build_dir))  # skills for the executor (cwd = build_dir)
        budget = _run_cost_budget_usd()
        _emit_progress(
            self.state, "build_started", mode=plan.mode, build_dir=build_dir, cost_budget_usd=budget
        )
        try:
            await cc_agent.run_executor(
                cwd=build_dir,
                prompt=_executor_prompt(self.state, plan, build_dir),
                budget_usd=budget,
                on_usage=self._record_executor_usage,
            )
        except cc_agent.CCBudgetExceeded as exc:
            # Partial files are already on disk; leave the user something + a changelog.
            log.warning("build stopped at cost budget: %s", exc)
            self.state.status = "failed"
            changelog_mode = await self._write_budget_changelog(build_dir, plan, exc.cost_usd)
            self.state.qa_report = QAReport(
                passed=False,
                integration_notes=(
                    f"Build stopped at the cost budget (est. ${exc.cost_usd:.2f} spent). "
                    f"The partial package and CHANGELOG.md ({changelog_mode}) are included — "
                    "re-run the job or raise CODEBUILDER_MAX_RUN_COST_USD to continue."
                ),
            )
            _emit_progress(
                self.state, "build_budget_exceeded", est_cost_usd=exc.cost_usd, changelog=changelog_mode
            )
        except Exception as exc:  # noqa: BLE001 — report a builder crash via QA, don't brick resume
            log.exception("executor agent failed")
            self.state.status = "failed"
            self.state.qa_report = QAReport(
                passed=False,
                integration_notes=f"Executor agent failed: {exc}",
            )
        _emit_progress(self.state, "build_finished", mode=plan.mode)

    @listen(build)
    async def finalize(self, _prior=None):
        build_dir = getattr(self, "_build_dir", None)
        plan = self.state.plan
        build_failed = self.state.status == "failed"

        # Nothing was built (e.g. "no plan to execute") — there's nothing to
        # package. Any real build (even a failed/budget-stopped one) has a
        # build_dir and falls through so its partial work is still delivered.
        if plan is None or build_dir is None:
            try:
                history.record(self.state)
            except Exception as exc:  # noqa: BLE001 — history is observability, never fatal
                log.warning("history.record on build failure failed: %s", exc)
            return self._completion_payload(build_dir or self.state.workspace_dir)

        # Only run (and repair) QA on a healthy build. A crashed/budget-stopped
        # build already has its qa_report from build(); re-running QA would waste
        # time/tokens — we still zip + upload the partial package below.
        if not build_failed:
            _emit_progress(self.state, "final_qa_started")
            self.state.qa_report = self._run_final_qa(build_dir)
            await self._repair_final_qa_if_needed(build_dir)

        if self.state.plan and self.state.plan.mode == "patch_existing":
            try:
                self.state.patch = git_tool.diff(build_dir)
            except Exception as exc:  # noqa: BLE001
                log.warning("patch generation failed: %s", exc)
                self.state.patch = ""

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
            except Exception as exc:  # noqa: BLE001 — archive failure is reported in QA
                log.warning("zip generation failed: %s", exc)
                if self.state.qa_report:
                    self.state.qa_report.passed = False
                    _append_note(self.state.qa_report, f"Project archive generation failed: {exc}")

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

            if self.state.qa_report.passed and _upload_file_artifacts_enabled(self.state.plan):
                try:
                    uploaded_refs.extend(artifact_refs(upload_workspace(build_dir, prefix=prefix)))
                except Exception as exc:  # noqa: BLE001
                    log.warning("workspace artifact upload failed: %s", exc)
            elif self.state.qa_report.passed:
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

        completion = self._completion_payload(build_dir)
        _emit_progress(
            self.state,
            "final_qa_completed",
            **completion,
            passed=bool(self.state.qa_report and self.state.qa_report.passed),
            repair_attempts=self.state.final_qa_repair_attempts,
            integration_notes=self.state.qa_report.integration_notes if self.state.qa_report else "",
        )

        try:
            history.record(self.state)
        except Exception as exc:  # noqa: BLE001 — history is observability, never fatal
            log.warning("history.record on finalize failed: %s", exc)

        return completion

    # --- helpers ---------------------------------------------------------

    def _record_executor_usage(self, summary: dict) -> None:
        """Accumulate authoritative per-call cost toward the build budget, and
        surface it. (Mid-run the cap uses cc_agent's estimate; this is the exact
        total once each call finishes, used to size the remaining repair budget.)"""
        cost = summary.get("cost_usd")
        if isinstance(cost, (int, float)):
            self._build_cost_usd = getattr(self, "_build_cost_usd", 0.0) + float(cost)
        _emit_usage(self.state, summary)

    def _remaining_budget(self) -> float | None:
        budget = _run_cost_budget_usd()
        if budget is None:
            return None
        return max(0.0, budget - getattr(self, "_build_cost_usd", 0.0))

    async def _write_budget_changelog(self, build_dir: str, plan: Plan, est_cost: float) -> str:
        """Leave a CHANGELOG.md when the build stops at budget. `agent` (default)
        asks CC for a bounded wrap-up; `deterministic` builds one from the git
        diff + plan for free; `off` skips. `agent` falls back to deterministic."""
        mode = os.environ.get("CODEBUILDER_BUDGET_CHANGELOG", "agent").strip().lower()
        if mode == "off":
            return "off"
        if mode == "agent":
            try:
                await cc_agent.run_executor(
                    cwd=build_dir,
                    prompt=_changelog_prompt(self.state, plan),
                    effort="low",
                    max_turns=8,
                    on_usage=lambda s: _emit_usage(self.state, s),
                )
                if (Path(build_dir) / "CHANGELOG.md").is_file():
                    return "agent"
                log.warning("agent changelog produced no CHANGELOG.md; using deterministic")
            except Exception as exc:  # noqa: BLE001 — never fail the wrap-up
                log.warning("agent changelog failed (%s); using deterministic", exc)
        self._write_deterministic_changelog(build_dir, plan, est_cost)
        return "deterministic"

    def _write_deterministic_changelog(self, build_dir: str, plan: Plan, est_cost: float) -> None:
        try:
            changed = git_tool.changed_files(build_dir) or []
        except Exception:  # noqa: BLE001
            changed = []
        files = "\n".join(f"- {p}" for p in changed) or "- (no tracked changes detected)"
        body = (
            "# CodeBuilder — build stopped at cost budget\n\n"
            "The build stopped because it reached the configured cost budget "
            f"(`CODEBUILDER_MAX_RUN_COST_USD`). Estimated spend: ${est_cost:.2f}.\n\n"
            "## Files created / modified so far\n"
            f"{files}\n\n"
            "## Original plan (full scope)\n\n"
            f"{plan.plan_markdown}\n\n"
            "## Next run\n"
            "Re-run the job (or raise `CODEBUILDER_MAX_RUN_COST_USD`) to continue. "
            "The files above are what got done; the plan above is the full scope.\n"
        )
        try:
            (Path(build_dir) / "CHANGELOG.md").write_text(body, encoding="utf-8")
        except OSError as exc:
            log.warning("failed to write deterministic CHANGELOG.md: %s", exc)

    def _run_final_qa(self, build_dir: str) -> QAReport:
        plan = self.state.plan
        is_patch = bool(plan and plan.mode == "patch_existing")
        # Patch jobs: scope ruff to the files the executor touched so pre-existing
        # lint debt in the customer's untouched files can't fail QA.
        changed_paths = git_tool.changed_files(build_dir) if is_patch else None
        return run_final_qa(
            build_dir,
            changed_paths=changed_paths,
            require_installable=bool(plan and plan.mode == "new_project"),
            allow_no_tests=bool(is_patch and not has_pytest_files(build_dir)),
            run_tests=_env_bool("CODEBUILDER_RUN_TESTS", True),
        )

    def _max_final_qa_repairs(self) -> int:
        return _env_int("CODEBUILDER_MAX_FINAL_QA_REPAIRS", DEFAULT_MAX_FINAL_QA_REPAIRS)

    async def _repair_final_qa_if_needed(self, build_dir: str) -> None:
        attempts = self._max_final_qa_repairs()
        plan = self.state.plan
        if plan is None:
            return
        for attempt in range(1, attempts + 1):
            if self.state.qa_report is None or self.state.qa_report.passed:
                return
            remaining = self._remaining_budget()
            if remaining is not None and remaining <= 0:
                _append_note(
                    self.state.qa_report,
                    "Skipped QA repair: cost budget already reached.",
                )
                return
            self.state.final_qa_repair_attempts = attempt
            _emit_progress(self.state, "final_qa_repair_started", attempt=attempt, max_attempts=attempts)
            try:
                await cc_agent.run_executor(
                    cwd=build_dir,
                    prompt=_repair_prompt(self.state, plan, self.state.qa_report),
                    budget_usd=remaining,
                    on_usage=self._record_executor_usage,
                )
            except cc_agent.CCBudgetExceeded as exc:
                log.warning("QA repair stopped at cost budget: %s", exc)
                _append_note(
                    self.state.qa_report,
                    f"QA repair stopped at the cost budget (est. ${exc.cost_usd:.2f} spent this repair).",
                )
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("final QA repair attempt %s failed: %s", attempt, exc)
                _append_note(self.state.qa_report, f"Repair attempt {attempt} errored: {exc}")
                return
            self.state.qa_report = self._run_final_qa(build_dir)
        if self.state.qa_report and not self.state.qa_report.passed:
            _append_note(
                self.state.qa_report,
                f"Final QA still failing after {attempts} repair attempt(s).",
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
            payload["qa_report_markdown"] = self._qa_report_markdown(build_dir)
        if self.state.zip_path:
            payload["zip_path"] = self.state.zip_path
        if self.state.zip_url:
            payload["zip_url"] = self.state.zip_url
        if self.state.project_archive:
            payload["project_archive"] = self.state.project_archive.model_dump(mode="json")
        if self.state.patch:
            payload["patch"] = self.state.patch
        return payload

    def _qa_report_markdown(self, build_dir: str | None = None) -> str:
        report = self.state.qa_report
        if report is None:
            return ""
        status = "passed" if report.passed else "failed"
        lines = [
            "# CodeBuilder QA Report",
            "",
            f"- Status: {status}",
            f"- Project: {self.state.project_name or self.state.id}",
            f"- Build dir: {build_dir or self.state.workspace_dir}",
            f"- Repair attempts: {self.state.final_qa_repair_attempts}",
        ]
        if self.state.project_archive:
            lines.append(f"- Archive: {self.state.project_archive.local_path}")
            if self.state.project_archive.url:
                lines.append(f"- Download: {self.state.project_archive.url}")
        sections = [
            ("Integration Notes", report.integration_notes),
            ("Lint Output", report.lint_output),
            ("Test Output", report.test_output),
        ]
        for title, value in sections:
            if not value:
                continue
            lines.extend(["", f"## {title}", "", "```text", _markdown_excerpt(value), "```"])
        if not report.passed:
            lines.extend(
                [
                    "",
                    "## Suggested next request",
                    "",
                    "Please fix the QA failures in this package. Use the lint and test "
                    "output above as the source of truth, then rerun the relevant tests.",
                ]
            )
        return "\n".join(lines) + "\n"


def kickoff():
    """Local smoke test with hardcoded inputs (a new_project job)."""
    logging.basicConfig(level=logging.INFO)
    CodebuilderFlow().kickoff(
        inputs={
            "session_id": "local-test",
            "project_name": "hello-cli",
            "brief": "Build a tiny Python CLI that prints a friendly greeting and has a pytest test.",
            "goals": ["A working `hello` command", "One passing test"],
            "tech_stack": ["python"],
            "attachments": [],
        }
    )


def resume(job_id: str, feedback: str = ""):
    """Resume a paused job with human feedback (approve / amend text / reject)."""
    return CodebuilderFlow.from_pending(job_id).resume(feedback)


def plot():
    CodebuilderFlow().plot("codebuilder_flow")


if __name__ == "__main__":
    kickoff()
