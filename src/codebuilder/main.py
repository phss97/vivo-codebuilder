"""Codebuilder Flow — plans, gates on human approval, then builds.

The planning and building are done by Claude Agent SDK agents (see cc_agent):
an Opus planner and a Sonnet executor. The CrewAI Flow shell is kept only for
what AMP + the frontend depend on — kickoff, the @human_feedback HITL gate,
progress/completion webhooks, S3 upload, and per-project history.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from crewai.flow import Flow, listen, or_, router, start
from crewai.flow.human_feedback import human_feedback

from codebuilder import cc_agent, history, package_workspace
from codebuilder.runtime_qa import (
    artifact_refs,
    check_declared_tests,
    check_preserved_dependencies,
    check_spec_contract,
    plan_spec_hash,
    project_dependency_names,
    qa_report_for_prompt,
    qa_report_for_repair,
    run_final_qa,
    run_verification_commands,
    validate_plan,
)
from codebuilder.schemas import (
    ArtifactRef,
    Attachment,
    CommandResult,
    CodebuilderState,
    IdentifierContract,
    IntakeAssessment,
    IntakeQuestion,
    PackageResult,
    Plan,
    ProductionReview,
    ProjectArchiveRef,
    QAIssue,
    QAReport,
    QuarantineArchiveRef,
    QuarantineReport,
    WorkPackageSpec,
)
from codebuilder.tools import attachment_tool, git_tool
from codebuilder.tools.lint_runner_tool import apply_ruff_fixes
from codebuilder.tools.s3_artifacts import (
    SKIP_DIRS,
    SKIP_FILES,
    upload_file,
    upload_workspace,
)


log = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(
    os.environ.get("CODEBUILDER_WORKSPACE_ROOT", "./workspaces")
).resolve()
SKILLS_SRC = Path(__file__).parent / "skills"
DEFAULT_MAX_FINAL_QA_REPAIRS = 3
MAX_QA_BUDGET_RESERVE_USD = 10.0
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
    report.integration_notes = " ".join(
        part for part in (report.integration_notes, note) if part
    )


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
        resp = requests.post(
            webhook,
            json=body,
            headers=headers,
            timeout=PROGRESS_WEBHOOK_TIMEOUT_SECONDS,
        )
        if resp.status_code >= 400:
            log.warning(
                "progress webhook POST for %s returned %s", event_type, resp.status_code
            )
    except requests.RequestException as exc:
        log.warning("progress webhook POST failed for %s: %s", event_type, exc)


def _emit_prompt_prepared(
    state: CodebuilderState, stage: str, prompt: str, **payload: Any
) -> None:
    _emit_progress(
        state,
        "planner_inputs_prepared",
        stage=stage,
        prompt_chars=len(prompt),
        **payload,
    )


def _emit_usage(state: CodebuilderState, summary: dict) -> None:
    """Surface per-agent-call token/cost to logs + the progress webhook. Fires on
    success and failure so wasted spend on a crashed run is visible."""
    state.llm_usage.append(summary)
    _emit_progress(state, "llm_usage", **summary)


def _run_cost_budget_usd() -> float | None:
    """Cost cap for the build phase (executor + review + repairs). Unset = no cap."""
    raw = os.environ.get("CODEBUILDER_MAX_RUN_COST_USD")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        log.warning(
            "CODEBUILDER_MAX_RUN_COST_USD=%r is not a number; no cap applied", raw
        )
        return None
    return value if value > 0 else None


def _initial_build_budget_usd(total_budget: float | None) -> float | None:
    if total_budget is None:
        return None
    reserve = min(MAX_QA_BUDGET_RESERVE_USD, total_budget / 2)
    return total_budget - reserve


def _zip_build(
    build_dir: str,
    out_dir: Path,
    project_name: str,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Zip the built project into ``out_dir/<project>.zip``. Overwrites if present."""
    src = Path(build_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_zip_stem(project_name)}.zip"
    if out_path.exists():
        out_path.unlink()

    arcroot = out_path.stem  # wrap contents under a top-level folder in the archive
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if path.resolve() == out_path.resolve():
                continue
            if path.name in SKIP_FILES:
                continue
            rel = path.relative_to(src)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            zf.write(path, arcname=f"{arcroot}/{rel.as_posix()}")
        for name, value in (metadata or {}).items():
            zf.writestr(f"{arcroot}/.codebuilder/{name}", value)
    return out_path


def _ordered_packages(plan: Plan) -> list[WorkPackageSpec]:
    """Return the validated package DAG in a stable topological order."""
    remaining = {package.id: set(package.depends_on) for package in plan.work_packages}
    by_id = {package.id: package for package in plan.work_packages}
    ordered: list[WorkPackageSpec] = []
    while remaining:
        ready = [
            package.id
            for package in plan.work_packages
            if package.id in remaining and not remaining[package.id]
        ]
        if not ready:
            raise ValueError("work package dependency cycle")
        for package_id in ready:
            ordered.append(by_id[package_id])
            remaining.pop(package_id)
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return ordered


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
    candidates = sorted(
        (c for c in inputs_dir.iterdir() if c.is_dir()), key=lambda c: c.name
    )
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
    return (
        state.language
        or "(detect the language from the brief and goals, and write all comments/docstrings in it)"
    )


def _looks_like_rpa(state: CodebuilderState, project_root: str | None = None) -> bool:
    request_text = " ".join(
        [state.brief, state.project_name, *state.goals, *state.tech_stack]
    ).lower()
    if "rpa" in request_text or "robotic process automation" in request_text:
        return True
    if not project_root:
        return False
    root = Path(project_root)
    pyproject = root / "pyproject.toml"
    try:
        if (
            pyproject.is_file()
            and "pyinstaller" in pyproject.read_text(encoding="utf-8").lower()
        ):
            return True
    except OSError:
        pass
    source_root = root / "src" if (root / "src").is_dir() else root
    names = {
        path.name for path in source_root.rglob("*.py") if ".venv" not in path.parts
    }
    return {"producer.py", "consumer.py", "orchestrator.py"}.issubset(names)


def _build_effort(state: CodebuilderState, build_dir: str) -> cc_agent.Effort:
    plan = state.plan
    failed_preflight = bool(
        state.preflight_qa_report and not state.preflight_qa_report.passed
    )
    if (
        (plan and plan.domain.lower() == "rpa")
        or _looks_like_rpa(state, build_dir)
        or failed_preflight
    ):
        return "high"
    return cc_agent.EXECUTOR_EFFORT


def _feedback_text(prior: Any) -> str:
    return str(getattr(prior, "feedback", "") or "").strip()


def _canonical_spec_json(plan: Plan) -> str:
    return json.dumps(
        plan.model_dump(mode="json", exclude={"plan_markdown"}),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )


def _resolve_authoritative_asset(
    workspace_dir: str, asset_path: str, mode: str
) -> Path | None:
    """Resolve one exact relative asset without following job-created symlinks."""
    value = asset_path.strip()
    relative = PurePosixPath(value)
    if (
        not value
        or value != relative.as_posix()
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None
    inputs_path = Path(workspace_dir) / "inputs"
    if inputs_path.is_symlink():
        return None
    if mode == "patch_existing":
        patch_root = _resolve_patch_root(workspace_dir)
        if patch_root is None:
            return None
        raw_root = Path(patch_root)
        try:
            workspace_prefix = raw_root.relative_to(Path(workspace_dir)).parts
        except ValueError:
            return None
        if relative.parts[: len(workspace_prefix)] == workspace_prefix:
            relative = PurePosixPath(*relative.parts[len(workspace_prefix) :])
            if not relative.parts:
                return None
    else:
        raw_root = inputs_path
    current = raw_root
    while current != inputs_path:
        if current.is_symlink() or current.parent == current:
            return None
        current = current.parent
    if raw_root.is_symlink():
        return None
    try:
        inputs_root = inputs_path.resolve()
        root = raw_root.resolve()
    except (OSError, RuntimeError):
        return None
    if root != inputs_root and inputs_root not in root.parents:
        return None
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return None
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return (
        resolved
        if resolved.is_file() and (resolved == root or root in resolved.parents)
        else None
    )


def _validated_workspace_tree(workspace_dir: str, value: str) -> Path | None:
    """Return a non-root directory held inside this job workspace."""
    if not value:
        return None
    root = Path(workspace_dir).resolve()
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return None
    return resolved if relative.parts and resolved.is_dir() else None


def _bind_authoritative_asset_hashes(
    plan: Plan,
    workspace_dir: str,
    intake: IntakeAssessment | None = None,
) -> None:
    """Bind real asset hashes before approval, while the human can review them."""
    raw_root = (
        _resolve_patch_root(workspace_dir)
        if plan.mode == "patch_existing"
        else str(Path(workspace_dir) / "inputs")
    )
    if raw_root is not None:
        root = Path(raw_root).resolve()
        assets = [
            *(intake.authoritative_assets if intake else []),
            *plan.authoritative_assets,
        ]
        for asset in assets:
            candidate = _resolve_authoritative_asset(
                workspace_dir, asset.path, plan.mode
            )
            if candidate is not None:
                asset.path = candidate.relative_to(root).as_posix()
    required = {asset.path for asset in (intake.authoritative_assets if intake else [])}
    declared = {asset.path for asset in plan.authoritative_assets}
    if missing := sorted(required - declared):
        raise ValueError(
            "Plan omitted authoritative intake assets: " + ", ".join(missing)
        )
    for asset in plan.authoritative_assets:
        if not asset.immutable:
            continue
        candidate = _resolve_authoritative_asset(workspace_dir, asset.path, plan.mode)
        if candidate is None:
            raise ValueError(
                f"Authoritative asset could not be resolved before approval: {asset.path}"
            )
        asset.sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()


def _intake_prompt(state: CodebuilderState) -> str:
    answers = "\n\n".join(state.intake_answers) or "(none)"
    readiness = (
        "- Set ready=true only when scope, authoritative assets, exact machine "
        "identifiers, and credible test/build/typecheck commands are known.\n"
        "- For an unfamiliar stack, ask the user for exact verification commands; "
        "never silently downgrade to semantic-only QA.\n"
        if state.attachments
        else "- Set ready=true once the requested scope and stack are clear. This is "
        "a new project: the planner defines the verification commands, so list any "
        "you would expect in missing_verification_commands but never block on them.\n"
    )
    return "\n\n".join(
        [
            "You are the read-only intake analyst for CodeBuilder. Inventory the "
            "entire repository tree and inspect the source, tests, schemas, entry "
            "points, build configuration, and attached documentation that constrain "
            "the requested change. Decide whether a planner can write an executable "
            "specification. Do not write a plan and do not modify files.",
            f"## Brief\n{state.brief or '(none)'}",
            "## Goals\n" + ("\n".join(f"- {goal}" for goal in state.goals) or "(none)"),
            f"## Stack hints\n{', '.join(state.tech_stack) or '(unspecified)'}",
            "## Materialized attachments\n"
            + _format_attachment_records(state.attachment_records),
            "## Existing deterministic preflight\n"
            + qa_report_for_prompt(state.preflight_qa_report),
            f"## Earlier clarification rounds (already resolved — do not re-ask)\n{answers}",
            "## Decision contract\n"
            + readiness
            + "- Ask at most three genuinely blocking questions. Prefer explicit "
            "assumptions for non-blockers.\n"
            "- List an authoritative_asset only when that file must be delivered "
            "byte-for-byte unchanged; list reference documentation only under "
            "evidence_inspected.\n"
            "- For patch jobs, record authoritative_asset paths relative to the "
            "attached project root, without an inputs/<project>/ prefix.\n"
            "- Existing package/module/symbol/database/API/environment names are "
            "authoritative and must be recorded verbatim, never translated.",
        ]
    )


def _planner_prompt(state: CodebuilderState) -> str:
    records = _format_attachment_records(state.attachment_records)
    prior_history = (
        history.summarize_for_planner(state.project_key) if state.project_key else ""
    )
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
    if state.preflight_qa_report is not None:
        sections.append(
            "## Deterministic preflight QA for the attached project\n"
            "These failures are diagnostic, not a reason to stop planning. Plan concrete "
            "fixes for them alongside the requested work.\n\n"
            f"{qa_report_for_prompt(state.preflight_qa_report)}"
        )
    if state.baseline_dependencies:
        sections.append(
            "## Existing dependency baseline\n"
            "This is a patch job. Preserve every dependency name below; an import that "
            "is absent from `src/` may still be required by tests or build tooling.\n\n"
            + "\n".join(f"- `{name}`" for name in state.baseline_dependencies)
        )
    if state.intake_assessment is not None:
        sections.append(
            "## Approved intake assessment\n"
            + json.dumps(
                state.intake_assessment.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
        )

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
        "- Produce a revisioned structured specification. `plan_markdown` is a "
        "derived compatibility view: leave it empty because CodeBuilder renders it "
        "from the structured fields.\n"
        "- Set one exact English-ASCII `package_name` for new projects. Preserve "
        "existing package/module/symbol names byte-for-byte for patch jobs.\n"
        "- For patch jobs, make every `authoritative_assets.path` relative to the "
        "attached project root; never prefix it with `inputs/<project>/`.\n"
        "- Break the work into dependency-ordered `work_packages`. Each package must "
        "state what to build, expected behavior, stable success-criterion IDs, exact "
        "test cases covering every criterion, and every owned file exactly once.\n"
        "- Every Python file must declare its binding `public_api` names/signatures, "
        "including an explicitly empty list when it exports nothing. Never substitute "
        "synonyms or translations such as create/build.\n"
        "- Put exact shell-free argv arrays for required lint/typecheck/test/build "
        "commands in `verification_commands`; include at least one required test. "
        "Keep network=false unless that command explicitly requires approved network "
        "access; the human will review this capability. Only category=build may write "
        "to its disposable verification copy.\n"
        "- When the plan declares or depends on a DDL/schema/migration file (`*.sql` "
        "or under `migrations/`), one test case must assert that the model/ORM field "
        "names match that schema column-for-column, with its `verifies_schema` set to "
        "that exact path. This is what catches a translated `nome` living beside a "
        "correct `job_name`.\n"
        "- Use `terminology` only for repeated human-facing prose. Machine identifiers, "
        "external literals, database fields, environment variables, and entry points "
        "belong in `identifier_contract` and are never translated.\n"
        "- For an existing RPA project, trace the real production path from the "
        "entry point through Settings, dependency composition, external adapters, "
        "and login/connect cleanup. Include a `Canonical contracts` section that "
        "selects one coherent contract for entity IDs/statuses, DTO fields, Protocol "
        "and repository/adapter signatures, Settings/environment names, and the "
        "owner of each external-resource lifecycle. Base it on the current README, "
        "database scripts, production path, and tested behavior. Do not trust tests "
        "that replace the complete production adapter or reproduce a different fake "
        "contract.\n"
        "- Put only genuinely blocking decisions in `open_questions` (max 3, "
        "empty when possible — prefer stating `assumptions` instead).\n"
        "- Do NOT write any files; you are read-only."
    )
    return "\n\n".join(sections)


def _test_author_prompt(
    state: CodebuilderState, plan: Plan, package: WorkPackageSpec, guidance: str = ""
) -> str:
    parts = [
        "Write the approved tests for exactly one work package. Tests are the frozen "
        "acceptance evidence for the next agent, so do not implement production code "
        "and do not weaken an assertion merely to make the current tree pass.",
        f"## Work package\n{json.dumps(package.model_dump(mode='json'), indent=2, ensure_ascii=False)}",
        f"## Approved spec hash\n{state.approved_spec_hash}",
        f"## Canonical identifier and terminology contract\n{_canonical_spec_json(plan)}",
    ]
    if guidance:
        parts.append(f"## Human retry guidance\n{guidance}")
    parts.append(
        "## Completion contract\nCreate every declared test path and exact test name, "
        "cover every mapped criterion, use canonical identifiers verbatim, run the "
        "most focused available checks, and stop if the approved contract is ambiguous."
    )
    return "\n\n".join(parts)


def _package_executor_prompt(
    state: CodebuilderState,
    plan: Plan,
    package: WorkPackageSpec,
    *,
    repair_report: QAReport | None = None,
    guidance: str = "",
) -> str:
    production_files = [file.path for file in package.files if file.kind != "test"]
    parts = [
        "Implement exactly one approved work package in the current staging tree. "
        "The tests and specification are immutable. Modify only the production/config/"
        "documentation paths listed below; do not touch any test, unrelated file, "
        "quality threshold, or authoritative asset.",
        "## Allowed implementation paths\n"
        + ("\n".join(f"- `{path}`" for path in production_files) or "(none)"),
        f"## Work package\n{json.dumps(package.model_dump(mode='json'), indent=2, ensure_ascii=False)}",
        f"## Approved spec hash\n{state.approved_spec_hash}",
        f"## Exact identifier contract\n{json.dumps(plan.identifier_contract.model_dump(mode='json'), indent=2, ensure_ascii=False)}",
        f"## Human-facing terminology registry\n{json.dumps([entry.model_dump(mode='json') for entry in plan.terminology], indent=2, ensure_ascii=False)}",
        f"## Output language\nHuman-facing prose/comments/docstrings: {state.language or 'English'}. Code identifiers remain exact English ASCII or preserved source names.",
    ]
    if repair_report is not None:
        parts.append(
            "## Current QA issues to repair\n" + qa_report_for_repair(repair_report)
        )
    if guidance:
        parts.append(f"## Human guidance\n{guidance}")
    parts.append(
        "## Definition of done\nFix the shared root cause across every caller inside "
        "the allowed paths. Run focused checks, but never edit tests/specs to obtain "
        "green output. QA will rerun the complete approved command set after this call."
    )
    return "\n\n".join(parts)


def _semantic_review_prompt(
    state: CodebuilderState,
    plan: Plan,
    report: QAReport,
    package: WorkPackageSpec | None,
) -> str:
    scope = (
        package.model_dump(mode="json")
        if package is not None
        else {
            "id": "__final__",
            "success_criteria": [
                criterion.model_dump(mode="json")
                for item in plan.work_packages
                for criterion in item.success_criteria
            ],
        }
    )
    return "\n\n".join(
        [
            "Perform a read-only semantic acceptance review after deterministic "
            "commands passed. The current source and tests are the only runtime "
            "evidence. The approved spec defines intent but is not proof that it was "
            "implemented. Report only blockers; never edit or override command failures.",
            f"## Review scope\n{json.dumps(scope, indent=2, ensure_ascii=False)}",
            f"## Approved spec hash\n{state.approved_spec_hash}",
            f"## Full approved specification\n{_canonical_spec_json(plan)}",
            f"## Deterministic evidence\n{qa_report_for_prompt(report)}",
            "## Required decision\nFor each blocker, cite current file/symbol evidence, "
            "the affected criterion IDs, classify the owner as code/test/spec/environment, "
            "and give a concrete repair instruction. Return no issues when passed=true.",
        ]
    )


def _executor_prompt(state: CodebuilderState, plan: Plan) -> str:
    mode_note = (
        "You are modifying an EXISTING project in place at the current directory. "
        "Make targeted changes; do not rewrite unrelated files."
        if plan.mode == "patch_existing"
        else "You are creating a NEW project in the current directory (empty)."
    )
    sections = [
        "You are the build agent for CodeBuilder. Implement the approved plan "
        "below in the current working directory. Write complete, working code — "
        "no placeholders, TODOs, or stubbed functions. Use the `rpa` and "
        "`code-review-gate` skills for standards when they apply.",
        mode_note,
        f"## Output language\nWrite all comments and docstrings in: {state.language or 'English'}.",
        f"## Original brief\n{state.brief or '(none)'}",
        f"## Approved plan\n{plan.plan_markdown}",
    ]
    if state.preflight_qa_report is not None:
        sections.append(
            "## Preflight failures to fix\n"
            f"{qa_report_for_prompt(state.preflight_qa_report)}"
        )
    if plan.mode == "patch_existing":
        sections.append(
            "## Repair strategy\n"
            "Establish the canonical contracts before editing. Repair one root-cause "
            "cluster at a time, update every producer and consumer of that contract, "
            "then run targeted MyPy and tests for the cluster before continuing. "
            "Existing tests are acceptance evidence, but when a test contradicts the "
            "README, database scripts, and selected production contract, update the "
            "test and every caller together. For full-package recovery, run Ruff safe "
            "fixes and formatting before semantic repairs."
        )
        if state.baseline_dependencies:
            sections.append(
                "## Dependencies that must remain declared\n"
                + "\n".join(f"- `{name}`" for name in state.baseline_dependencies)
            )
    sections.append(
        "## Definition of done\nRun the complete package checks before finishing: "
        "`uv sync --locked`, `ruff check .`, `ruff format --check .`, native "
        "`mypy`, configuration/dependency/entry-point validation, and the full "
        "`pytest` suite. Fix failures across the repository, including existing "
        "debt that prevents the delivered package from passing. For RPA projects, "
        "exercise the real Settings, composition root, adapter contracts, and "
        "login/connect cleanup while mocking only external transports. Keep README "
        "environment examples aligned with .env.example, and make missing required "
        "configuration fail with actionable setup guidance instead of unsafe defaults. "
        "Do not remove or weaken tests, lower coverage thresholds, relax Ruff/MyPy, "
        "or add fake configuration or credential defaults."
    )
    return "\n\n".join(sections)


def _repair_prompt(
    state: CodebuilderState,
    plan: Plan,
    report: QAReport,
    attempt: int,
    max_attempts: int,
) -> str:
    return "\n\n".join(
        [
            f"QA repair attempt {attempt}/{max_attempts}. Fix every current failure "
            "in the working directory. Cluster related errors, inspect every caller "
            "of a contract before changing it, and fix the shared root cause once. "
            "Do not weaken tests, typing, lint configuration, or acceptance criteria. "
            "Run targeted failing checks while repairing, then re-run `uv sync --locked`, "
            "`ruff check .`, `ruff format --check .`, native `mypy`, configuration/runtime "
            "contract checks, and the full `pytest` suite. Do not finish while any "
            "required check is still failing.",
            f"## Output language\nWrite all comments and docstrings in: {state.language or 'English'}.",
            f"## QA report\n{qa_report_for_repair(report)}",
            f"## Original plan\n{plan.plan_markdown}",
        ]
    )


def _production_review_prompt(state: CodebuilderState) -> str:
    return "\n\n".join(
        [
            "You are the final production-wiring reviewer for an RPA package. "
            "Read the implementation and return only blockers that could make the "
            "installed package fail in production despite green lint, MyPy, and tests. "
            "Do not report style preferences or unavailable customer infrastructure.",
            "## Evidence boundary\n"
            "The current files in the working directory are the only source of truth. "
            "Re-read the affected file before reporting an issue. Do not use the "
            "approved plan, CODEBUILDER_REPORT.md, prior QA reports, previous reviews, "
            "or comments describing old defects as evidence. Deterministic QA already "
            "passed this current tree; treat that as context, not proof.",
            "## Required trace\n"
            "- Follow every console/module entry point through Settings and the "
            "dependency-composition root into external adapters.\n"
            "- Verify every settings attribute exists and every injected dependency "
            "uses its real Protocol instead of Any/getattr.\n"
            "- Verify secrets are fetched through the declared secret-provider API.\n"
            "- Verify login/connect and logout/disconnect lifecycle is owned by the "
            "orchestrator and cleanup runs on success and failure.\n"
            "- Verify tests exercise those real production classes while replacing "
            "only COM, HTTP, database, filesystem, or other external transports.\n"
            "- Verify settings tests cannot accidentally read the developer's .env.",
            "## Decision rule\n"
            "Set passed=false only for concrete execution blockers visible in the "
            "current source. Every issue must name the current file and symbol or line, "
            "state what the code currently does, and identify the broken runtime "
            "contract. Return an empty issues list when passed=true.",
            f"## Original brief\n{state.brief or '(none)'}",
            f"## Output language\nWrite issues in: {state.language or 'English'}.",
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

        patch_root = _resolve_patch_root(self.state.workspace_dir)
        if patch_root is not None:
            self.state.baseline_dependencies = project_dependency_names(patch_root)
            _emit_progress(self.state, "preflight_qa_started", build_dir=patch_root)
            try:
                self.state.preflight_qa_report = run_final_qa(
                    patch_root,
                    require_installable=(Path(patch_root) / "pyproject.toml").is_file(),
                    require_typecheck=_looks_like_rpa(self.state, patch_root),
                    locked_sync=True,
                )
            except Exception as exc:  # noqa: BLE001 — preflight is diagnostic
                log.exception("preflight QA failed unexpectedly")
                self.state.preflight_qa_report = QAReport(
                    passed=False,
                    integration_notes=f"Preflight QA could not complete: {exc}",
                )
            _emit_progress(
                self.state,
                "preflight_qa_completed",
                passed=self.state.preflight_qa_report.passed,
            )

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
    async def assess_intake(self) -> dict:
        self.state.phase = "intake"
        prompt = _intake_prompt(self.state)
        _emit_prompt_prepared(self.state, "intake", prompt)
        assessment = await cc_agent.run_intake(
            cwd=self.state.workspace_dir,
            prompt=prompt,
            on_usage=lambda summary: _emit_usage(self.state, summary),
        )
        self._apply_intake_gate(assessment)
        self.state.intake_assessment = assessment
        return assessment.model_dump(mode="json")

    def _apply_intake_gate(self, assessment: IntakeAssessment) -> None:
        """Only a patch job can be blocked on unknown verification commands — for a
        new project the planner invents them, so the question has no answer."""
        if assessment.blocking_questions or (
            self.state.attachments and assessment.missing_verification_commands
        ):
            assessment.ready = False

    @router(
        assess_intake,
        emit=["intake_ready", "intake_needs_input"],
    )
    def route_intake(self) -> str:
        assessment = self.state.intake_assessment
        return (
            "intake_ready"
            if assessment is not None and assessment.ready
            else "intake_needs_input"
        )

    @listen("intake_needs_input")
    @human_feedback(
        message="Answer the blocking intake questions so CodeBuilder can produce an executable specification, or reject to cancel.",
        emit=["intake_answered", "job_rejected"],
        llm=GUARDRAIL_LLM,
        default_outcome="intake_answered",
    )
    def request_intake_feedback(self) -> dict:
        self.state.phase = "awaiting_intake"
        self.state.status = "awaiting_approval"
        assessment = self.state.intake_assessment or IntakeAssessment(
            ready=False,
            blocking_questions=[
                IntakeQuestion(
                    id="intake_unavailable",
                    question="Please restate the requested scope and verification commands.",
                )
            ],
        )
        return {
            "phase": "intake",
            "assessment": assessment.model_dump(mode="json"),
            "questions": [
                question.model_dump(mode="json")
                for question in assessment.blocking_questions
            ],
        }

    @listen("intake_answered")
    async def reassess_intake(self, prior) -> dict:
        feedback = _feedback_text(prior)
        if feedback:
            # Pair the answer with what was asked; a bare answer list left the next
            # round unable to tell what had already been resolved, so it re-asked.
            asked = "\n".join(
                f"Q: {question.question}"
                for question in (
                    self.state.intake_assessment.blocking_questions
                    if self.state.intake_assessment
                    else []
                )
            )
            self.state.intake_answers.append(
                f"{asked}\nA: {feedback}" if asked else f"A: {feedback}"
            )
        self.state.phase = "intake"
        try:
            prompt = _intake_prompt(self.state)
            _emit_prompt_prepared(self.state, "reassess_intake", prompt)
            assessment = await cc_agent.run_intake(
                cwd=self.state.workspace_dir,
                prompt=prompt,
                on_usage=lambda summary: _emit_usage(self.state, summary),
            )
            self._apply_intake_gate(assessment)
        except Exception as exc:  # noqa: BLE001 — resumed HITL methods must re-gate
            log.warning("intake reassessment failed; re-gating: %s", exc)
            assessment = IntakeAssessment(
                ready=False,
                understood_scope=(
                    self.state.intake_assessment.understood_scope
                    if self.state.intake_assessment
                    else ""
                ),
                blocking_questions=[
                    IntakeQuestion(
                        id="intake_retry",
                        question=(
                            "The automatic reassessment failed. Please restate the "
                            "missing information or reject the request."
                        ),
                        rationale=str(exc),
                    )
                ],
            )
        self.state.intake_assessment = assessment
        return assessment.model_dump(mode="json")

    @router(
        reassess_intake,
        emit=["intake_ready", "intake_needs_input"],
    )
    def route_reassessed_intake(self) -> str:
        assessment = self.state.intake_assessment
        return (
            "intake_ready"
            if assessment is not None and assessment.ready
            else "intake_needs_input"
        )

    @listen("intake_ready")
    @human_feedback(
        message="Review the generated plan. Reply 'approve' to start coding, describe changes to amend, or 'reject' to cancel.",
        # emit[0] is CrewAI's fallback whenever the classifier LLM fails (outage,
        # bad JSON, missing key) — default_outcome only covers empty feedback. So
        # the safe outcome must be first, or an outage silently approves the spec.
        emit=["spec_amend", "spec_approved", "job_rejected"],
        llm=GUARDRAIL_LLM,
        default_outcome="spec_amend",
    )
    async def plan(self) -> dict:
        self.state.phase = "specification"
        # plan() runs DURING resume for any job that passed an intake gate, after
        # resume_async cleared the pending-feedback row. It must NEVER raise.
        try:
            plan_obj = await self._plan_with_repair(_planner_prompt(self.state), "plan")
        except Exception as exc:  # noqa: BLE001 — see above
            return self._degraded_plan_gate(exc)
        plan_obj.revision = 1
        plan_obj.plan_markdown = plan_obj.render_markdown()
        self.state.plan = plan_obj
        # Resolve the output language: caller override wins, else planner's
        # detection, else English.
        self.state.language = self.state.language or plan_obj.language or "English"
        self.state.status = "awaiting_approval"
        return {
            "phase": "specification",
            "plan": plan_obj.model_dump(mode="json"),
            **plan_obj.model_dump(mode="json"),
        }

    async def _plan_with_repair(
        self,
        prompt: str,
        label: str,
        *,
        require_structured: bool = True,
        **emit_payload: Any,
    ) -> Plan:
        """Run the planner, feeding validate_plan's own rejection text back for a retry.

        validate_plan is strict and all-or-nothing, so without this one missed rule
        the planner could have fixed itself discards the whole run.
        """
        attempts = max(
            1, int(os.environ.get("CODEBUILDER_PLANNER_REPAIR_ATTEMPTS") or "2")
        )
        rejection: ValueError | None = None
        for attempt in range(1, attempts + 1):
            attempt_prompt = (
                prompt
                if rejection is None
                else (
                    f"{prompt}\n\n## Rejected specification\n"
                    f"Your previous plan was rejected: {rejection}\n"
                    "Fix exactly these problems and return the whole corrected plan."
                )
            )
            _emit_prompt_prepared(
                self.state, label, attempt_prompt, attempt=attempt, **emit_payload
            )
            plan_obj = None
            try:
                plan_obj = await cc_agent.run_planner(
                    cwd=self.state.workspace_dir,
                    prompt=attempt_prompt,
                    on_usage=lambda s: _emit_usage(self.state, s),
                )
                plan_obj = validate_plan(plan_obj)
                if require_structured and not plan_obj.is_structured:
                    raise ValueError(
                        "Planner returned a legacy plan without work packages."
                    )
                if plan_obj.is_structured:
                    _bind_authoritative_asset_hashes(
                        plan_obj, self.state.workspace_dir, self.state.intake_assessment
                    )
                    validate_plan(plan_obj)
                return plan_obj
            except ValueError as exc:
                rejection = exc
                # Log the plan itself, not just the rule that killed it — one
                # error string is not enough to tell a planner bug from a
                # validator bug after a 15-minute high-effort run.
                log.warning(
                    "planner attempt %d/%d rejected: %s\nrejected plan: %s",
                    attempt,
                    attempts,
                    exc,
                    plan_obj.model_dump_json() if plan_obj else "<no plan returned>",
                )
        raise rejection or ValueError("Planner returned no usable plan.")

    def _degraded_plan_gate(self, exc: Exception) -> dict:
        """Re-gate on a planner failure instead of stranding the pending-feedback row.

        CrewAI clears that row before running a resumed listener and only re-saves
        it for HumanFeedbackPending, so any other exception makes the job
        permanently unresumable — and silent in the UI.
        """
        log.warning("planning failed (%s); re-gating for human input", exc)
        self.state.phase = "specification"
        self.state.status = "awaiting_approval"
        detail = (
            f"Automatic planning failed: {exc}. Restate the request with more detail "
            "to try again, or reject it."
        )
        return {
            "phase": "specification",
            "plan": None,
            "planner_error": str(exc),
            "revision_error": detail,
            "open_questions": [detail],
            # Without explicit actions the frontend renders no buttons at all on a
            # plan-less card — a dead end instead of a gate.
            "actions": ["rejected", "amend"],
        }

    @listen(or_("spec_amend", "amend", "qa_amend"))
    @human_feedback(
        message="Revised plan — please review again. Approve, amend further, or reject.",
        emit=["spec_amend", "spec_approved", "job_rejected"],
        llm=GUARDRAIL_LLM,
        default_outcome="spec_amend",
    )
    async def revise_plan(self, prior) -> dict:
        previous_revision = self.state.plan.revision if self.state.plan else 0
        structured_revision = bool(self.state.plan and self.state.plan.is_structured)
        revision_error = ""
        self.state.amendments = _feedback_text(prior)
        self.state.amend_cycles += 1
        self.state.phase = "specification"
        # revise_plan runs DURING resume, AFTER resume_async cleared the
        # pending-feedback row. It must NEVER raise, or the job becomes
        # unresumable ("No pending feedback found"). On any failure, fall back to
        # the prior plan (annotated) and let @human_feedback re-gate.
        try:
            plan_obj = await self._plan_with_repair(
                _planner_prompt(self.state),
                "revise_plan",
                require_structured=structured_revision,
                amend_cycle=self.state.amend_cycles,
            )
        except Exception as exc:  # noqa: BLE001 — a revise failure must never brick the job
            fallback = self._prior_plan_snapshot(prior)
            if fallback is None:
                return self._degraded_plan_gate(exc)
            log.warning("plan revision failed (%s); re-gating with the prior plan", exc)
            revision_error = (
                f"Automatic plan revision failed ({exc}). The previous plan is shown "
                "unchanged; re-state the change or approve it as-is."
            )
            plan_obj = fallback
        if plan_obj.is_structured:
            plan_obj.revision = previous_revision + 1
            plan_obj.plan_markdown = plan_obj.render_markdown()
        self.state.plan = plan_obj
        self.state.language = self.state.language or plan_obj.language or "English"
        self.state.status = "awaiting_approval"
        return {
            "phase": "specification",
            "plan": plan_obj.model_dump(mode="json"),
            "revision_error": revision_error,
            **plan_obj.model_dump(mode="json"),
        }

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

    @listen(or_("job_rejected", "rejected"))
    def on_rejected(self, prior):
        log.info("job %s rejected by human", self.state.id)
        self.state.status = "failed"
        try:
            history.record(self.state)
        except Exception as exc:  # noqa: BLE001 — history is observability, never fatal
            log.warning("history.record on rejection failed: %s", exc)
        return {"status": "failed", "reason": getattr(prior, "feedback", "")}

    @listen(or_("spec_approved", "approved"))
    async def build(self, prior):
        self.state.amendments = _feedback_text(prior) or self.state.amendments
        self.state.status = "executing"
        plan = self.state.plan
        if plan is None:
            # Without an explicit route, route_build coerces this to
            # "execution_complete" and a planless job reports as a success.
            report = QAReport(
                passed=False,
                integration_notes="Build could not start because no approved plan was available.",
            )
            self.state.qa_report = report
            self.state.current_failure = report
            self.state.current_package_id = "__setup__"
            return {"route": "qa_exhausted", "reason": "no plan to execute"}

        if plan.is_structured:
            return await self._build_structured(plan)

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
                    git_tool.init_and_commit(
                        build_dir, "codebuilder baseline (pre-patch)"
                    )
        else:
            build_dir = str(Path(self.state.workspace_dir) / "output")
            Path(build_dir).mkdir(parents=True, exist_ok=True)
            git_tool.init_and_commit(build_dir)

        self._build_dir = build_dir
        self._build_cost_usd = 0.0
        if plan.mode == "patch_existing" and not self.state.baseline_dependencies:
            self.state.baseline_dependencies = project_dependency_names(build_dir)
        _install_skills(Path(build_dir))  # skills for the executor (cwd = build_dir)
        budget = _run_cost_budget_usd()
        initial_budget = _initial_build_budget_usd(budget)
        build_effort = _build_effort(self.state, build_dir)
        _emit_progress(
            self.state,
            "build_started",
            mode=plan.mode,
            build_dir=build_dir,
            requested_model=cc_agent.EXECUTOR_MODEL,
            effort=build_effort,
            cost_budget_usd=budget,
            initial_build_budget_usd=initial_budget,
            qa_budget_reserve_usd=(budget - initial_budget)
            if budget is not None and initial_budget is not None
            else None,
        )
        try:
            await cc_agent.run_executor(
                cwd=build_dir,
                prompt=_executor_prompt(self.state, plan),
                effort=build_effort,
                budget_usd=initial_budget,
                on_usage=self._record_executor_usage,
            )
        except cc_agent.CCBudgetExceeded as exc:
            log.warning("build stopped at cost budget: %s", exc)
            self._build_cost_usd = max(self._build_cost_usd, exc.cost_usd)
            self._build_interruption_note = (
                f"Initial build stopped at its budget allocation "
                f"(est. ${exc.cost_usd:.2f} spent); final QA and the reserved repair "
                "budget will continue from the partial workspace."
            )
            self.state.qa_report = QAReport(
                passed=False,
                integration_notes=self._build_interruption_note,
            )
            _emit_progress(
                self.state, "build_budget_exceeded", est_cost_usd=exc.cost_usd
            )
        except Exception as exc:  # noqa: BLE001 — report a builder crash via QA, don't brick resume
            log.exception("executor agent failed")
            self.state.status = "failed"
            self.state.qa_report = QAReport(
                passed=False,
                integration_notes=f"Executor agent failed: {exc}",
            )
        _emit_progress(self.state, "build_finished", mode=plan.mode)
        return {"route": "execution_complete", "legacy": True}

    @router(
        build,
        emit=["execution_complete", "qa_exhausted", "quarantine_ready"],
    )
    def route_build(self, result: dict | None = None) -> str:
        if isinstance(result, dict):
            return str(result.get("route") or "execution_complete")
        return "execution_complete"

    async def _build_structured(self, plan: Plan) -> dict:
        """Execute the approved DAG without ever mutating the last-green tree."""
        workspace = Path(self.state.workspace_dir)
        source = (
            _resolve_patch_root(self.state.workspace_dir)
            if plan.mode == "patch_existing"
            else None
        )
        if plan.mode == "patch_existing" and source is None:
            report = QAReport(
                passed=False,
                package_id="__setup__",
                issues=[
                    QAIssue(
                        source="contract",
                        owner="spec",
                        message="The approved patch spec has no attached project.",
                        repair_instruction="Attach the project or amend the mode.",
                    )
                ],
            )
            return self._record_failure(report, "", "__setup__")

        if plan.mode == "new_project":
            source_path = workspace / "base"
            if source_path.exists():
                shutil.rmtree(source_path)
            source_path.mkdir()
            for asset in plan.authoritative_assets:
                if not asset.immutable:
                    continue
                resolved = _resolve_authoritative_asset(
                    self.state.workspace_dir, asset.path, plan.mode
                )
                if resolved is None:
                    continue  # binding already reports this before human approval
                destination = source_path / asset.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resolved, destination)
        else:
            assert source is not None
            source_path = Path(source)
        try:
            canonical = package_workspace.stage_tree(
                workspace, source_path, workspace / "output"
            )
            package_workspace.harden_tree(canonical)
        except (OSError, package_workspace.WorkspaceSafetyError) as exc:
            report = self._issue_report(
                "__setup__",
                QAIssue(
                    source="contract",
                    owner="environment",
                    message="The approved project baseline is not safe to stage.",
                    evidence=str(exc),
                    repair_instruction="Remove special files or unsafe links from the attachment and retry.",
                ),
            )
            return self._record_failure(report, "", "__setup__")
        git_tool.init_and_commit(canonical, "codebuilder approved baseline")
        _install_skills(canonical)

        self._build_dir = str(canonical)  # legacy finalize/tests compatibility
        self._build_cost_usd = 0.0
        self.state.canonical_build_dir = str(canonical)
        self.state.approved_spec_hash = plan_spec_hash(plan)
        self.state.package_cursor = 0
        self.state.current_package_id = ""
        self.state.current_stage_dir = ""
        self.state.package_results = []
        self.state.package_repair_attempts = {}
        self.state.skipped_package_ids = []
        self.state.frozen_test_hashes = {}
        self.state.current_failure = None
        self.state.quarantine_report = None
        self.state.quarantine_archive = None
        self.state.zip_path = ""
        self.state.zip_url = ""
        self.state.project_archive = None
        _emit_progress(
            self.state,
            "structured_build_started",
            spec_hash=self.state.approved_spec_hash,
            package_count=len(plan.work_packages),
        )
        return await self._continue_structured_build(plan)

    async def _continue_structured_build(self, plan: Plan) -> dict:
        packages = _ordered_packages(plan)
        for index in range(self.state.package_cursor, len(packages)):
            package = packages[index]
            self.state.package_cursor = index
            if package.id in self.state.skipped_package_ids:
                continue
            result = await self._run_package(plan, package)
            if result.get("route") == "qa_exhausted":
                return result
            self.state.package_cursor = index + 1

        if self.state.skipped_package_ids:
            self._prepare_quarantine()
            return {"route": "quarantine_ready"}

        canonical = self.state.canonical_build_dir
        final_stage = package_workspace.stage_tree(
            self.state.workspace_dir,
            canonical,
            Path(self.state.workspace_dir) / "stages" / "__final__",
        )
        trusted = package_workspace.stage_tree(
            self.state.workspace_dir,
            canonical,
            Path(self.state.workspace_dir) / "trusted-tests" / "__final__",
        )
        self.state.current_package_id = "__final__"
        self.state.current_stage_dir = str(final_stage)
        self.state.frozen_test_hashes = package_workspace.snapshot_files(
            trusted, self._scope_test_paths(plan, None)
        )
        report = await self._structured_qa(final_stage, plan, None)
        if not report.passed:
            return self._record_failure(report, str(final_stage), "__final__")

        self.state.qa_report = report
        self.state.current_failure = None
        return {"route": "execution_complete"}

    async def _run_package(
        self, plan: Plan, package: WorkPackageSpec, guidance: str = ""
    ) -> dict:
        workspace = self.state.workspace_dir
        canonical = self.state.canonical_build_dir
        stage = package_workspace.stage_tree(
            workspace,
            canonical,
            Path(workspace) / "stages" / package.id,
        )
        _install_skills(stage)
        git_tool.init_and_commit(stage, f"codebuilder {package.id} baseline")
        self.state.current_package_id = package.id
        self.state.current_stage_dir = str(stage)
        test_paths = self._scope_test_paths(plan, package)
        baseline = package_workspace.snapshot_files(stage)
        _emit_progress(self.state, "test_author_started", package_id=package.id)
        try:
            await cc_agent.run_test_author(
                cwd=stage,
                prompt=_test_author_prompt(self.state, plan, package, guidance),
                declared_test_files=test_paths,
                budget_usd=(
                    self._remaining_budget()
                    if guidance
                    else self._remaining_initial_budget()
                ),
                on_usage=self._record_executor_usage,
            )
        except Exception as exc:  # noqa: BLE001 — surface through the QA decision gate
            report = self._issue_report(
                package.id,
                QAIssue(
                    source="review",
                    owner="environment",
                    message=f"Test author could not complete: {exc}",
                    repair_instruction="Retry after fixing the agent environment.",
                ),
            )
            return self._record_failure(report, str(stage), package.id)

        changed = package_workspace.changed_paths(
            baseline, package_workspace.snapshot_files(stage)
        )
        unexpected = sorted(set(changed) - set(test_paths))
        missing = [
            path
            for path in test_paths
            if (stage / path).is_symlink() or not (stage / path).is_file()
        ]
        if unexpected:
            package_workspace.restore_files(workspace, stage, canonical, unexpected)
        declared_test_output = check_declared_tests(str(stage), package)
        if unexpected or missing or declared_test_output != "PASS":
            details = []
            if unexpected:
                details.append("unapproved changes: " + ", ".join(unexpected))
            if missing:
                details.append("missing tests: " + ", ".join(missing))
            if declared_test_output != "PASS":
                details.append(declared_test_output)
            report = self._issue_report(
                package.id,
                QAIssue(
                    source="contract",
                    owner="test",
                    message="Test author violated the approved test contract.",
                    evidence="; ".join(details),
                    repair_instruction="Rewrite only the declared tests from the approved criteria.",
                ),
            )
            return self._record_failure(report, str(stage), package.id)

        package_workspace.harden_tree(stage)
        trusted = package_workspace.stage_tree(
            workspace,
            canonical,
            Path(workspace) / "trusted-tests" / package.id,
        )
        package_workspace.promote_files(workspace, stage, trusted, test_paths)
        self.state.frozen_test_hashes.update(
            package_workspace.snapshot_files(trusted, test_paths)
        )
        red_results, red_issues = self._run_protected_verification(
            stage, plan, f"{package.id}-red"
        )
        required = {
            command.id: command.required for command in plan.verification_commands
        }
        for result in red_results:
            if not required.get(result.command_id, True) or result.returncode not in {
                124,
                125,
                126,
                127,
            }:
                continue
            red_issues.append(
                QAIssue(
                    source="command",
                    owner=(
                        "spec" if result.returncode in {125, 126} else "environment"
                    ),
                    message=f"Pre-implementation command {result.command_id!r} could not run safely.",
                    evidence="\n".join(
                        value for value in (result.stdout, result.stderr) if value
                    ),
                    repair_instruction="Amend the verification command or restore the required environment.",
                )
            )
        _emit_progress(
            self.state,
            "tests_frozen",
            package_id=package.id,
            test_paths=test_paths,
            preimplementation_results=[
                result.model_dump(mode="json") for result in red_results
            ],
        )
        if red_issues:
            report = QAReport(
                passed=False,
                spec_hash=self.state.approved_spec_hash,
                package_id=package.id,
                command_results=red_results,
                issues=red_issues,
                contract_issues=red_issues,
                integration_notes="Pre-implementation verification could not run safely.",
            )
            return self._record_failure(report, str(stage), package.id)

        report = await self._mutate_and_check(
            stage, trusted, plan, package, guidance=guidance
        )
        attempts = self._max_final_qa_repairs()
        while (
            not report.passed
            and self._only_code_issues(report)
            and self.state.package_repair_attempts.get(package.id, 0) < attempts
        ):
            count = self.state.package_repair_attempts.get(package.id, 0) + 1
            self.state.package_repair_attempts[package.id] = count
            self.state.final_qa_repair_attempts += 1
            report = await self._mutate_and_check(
                stage, trusted, plan, package, repair_report=report
            )

        if not report.passed:
            report.repair_count = self.state.package_repair_attempts.get(package.id, 0)
            return self._record_failure(report, str(stage), package.id)
        return self._promote_green_stage(plan, package, stage, report)

    async def _mutate_and_check(
        self,
        stage: Path,
        trusted: Path,
        plan: Plan,
        package: WorkPackageSpec | None,
        *,
        repair_report: QAReport | None = None,
        guidance: str = "",
    ) -> QAReport:
        scope = package or self._aggregate_scope(plan)
        boundary_issues: list[QAIssue] = []
        remaining = (
            self._remaining_budget()
            if repair_report is not None or guidance
            else self._remaining_initial_budget()
        )
        if remaining is not None and remaining <= 0:
            return self._issue_report(
                scope.id,
                QAIssue(
                    source="review",
                    owner="environment",
                    message="The build cost budget was reached.",
                    repair_instruction="Raise the budget or terminate the run.",
                ),
            )
        try:
            await cc_agent.run_executor(
                cwd=stage,
                prompt=_package_executor_prompt(
                    self.state,
                    plan,
                    scope,
                    repair_report=repair_report,
                    guidance=guidance,
                ),
                effort=(
                    cc_agent.REPAIR_EFFORT
                    if repair_report is not None
                    else _build_effort(self.state, str(stage))
                ),
                budget_usd=remaining,
                on_usage=self._record_executor_usage,
            )
        except cc_agent.CCBudgetExceeded as exc:
            boundary_issues.append(
                QAIssue(
                    source="review",
                    owner="environment",
                    message=f"Implementation stopped at the cost budget (${exc.cost_usd:.2f}).",
                    repair_instruction="Raise the budget or terminate the run.",
                )
            )
        except Exception as exc:  # noqa: BLE001
            boundary_issues.append(
                QAIssue(
                    source="review",
                    owner="environment",
                    message=f"Implementation agent could not complete: {exc}",
                    repair_instruction="Retry after fixing the agent environment.",
                )
            )
        boundary_issues.extend(
            self._restore_stage_boundaries(stage, trusted, plan, package)
        )
        package_workspace.harden_tree(stage)
        report = await self._structured_qa(stage, plan, package)
        if boundary_issues:
            report.issues.extend(boundary_issues)
            report.passed = False
            _append_note(
                report,
                "Workspace boundary violations were restored before QA: "
                + "; ".join(issue.message for issue in boundary_issues),
            )
        return report

    def _restore_stage_boundaries(
        self,
        stage: Path,
        trusted: Path,
        plan: Plan,
        package: WorkPackageSpec | None,
    ) -> list[QAIssue]:
        immutable_assets = {
            asset.path for asset in plan.authoritative_assets if asset.immutable
        }
        allowed = set(self._scope_paths(plan, package)) - immutable_assets
        tests = self._scope_test_paths(plan, package)
        canonical_snapshot = package_workspace.snapshot_files(
            self.state.canonical_build_dir
        )
        stage_snapshot = package_workspace.snapshot_files(stage)
        changed = package_workspace.changed_paths(
            canonical_snapshot,
            stage_snapshot,
        )
        mutated_assets = sorted(set(changed) & immutable_assets)
        special_nodes = {
            path
            for path, value in stage_snapshot.items()
            if value.startswith("special:")
        }
        unexpected = sorted(
            (set(changed) - allowed - immutable_assets)
            | (special_nodes - set(tests) - immutable_assets)
        )
        mutated_tests = package_workspace.changed_paths(
            package_workspace.snapshot_files(trusted, tests),
            package_workspace.snapshot_files(stage, tests),
        )
        if unexpected:
            package_workspace.restore_files(
                self.state.workspace_dir,
                stage,
                self.state.canonical_build_dir,
                unexpected,
            )
        if mutated_tests:
            package_workspace.restore_files(
                self.state.workspace_dir, stage, trusted, mutated_tests
            )
        if mutated_assets:
            package_workspace.restore_files(
                self.state.workspace_dir,
                stage,
                self.state.canonical_build_dir,
                mutated_assets,
            )
        issues: list[QAIssue] = []
        if unexpected:
            issues.append(
                QAIssue(
                    source="contract",
                    owner="code",
                    message="Implementation changed files outside its package.",
                    evidence=", ".join(unexpected),
                    repair_instruction="Make the fix only in approved package files.",
                )
            )
        if mutated_tests:
            issues.append(
                QAIssue(
                    source="contract",
                    owner="code",
                    message="Implementation attempted to change frozen tests.",
                    evidence=", ".join(mutated_tests),
                    repair_instruction="Fix production code; frozen tests are immutable.",
                )
            )
        if mutated_assets:
            issues.append(
                QAIssue(
                    source="contract",
                    owner="code",
                    message="Implementation attempted to change authoritative assets.",
                    evidence=", ".join(mutated_assets),
                    repair_instruction="Treat approved authoritative assets as immutable.",
                )
            )
        return issues

    def _run_protected_verification(
        self, build_path: Path, plan: Plan, evidence_id: str
    ) -> tuple[list[CommandResult], list[QAIssue]]:
        workspace = Path(self.state.workspace_dir)
        issues: list[QAIssue] = []
        results: list[CommandResult] = []
        protected: list[tuple[Path, Path, dict[str, str]]] = []
        with tempfile.TemporaryDirectory(
            prefix=".verification-guard-", dir=workspace
        ) as guard_dir:
            guard = Path(guard_dir)
            candidates = [
                build_path,
                _validated_workspace_tree(
                    self.state.workspace_dir, self.state.canonical_build_dir
                ),
                _validated_workspace_tree(
                    self.state.workspace_dir,
                    str(workspace / "trusted-tests"),
                ),
            ]
            seen: set[Path] = set()
            for candidate in candidates:
                if candidate is None:
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                backup = package_workspace.stage_tree(
                    workspace, resolved, guard / f"protected-{len(protected)}"
                )
                protected.append(
                    (resolved, backup, package_workspace.snapshot_files(resolved))
                )

            try:
                with tempfile.TemporaryDirectory(
                    prefix=f"codebuilder-verify-{evidence_id}-"
                ) as temporary:
                    disposable = package_workspace.copy_clean_tree(
                        build_path, Path(temporary) / "project"
                    )
                    before = package_workspace.snapshot_files(disposable)
                    results = run_verification_commands(
                        str(disposable), plan.verification_commands
                    )
                    try:
                        changes = package_workspace.changed_paths(
                            before, package_workspace.snapshot_files(disposable)
                        )
                    except (OSError, package_workspace.WorkspaceSafetyError) as exc:
                        changes = [f"<disposable tree unavailable: {exc}>"]
            except (OSError, package_workspace.WorkspaceSafetyError) as exc:
                changes = [f"<verification isolation failed: {exc}>"]
            categories = {
                command.id: command.category for command in plan.verification_commands
            }
            changes = sorted(
                {
                    *changes,
                    *(
                        f"{result.command_id}:{path}"
                        for result in results
                        for path in result.mutated_paths
                        if categories.get(result.command_id) != "build"
                    ),
                }
            )

            protected_changes: list[str] = []
            for original, backup, before in protected:
                try:
                    after = package_workspace.snapshot_files(original)
                    changed = package_workspace.changed_paths(before, after)
                    if changed:
                        package_workspace.restore_files(
                            workspace, original, backup, changed
                        )
                except (OSError, package_workspace.WorkspaceSafetyError) as exc:
                    changed = [f"<tree unavailable: {exc}>"]
                protected_changes.extend(f"{original.name}/{path}" for path in changed)

        if changes:
            issues.append(
                QAIssue(
                    source="contract",
                    owner="spec",
                    message="A verification command mutated its disposable project copy.",
                    evidence=", ".join(changes),
                    repair_instruction="Use read-only verification commands; move generation into an approved work package.",
                )
            )
        if protected_changes:
            issues.append(
                QAIssue(
                    source="contract",
                    owner="spec",
                    message="A verification command attempted to mutate protected build evidence.",
                    evidence=", ".join(protected_changes),
                    repair_instruction="Use verification commands that cannot write outside their disposable copy.",
                )
            )
        return results, issues

    async def _structured_qa(
        self, build_dir: str | Path, plan: Plan, package: WorkPackageSpec | None
    ) -> QAReport:
        build_path = Path(build_dir)
        package_id = package.id if package else "__final__"
        contract_plan = plan
        if package is not None:
            contract_plan = plan.model_copy(
                deep=True,
                update={
                    "tech_stack": [],
                    "identifier_contract": IdentifierContract(),
                    "work_packages": [package.model_copy(update={"depends_on": []})],
                },
            )

        # Verification is evidence, never a build step, so commands run only on
        # a disposable clean copy of the stage.
        results, issues = self._run_protected_verification(build_path, plan, package_id)
        required = {
            command.id: command.required for command in plan.verification_commands
        }
        contract_output = check_spec_contract(str(build_path), contract_plan)
        if contract_output != "PASS":
            issues.append(
                QAIssue(
                    source="contract",
                    owner="spec"
                    if contract_output.startswith("Invalid plan:")
                    else "code",
                    message="Approved specification contract failed.",
                    evidence=contract_output,
                    repair_instruction="Restore the exact approved files, tests, and identifiers.",
                )
            )
        dependency_output = (
            check_preserved_dependencies(
                str(build_path), self.state.baseline_dependencies
            )
            if plan.mode == "patch_existing"
            else "PASS"
        )
        if dependency_output != "PASS":
            issues.append(
                QAIssue(
                    source="contract",
                    owner="code",
                    message="Existing project dependencies were removed.",
                    evidence=dependency_output,
                    repair_instruction="Restore every dependency present in the approved baseline.",
                )
            )
        for result in results:
            if result.passed:
                continue
            issues.append(
                QAIssue(
                    source="command",
                    owner=(
                        "spec"
                        if result.returncode in {125, 126}
                        else "environment"
                        if result.returncode in {124, 127}
                        else "code"
                    ),
                    message=f"Verification command {result.command_id!r} failed.",
                    evidence="\n".join(
                        value for value in (result.stdout, result.stderr) if value
                    ),
                    repair_instruction="Fix the implementation without changing the approved tests.",
                    blocking=required.get(result.command_id, True),
                )
            )
        report = QAReport(
            passed=not any(issue.blocking for issue in issues),
            spec_hash=self.state.approved_spec_hash,
            package_id=package_id,
            command_results=results,
            issues=issues,
            contract_issues=[issue for issue in issues if issue.source == "contract"],
            lint_output=self._command_evidence(plan, results, "lint"),
            type_output=self._command_evidence(plan, results, "typecheck"),
            test_output=self._command_evidence(plan, results, "test"),
            integration_notes=(
                "Approved spec contract: "
                + ("PASS" if contract_output == "PASS" else contract_output)
                + "\nDependency preservation: "
                + dependency_output
            ),
        )
        if report.passed:
            await self._apply_semantic_review(str(build_path), plan, package, report)
        _emit_progress(
            self.state,
            "package_qa_completed",
            package_id=report.package_id,
            passed=report.passed,
            issues=[issue.model_dump(mode="json") for issue in report.issues],
        )
        return report

    async def _apply_semantic_review(
        self,
        build_dir: str,
        plan: Plan,
        package: WorkPackageSpec | None,
        report: QAReport,
    ) -> None:
        try:
            review = await cc_agent.run_reviewer(
                cwd=build_dir,
                prompt=_semantic_review_prompt(self.state, plan, report, package),
                budget_usd=self._remaining_budget(),
                on_usage=self._record_executor_usage,
            )
        except Exception as exc:  # noqa: BLE001 — semantic QA fails closed
            review = ProductionReview(
                passed=False,
                qa_issues=[
                    QAIssue(
                        source="review",
                        owner="environment",
                        message=f"Semantic review could not complete: {exc}",
                        repair_instruction="Retry the review or terminate the run.",
                    )
                ],
            )
        review_issues = list(review.qa_issues)
        review_issues.extend(
            QAIssue(source="review", owner="code", message=message)
            for message in review.issues
        )
        if not review.passed and not review_issues:
            review_issues.append(
                QAIssue(
                    source="review",
                    owner="environment",
                    message="Semantic reviewer failed without a concrete issue.",
                    repair_instruction="Retry the review or terminate the run.",
                )
            )
        if review.passed and review_issues:
            review.passed = False
        self.state.production_review = review
        if not review.passed:
            report.review_issues = review_issues
            report.issues.extend(review_issues)
            report.passed = False
            _append_note(
                report,
                "Semantic review failed: "
                + "; ".join(issue.message for issue in review_issues),
            )

    def _promote_green_stage(
        self,
        plan: Plan,
        package: WorkPackageSpec | None,
        stage: Path,
        report: QAReport,
    ) -> dict:
        package_workspace.promote_files(
            self.state.workspace_dir,
            stage,
            self.state.canonical_build_dir,
            self._scope_paths(plan, package),
        )
        if package is not None:
            self._set_package_result(
                PackageResult(
                    package_id=package.id,
                    spec_hash=self.state.approved_spec_hash,
                    status="passed",
                    command_results=report.command_results,
                    issues=report.issues,
                    repair_count=self.state.package_repair_attempts.get(package.id, 0),
                )
            )
        self.state.qa_report = report
        self.state.current_failure = None
        return {"route": "package_complete"}

    def _record_failure(self, report: QAReport, stage: str, package_id: str) -> dict:
        self.state.qa_report = report
        self.state.current_failure = report
        self.state.current_package_id = package_id
        self.state.current_stage_dir = stage
        self.state.phase = "qa_failure"
        if package_id not in {"__final__", "__setup__"}:
            self._set_package_result(
                PackageResult(
                    package_id=package_id,
                    spec_hash=self.state.approved_spec_hash,
                    status="failed",
                    command_results=report.command_results,
                    issues=report.issues,
                    repair_count=self.state.package_repair_attempts.get(package_id, 0),
                )
            )
        return {"route": "qa_exhausted"}

    @listen("qa_exhausted")
    @human_feedback(
        message="QA is still failing. Choose retry with guidance, amend the spec, skip this package and its dependents, or terminate with a quarantine bundle.",
        # emit[0] is the classifier-failure fallback (see plan()); amending is the
        # only outcome here that neither discards work nor burns another build.
        emit=["qa_amend", "qa_retry", "qa_skip", "qa_terminate"],
        llm=GUARDRAIL_LLM,
        default_outcome="qa_terminate",
    )
    def review_qa_failure(self) -> dict:
        self.state.phase = "qa_failure"
        self.state.status = "awaiting_approval"
        report = self.state.current_failure or QAReport(passed=False)
        return {
            "phase": "qa_failure",
            "package_id": self.state.current_package_id,
            "can_skip": self.state.current_package_id not in {"__final__", "__setup__"},
            "qa_report": report.model_dump(mode="json"),
            "package_results": [
                result.model_dump(mode="json") for result in self.state.package_results
            ],
        }

    def _degraded_qa_gate(self, exc: Exception) -> dict:
        """Re-gate a failed QA action rather than strand the job (see _degraded_plan_gate)."""
        log.warning("QA action failed (%s); re-gating for human input", exc)
        report = self.state.current_failure or QAReport(passed=False)
        self.state.current_failure = report.model_copy(
            update={
                "integration_notes": (
                    f"{report.integration_notes}\n"
                    f"The requested action could not be completed: {exc}"
                ).strip()
            }
        )
        return {"route": "qa_exhausted"}

    @listen("qa_retry")
    async def retry_failed_qa(self, prior) -> dict:
        # Runs during resume; raising strands the pending-feedback row.
        try:
            return await self._retry_failed_qa(prior)
        except Exception as exc:  # noqa: BLE001 — see above
            return self._degraded_qa_gate(exc)

    async def _retry_failed_qa(self, prior) -> dict:
        plan = self.state.plan
        if plan is None:
            return {"route": "qa_exhausted"}
        if self.state.current_package_id == "__setup__":
            return await self._build_structured(plan)
        guidance = _feedback_text(prior) or "Human requested a retry."
        package = self._package_by_id(plan, self.state.current_package_id)
        owners = {
            issue.owner
            for issue in (
                self.state.current_failure.issues if self.state.current_failure else []
            )
            if issue.blocking
        }
        if package is not None and "test" in owners:
            result = await self._run_package(plan, package, guidance)
            if result.get("route") == "qa_exhausted":
                return result
            self.state.package_cursor += 1
            return await self._continue_structured_build(plan)
        if owners and owners <= {"code"}:
            return await self._repair_current_stage(plan, package, guidance)
        report = await self._structured_qa(self.state.current_stage_dir, plan, package)
        if not report.passed:
            return self._record_failure(
                report, self.state.current_stage_dir, self.state.current_package_id
            )
        return await self._resume_after_green(plan, package, report)

    @router(
        retry_failed_qa,
        emit=["execution_complete", "qa_exhausted", "quarantine_ready"],
    )
    def route_qa_retry(self, result: dict | None = None) -> str:
        return str((result or {}).get("route") or "qa_exhausted")

    async def _repair_current_stage(
        self, plan: Plan, package: WorkPackageSpec | None, guidance: str
    ) -> dict:
        stage = Path(self.state.current_stage_dir)
        trusted = (
            Path(self.state.workspace_dir)
            / "trusted-tests"
            / (package.id if package else "__final__")
        )
        report = await self._mutate_and_check(
            stage,
            trusted,
            plan,
            package,
            repair_report=self.state.current_failure,
            guidance=guidance,
        )
        if not report.passed:
            return self._record_failure(
                report, str(stage), package.id if package else "__final__"
            )
        return await self._resume_after_green(plan, package, report)

    async def _resume_after_green(
        self, plan: Plan, package: WorkPackageSpec | None, report: QAReport
    ) -> dict:
        self._promote_green_stage(
            plan, package, Path(self.state.current_stage_dir), report
        )
        if package is None:
            return {"route": "execution_complete"}
        self.state.package_cursor += 1
        return await self._continue_structured_build(plan)

    @listen("qa_skip")
    async def skip_failed_package(self, _prior=None) -> dict:
        # Runs during resume; raising strands the pending-feedback row.
        try:
            return await self._skip_failed_package()
        except Exception as exc:  # noqa: BLE001 — see above
            return self._degraded_qa_gate(exc)

    async def _skip_failed_package(self) -> dict:
        plan = self.state.plan
        failed_id = self.state.current_package_id
        if plan is None or failed_id in {"__final__", "__setup__"}:
            self._prepare_quarantine()
            return {"route": "quarantine_ready"}
        self._preserve_failed_stage()
        skipped = {failed_id}
        changed = True
        while changed:
            changed = False
            for package in plan.work_packages:
                if package.id not in skipped and skipped.intersection(
                    package.depends_on
                ):
                    skipped.add(package.id)
                    changed = True
        self.state.skipped_package_ids = sorted(
            set(self.state.skipped_package_ids) | skipped
        )
        for package_id in skipped:
            if package_id != failed_id:
                self._set_package_result(
                    PackageResult(
                        package_id=package_id,
                        spec_hash=self.state.approved_spec_hash,
                        status="skipped",
                    )
                )
        self.state.package_cursor += 1
        return await self._continue_structured_build(plan)

    @router(
        skip_failed_package,
        emit=["execution_complete", "qa_exhausted", "quarantine_ready"],
    )
    def route_qa_skip(self, result: dict | None = None) -> str:
        return str((result or {}).get("route") or "quarantine_ready")

    @listen("qa_terminate")
    def terminate_failed_qa(self, _prior=None) -> dict:
        self._prepare_quarantine()
        return {"route": "quarantine_ready"}

    @router(terminate_failed_qa, emit=["quarantine_ready"])
    def route_qa_terminate(self, _result=None) -> str:
        return "quarantine_ready"

    def _scope_paths(self, plan: Plan, package: WorkPackageSpec | None) -> list[str]:
        packages = [package] if package is not None else plan.work_packages
        return list(
            dict.fromkeys(file.path for item in packages for file in item.files)
        )

    def _scope_test_paths(
        self, plan: Plan, package: WorkPackageSpec | None
    ) -> list[str]:
        packages = [package] if package is not None else plan.work_packages
        return list(
            dict.fromkeys(
                file.path
                for item in packages
                for file in item.files
                if file.kind == "test"
            )
        )

    def _aggregate_scope(self, plan: Plan) -> WorkPackageSpec:
        return WorkPackageSpec(
            id="__final__",
            title="Final integration",
            what_to_build="Repair only cross-package integration blockers.",
            expected_behavior="Every approved criterion and command passes together.",
            success_criteria=[
                criterion
                for package in plan.work_packages
                for criterion in package.success_criteria
            ],
            tests=[test for package in plan.work_packages for test in package.tests],
            files=[file for package in plan.work_packages for file in package.files],
        )

    @staticmethod
    def _command_evidence(
        plan: Plan, results: list[CommandResult], category: str
    ) -> str:
        categories = {
            command.id: command.category for command in plan.verification_commands
        }
        chunks = []
        for result in results:
            if categories.get(result.command_id) != category:
                continue
            output = "\n".join(
                value for value in (result.stdout, result.stderr) if value
            )
            chunks.append(
                f"{result.command_id}: {'PASS' if result.passed else 'FAIL'}"
                + (f"\n{output}" if output else "")
            )
        return "\n\n".join(chunks)

    def _issue_report(self, package_id: str, issue: QAIssue) -> QAReport:
        return QAReport(
            passed=False,
            spec_hash=self.state.approved_spec_hash,
            package_id=package_id,
            issues=[issue],
            integration_notes=issue.message,
        )

    @staticmethod
    def _only_code_issues(report: QAReport) -> bool:
        blocking = [issue for issue in report.issues if issue.blocking]
        return bool(blocking) and all(issue.owner == "code" for issue in blocking)

    def _set_package_result(self, result: PackageResult) -> None:
        self.state.package_results = [
            existing
            for existing in self.state.package_results
            if existing.package_id != result.package_id
        ]
        self.state.package_results.append(result)

    @staticmethod
    def _package_by_id(plan: Plan, package_id: str) -> WorkPackageSpec | None:
        return next(
            (package for package in plan.work_packages if package.id == package_id),
            None,
        )

    def _preserve_failed_stage(self) -> None:
        source = _validated_workspace_tree(
            self.state.workspace_dir, self.state.current_stage_dir
        )
        if source is None:
            return
        destination = (
            Path(self.state.workspace_dir)
            / "quarantine"
            / "failed-stages"
            / self.state.current_package_id
        )
        if not destination.exists():
            package_workspace.stage_tree(self.state.workspace_dir, source, destination)

    def _prepare_quarantine(self) -> None:
        plan = self.state.plan
        if plan is None:
            self.state.status = "failed"
            return
        canonical = _validated_workspace_tree(
            self.state.workspace_dir, self.state.canonical_build_dir
        )
        report_only = canonical is None
        if report_only:
            setup_root = Path(self.state.workspace_dir) / "quarantine" / "setup"
            canonical_candidate = setup_root / "last-green"
            failed_candidate = setup_root / "failed-stage"
            canonical_candidate.mkdir(parents=True, exist_ok=True)
            failed_candidate.mkdir(parents=True, exist_ok=True)
            canonical = _validated_workspace_tree(
                self.state.workspace_dir, str(canonical_candidate)
            )
            failed_stage = _validated_workspace_tree(
                self.state.workspace_dir, str(failed_candidate)
            )
            if canonical is None or failed_stage is None:
                self.state.status = "failed"
                return
        failed_results = [
            result.package_id
            for result in self.state.package_results
            if result.status == "failed"
        ]
        failed_id = (
            failed_results[0] if failed_results else self.state.current_package_id
        )
        if failed_id in {"", "__final__"} and self.state.skipped_package_ids:
            failed_id = self.state.skipped_package_ids[0]
        preserved = _validated_workspace_tree(
            self.state.workspace_dir,
            str(
                Path(self.state.workspace_dir)
                / "quarantine"
                / "failed-stages"
                / failed_id
            ),
        )
        if not report_only:
            failed_stage = preserved or _validated_workspace_tree(
                self.state.workspace_dir, self.state.current_stage_dir
            )
        if failed_stage is None:
            failed_stage = package_workspace.stage_tree(
                self.state.workspace_dir,
                canonical,
                Path(self.state.workspace_dir) / "quarantine" / "fallback-stage",
            )

        issues = [
            issue
            for result in self.state.package_results
            if result.status == "failed"
            for issue in result.issues
        ]
        if self.state.current_failure:
            issues = self.state.current_failure.issues or issues
        if not issues and self.state.skipped_package_ids:
            issues = [
                QAIssue(
                    source="review",
                    owner="spec",
                    message="The human skipped one or more failed work packages.",
                    evidence=", ".join(self.state.skipped_package_ids),
                    repair_instruction="Amend or retry the skipped packages before release.",
                )
            ]
        report = (
            self.state.current_failure or self.state.qa_report or QAReport(passed=False)
        )
        report.passed = False
        report.issues = issues or report.issues
        _append_note(report, "Release blocked; failed evidence is quarantined.")
        self.state.qa_report = report
        self.state.current_failure = report

        self.state.quarantine_report = QuarantineReport(
            spec_hash=self.state.approved_spec_hash,
            last_green_package_ids=[
                result.package_id
                for result in self.state.package_results
                if result.status == "passed"
            ],
            failed_package_id=failed_id,
            skipped_package_ids=self.state.skipped_package_ids,
            issues=report.issues,
            package_results=self.state.package_results,
        )
        package = self._package_by_id(plan, failed_id)
        approved_paths = [] if report_only else self._scope_paths(plan, package)
        destination = (
            Path(self.state.workspace_dir)
            / "quarantine"
            / f"{_safe_zip_stem(self.state.project_name or self.state.id)}-failed.zip"
        )
        archive = package_workspace.create_quarantine_zip(
            self.state.workspace_dir,
            canonical,
            failed_stage,
            approved_paths,
            _canonical_spec_json(plan),
            self._qa_report_markdown(str(failed_stage)),
            destination,
        )
        self.state.quarantine_archive = QuarantineArchiveRef(
            file_path=archive.name,
            size=archive.stat().st_size,
            local_path=str(archive),
        )
        self.state.status = "failed"

    @listen(or_("execution_complete", "quarantine_ready"))
    async def finalize(self, _prior=None):
        plan = self.state.plan
        structured = bool(plan and plan.is_structured)
        build_dir = (
            self.state.canonical_build_dir
            if structured
            else getattr(self, "_build_dir", None)
        )
        build_failed = self.state.status == "failed"

        # Nothing was built (e.g. "no plan to execute") — there's nothing to
        # package. Any real build (even a failed/budget-stopped one) has a
        # build_dir and falls through so its partial work is still delivered.
        if plan is None or not build_dir:
            try:
                history.record(self.state)
            except Exception as exc:  # noqa: BLE001 — history is observability, never fatal
                log.warning("history.record on build failure failed: %s", exc)
            return self._completion_payload(build_dir or self.state.workspace_dir)

        build_failure_note = (
            self.state.qa_report.integration_notes
            if build_failed and self.state.qa_report is not None
            else ""
        )
        build_interruption_note = getattr(self, "_build_interruption_note", "")
        if not structured:
            if not build_failed:
                self._apply_safe_generated_fixes(build_dir)
            _emit_progress(self.state, "final_qa_started")
            try:
                self.state.qa_report = self._run_final_qa(build_dir)
            except Exception as exc:  # noqa: BLE001 — preserve a deterministic failure report
                log.exception("final QA failed unexpectedly")
                self.state.qa_report = QAReport(
                    passed=False,
                    integration_notes=f"Final QA could not complete: {exc}",
                )
            if build_failure_note:
                self.state.qa_report.passed = False
                _append_note(self.state.qa_report, build_failure_note)
            elif build_interruption_note:
                _append_note(self.state.qa_report, build_interruption_note)
            if not build_failed:
                await self._certify_and_repair(build_dir)

        if build_dir and self.state.plan and self.state.plan.mode == "patch_existing":
            try:
                self.state.patch = git_tool.diff(build_dir)
            except Exception as exc:  # noqa: BLE001
                log.warning("patch generation failed: %s", exc)
                self.state.patch = ""

        if self.state.plan and self.state.qa_report and self.state.qa_report.passed:
            try:
                zip_path = _zip_build(
                    build_dir,
                    Path(self.state.workspace_dir),
                    self.state.project_name or self.state.id,
                    (
                        {
                            "approved-spec.json": _canonical_spec_json(plan),
                            "QA.md": self._qa_report_markdown(build_dir),
                        }
                        if structured
                        else None
                    ),
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
                    _append_note(
                        self.state.qa_report,
                        f"Project archive generation failed: {exc}",
                    )

        if self.state.qa_report:
            session_segment = (
                self.state.project_key or self.state.session_id or self.state.id
            )
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

            if self.state.quarantine_archive:
                quarantine_ref = upload_file(
                    self.state.quarantine_archive.local_path,
                    key=f"{prefix}/{self.state.quarantine_archive.file_path}",
                )
                if quarantine_ref:
                    quarantine_artifact = ArtifactRef(
                        **{**quarantine_ref, "kind": "quarantine_archive"}
                    )
                    self.state.quarantine_archive.url = quarantine_artifact.url
                    uploaded_refs.append(quarantine_artifact)
                elif os.environ.get("CODEBUILDER_ARTIFACT_BUCKET"):
                    _append_note(
                        self.state.qa_report,
                        "Quarantine archive upload failed; the local evidence bundle remains available.",
                    )

            if self.state.qa_report.passed and _upload_file_artifacts_enabled(
                self.state.plan
            ):
                try:
                    uploaded_refs.extend(
                        artifact_refs(upload_workspace(build_dir, prefix=prefix))
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("workspace artifact upload failed: %s", exc)
            elif self.state.qa_report.passed:
                _emit_progress(
                    self.state,
                    "file_artifact_upload_skipped",
                    reason="disabled_for_patch_existing",
                )

            self.state.qa_report.artifact_urls = uploaded_refs

        if self.state.qa_report and not self.state.qa_report.passed:
            self.state.zip_path = ""
            self.state.zip_url = ""
            self.state.project_archive = None
            self.state.qa_report.artifact_urls = [
                ref
                for ref in self.state.qa_report.artifact_urls
                if ref.kind != "project_archive"
            ]

        self.state.status = (
            "done"
            if self.state.qa_report is None or self.state.qa_report.passed
            else "failed"
        )
        self.state.phase = (
            "complete"
            if self.state.status == "done"
            else "quarantine"
            if self.state.quarantine_archive
            else "failed"
        )
        log.info("job %s complete", self.state.id)

        completion = self._completion_payload(build_dir)
        _emit_progress(
            self.state,
            "final_qa_completed",
            **completion,
            passed=bool(self.state.qa_report and self.state.qa_report.passed),
            repair_attempts=self.state.final_qa_repair_attempts,
            integration_notes=self.state.qa_report.integration_notes
            if self.state.qa_report
            else "",
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
        persisted = sum(
            float(item["cost_usd"])
            for item in self.state.llm_usage
            if item.get("stage") in {"test_author", "executor", "semantic_reviewer"}
            and isinstance(item.get("cost_usd"), (int, float))
        )
        return max(0.0, budget - max(getattr(self, "_build_cost_usd", 0.0), persisted))

    def _remaining_initial_budget(self) -> float | None:
        budget = _run_cost_budget_usd()
        cap = _initial_build_budget_usd(budget)
        remaining = self._remaining_budget()
        if cap is None or remaining is None or budget is None:
            return None
        spent = budget - remaining
        return max(0.0, cap - spent)

    def _run_final_qa(self, build_dir: str) -> QAReport:
        plan = self.state.plan
        return run_final_qa(
            build_dir,
            require_installable=(Path(build_dir) / "pyproject.toml").is_file(),
            require_typecheck=self._is_rpa_build(build_dir),
            locked_sync=True,
            baseline_dependencies=self.state.baseline_dependencies
            if plan is not None and plan.mode == "patch_existing"
            else None,
        )

    def _max_final_qa_repairs(self) -> int:
        return _env_int(
            "CODEBUILDER_MAX_FINAL_QA_REPAIRS", DEFAULT_MAX_FINAL_QA_REPAIRS
        )

    def _apply_safe_generated_fixes(self, build_dir: str) -> None:
        plan = self.state.plan
        if plan is None or plan.mode != "new_project":
            return
        output = apply_ruff_fixes(build_dir)
        if output != "PASS":
            log.warning("Ruff safe fixes did not complete cleanly: %s", output)

    def _is_rpa_build(self, build_dir: str) -> bool:
        plan = self.state.plan
        return bool(
            plan
            and (plan.domain.lower() == "rpa" or _looks_like_rpa(self.state, build_dir))
        )

    async def _apply_production_review(self, build_dir: str) -> bool:
        """Apply the RPA-only semantic gate. False means review infrastructure failed."""
        plan = self.state.plan
        report = self.state.qa_report
        if plan is None or report is None or not self._is_rpa_build(build_dir):
            self.state.production_review = None
            return True

        remaining = self._remaining_budget()
        if remaining is not None and remaining <= 0:
            issue = (
                "Production review skipped because the build cost budget was reached."
            )
            self.state.production_review = ProductionReview(
                passed=False, issues=[issue]
            )
            report.passed = False
            _append_note(report, issue)
            return False

        _emit_progress(self.state, "production_review_started")
        try:
            review = await cc_agent.run_reviewer(
                cwd=build_dir,
                prompt=_production_review_prompt(self.state),
                budget_usd=remaining,
                on_usage=self._record_executor_usage,
            )
        except cc_agent.CCBudgetExceeded as exc:
            issue = (
                "Production review stopped at the cost budget "
                f"(est. ${exc.cost_usd:.2f} spent this review)."
            )
            review = ProductionReview(passed=False, issues=[issue])
            completed = False
        except Exception as exc:  # noqa: BLE001 — fail certification, preserve archive
            issue = f"Production review could not complete: {exc}"
            review = ProductionReview(passed=False, issues=[issue])
            completed = False
        else:
            completed = True

        if review.passed and review.issues:
            review.passed = False
            review.issues.insert(
                0, "Reviewer returned issues while marking the package as passed."
            )
        elif not review.passed and not review.issues:
            review.issues.append(
                "Production reviewer marked the package as failed without a concrete issue."
            )
            completed = False
        self.state.production_review = review
        if not review.passed:
            report.passed = False
            _append_note(
                report,
                "Production wiring review failed:\n"
                + "\n".join(f"- {issue}" for issue in review.issues),
            )
        _emit_progress(
            self.state,
            "production_review_completed",
            passed=review.passed,
            issues=review.issues,
        )
        return completed

    async def _certify_and_repair(self, build_dir: str) -> None:
        """Require deterministic QA plus the RPA semantic gate within one repair cap."""
        attempts = self._max_final_qa_repairs()
        plan = self.state.plan
        if plan is None:
            return

        while self.state.qa_report is not None:
            if self.state.qa_report.passed:
                review_completed = await self._apply_production_review(build_dir)
                if not review_completed or self.state.qa_report.passed:
                    return

            attempt = self.state.final_qa_repair_attempts + 1
            if attempt > attempts:
                break
            remaining = self._remaining_budget()
            if remaining is not None and remaining <= 0:
                _append_note(
                    self.state.qa_report,
                    "Skipped QA repair: cost budget already reached.",
                )
                return
            self.state.final_qa_repair_attempts += 1
            _emit_progress(
                self.state,
                "final_qa_repair_started",
                attempt=attempt,
                max_attempts=attempts,
            )
            try:
                await cc_agent.run_executor(
                    cwd=build_dir,
                    prompt=_repair_prompt(
                        self.state,
                        plan,
                        self.state.qa_report,
                        attempt,
                        attempts,
                    ),
                    effort=cc_agent.REPAIR_EFFORT,
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
                _append_note(
                    self.state.qa_report, f"Repair attempt {attempt} errored: {exc}"
                )
                return
            self._apply_safe_generated_fixes(build_dir)
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
            "phase": self.state.phase,
            "final_qa_repair_attempts": self.state.final_qa_repair_attempts,
            "llm_usage": self.state.llm_usage,
        }
        if self.state.approved_spec_hash:
            payload["approved_spec_hash"] = self.state.approved_spec_hash
        if self.state.package_results:
            payload["package_results"] = [
                result.model_dump(mode="json") for result in self.state.package_results
            ]
        if self.state.current_failure:
            payload["current_failure"] = self.state.current_failure.model_dump(
                mode="json"
            )
        if build_dir:
            payload["build_dir"] = build_dir
        if self.state.preflight_qa_report:
            payload["preflight_qa_report"] = self.state.preflight_qa_report.model_dump(
                mode="json"
            )
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
            payload["project_archive"] = self.state.project_archive.model_dump(
                mode="json"
            )
        if self.state.quarantine_report:
            payload["quarantine_report"] = self.state.quarantine_report.model_dump(
                mode="json"
            )
        if self.state.quarantine_archive:
            payload["quarantine_archive"] = self.state.quarantine_archive.model_dump(
                mode="json"
            )
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
        if report.spec_hash:
            lines.append(f"- Approved spec: `{report.spec_hash}`")
        if report.package_id:
            lines.append(f"- Work package: `{report.package_id}`")
        if self.state.project_archive:
            lines.append(f"- Archive: {self.state.project_archive.local_path}")
            if self.state.project_archive.url:
                lines.append(f"- Download: {self.state.project_archive.url}")
        sections = [
            ("Integration Notes", report.integration_notes),
            ("Lint Output", report.lint_output),
            ("MyPy Output", report.type_output),
            ("Test Output", report.test_output),
        ]
        for title, value in sections:
            if not value:
                continue
            lines.extend(
                ["", f"## {title}", "", "```text", _markdown_excerpt(value), "```"]
            )
        if report.issues:
            lines.extend(["", "## Blocking issues", ""])
            lines.extend(
                f"- [{issue.owner}] {issue.message}"
                + (f" — {issue.evidence}" if issue.evidence else "")
                for issue in report.issues
                if issue.blocking
            )
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
