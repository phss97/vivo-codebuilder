"""Per-project history log.

A standalone SQLite table, independent of CrewAI's ``@persist()`` flow state.
Records one row per completed/failed job, keyed by a stable ``project_key`` so
that subsequent runs against the same repo/project can surface prior work to
the planner.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codebuilder.schemas import CodebuilderState


log = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("CODEBUILDER_HISTORY_DB", "./data/codebuilder_history.db")).resolve()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_key TEXT NOT NULL,
    project_name TEXT NOT NULL,
    job_id TEXT NOT NULL UNIQUE,
    mode TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_json TEXT,
    qa_report_json TEXT,
    files_touched TEXT,
    reviewer_issues TEXT,
    patch TEXT
);
CREATE INDEX IF NOT EXISTS idx_project_history_key
    ON project_history(project_key, created_at DESC);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = _SLUG_RE.sub("-", value).strip("-")
    return value


def canonicalize_git_url(url: str) -> str:
    """Reduce a git URL to ``host/org/repo`` for stable keying."""
    if not url:
        return ""
    s = url.strip().lower()
    # scp-style: git@github.com:org/repo.git
    if s.startswith("git@") and ":" in s and "://" not in s:
        host, _, path = s[4:].partition(":")
        s = f"{host}/{path}"
    else:
        for scheme in ("https://", "http://", "ssh://", "git://"):
            if s.startswith(scheme):
                s = s[len(scheme) :]
                break
        # strip user@ if still present
        if "@" in s.split("/", 1)[0]:
            s = s.split("@", 1)[1]
    s = s.rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]
    return s


def project_key_from(state: "CodebuilderState") -> str:
    """Derive the stable project key from state.

    ``patch_existing`` intent: first git attachment's URL, canonicalised.
    Otherwise: slug of ``project_name``. Returns ``""`` when neither is usable.
    """
    for att in state.attachments:
        if att.kind == "git" and att.uri:
            key = canonicalize_git_url(att.uri)
            if key:
                return key
    if state.project_name:
        return _slug(state.project_name)
    return ""


def record(state: "CodebuilderState") -> None:
    """Upsert a history row for this job. Called from ``finalize()``."""
    if not state.project_key:
        log.info("history.record skipped: no project_key on state")
        return

    plan_json = state.plan.model_dump_json() if state.plan else None
    qa_json = state.qa_report.model_dump_json() if state.qa_report else None
    mode = state.plan.mode if state.plan else ("patch_existing" if plan_json else "new_project")
    files_touched = json.dumps(sorted({a.file_path for a in state.artifacts if a.file_path}))
    reviewer_issues = json.dumps(
        [issue for rr in state.review_results for issue in (rr.issues or [])]
    )

    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO project_history (
                    project_key, project_name, job_id, mode, created_at, status,
                    plan_json, qa_report_json, files_touched, reviewer_issues, patch
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    project_key = excluded.project_key,
                    project_name = excluded.project_name,
                    mode = excluded.mode,
                    status = excluded.status,
                    plan_json = excluded.plan_json,
                    qa_report_json = excluded.qa_report_json,
                    files_touched = excluded.files_touched,
                    reviewer_issues = excluded.reviewer_issues,
                    patch = excluded.patch
                """,
                (
                    state.project_key,
                    state.project_name or (state.plan.project_name if state.plan else ""),
                    state.id,
                    mode,
                    datetime.now(timezone.utc).isoformat(),
                    state.status,
                    plan_json,
                    qa_json,
                    files_touched,
                    reviewer_issues,
                    state.patch or "",
                ),
            )
    except sqlite3.Error as exc:
        log.warning("history.record failed for job %s: %s", state.id, exc)


def recent(project_key: str, limit: int = 3) -> list[dict]:
    if not project_key:
        return []
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT project_name, job_id, mode, created_at, status,
                       qa_report_json, files_touched, reviewer_issues
                FROM project_history
                WHERE project_key = ?
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (project_key, limit),
            ).fetchall()
    except sqlite3.Error as exc:
        log.warning("history.recent failed for key %s: %s", project_key, exc)
        return []
    return [dict(r) for r in rows]


def summarize_for_planner(project_key: str, limit: int = 3) -> str:
    """Compact markdown summary of prior runs, fed into the planner prompt."""
    rows = recent(project_key, limit=limit)
    if not rows:
        return ""

    lines: list[str] = []
    for idx, row in enumerate(rows, start=1):
        files = json.loads(row.get("files_touched") or "[]")
        issues = json.loads(row.get("reviewer_issues") or "[]")
        qa_blob = row.get("qa_report_json")
        integration_notes = ""
        qa_passed: bool | None = None
        if qa_blob:
            try:
                qa = json.loads(qa_blob)
                integration_notes = (qa.get("integration_notes") or "").strip()
                qa_passed = qa.get("passed")
            except json.JSONDecodeError:
                pass

        lines.append(
            f"### Run {idx} — {row['created_at']} ({row['mode']}, status={row['status']})"
        )
        if qa_passed is not None:
            lines.append(f"- QA passed: {qa_passed}")
        if files:
            shown = ", ".join(files[:10])
            more = f" (+{len(files) - 10} more)" if len(files) > 10 else ""
            lines.append(f"- Files touched: {shown}{more}")
        if issues:
            lines.append("- Reviewer issues (top 3):")
            for issue in issues[:3]:
                lines.append(f"  - {issue}")
        if integration_notes:
            lines.append(f"- QA notes: {integration_notes}")
        lines.append("")

    return "\n".join(lines).rstrip()
