"""Provision and locate the generated project's own virtualenv for QA.

The generated projects follow a ``src/`` layout and declare their own
dependencies (the RPA standard mandates pytest/ruff/mypy/pyinstaller dev
deps), so lint and tests can only meaningfully run inside the *project's*
environment — running them with the orchestrator's interpreter fails at
pytest collection with ``ModuleNotFoundError`` for the package itself and
for every third-party dependency. ``ensure_project_env`` materializes that
environment with ``uv sync`` (which also validates that the generated
``pyproject.toml`` is actually installable), and ``project_python`` returns
the interpreter QA tools should invoke.

Provisioning is best-effort: when ``uv`` is unavailable, the env var
``CODEBUILDER_PROVISION_PROJECT_ENV`` disables it, or the workspace has no
``pyproject.toml``, callers fall back to the orchestrator's interpreter
(the pre-existing behavior).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

SYNC_TIMEOUT_SECONDS = 300
_HASH_MARKER = ".codebuilder-pyproject-hash"


def _venv_python(build_dir: Path) -> Path:
    if os.name == "nt":
        return build_dir / ".venv" / "Scripts" / "python.exe"
    return build_dir / ".venv" / "bin" / "python"


def provisioning_enabled() -> bool:
    raw = os.environ.get("CODEBUILDER_PROVISION_PROJECT_ENV", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def project_python(workspace_dir: str) -> str:
    """Interpreter QA tools should use: the project venv when provisioned,
    else the orchestrator's own interpreter."""
    python = _venv_python(Path(workspace_dir).resolve())
    return str(python) if python.is_file() else sys.executable


def ensure_project_env(workspace_dir: str) -> str:
    """Sync ``<workspace>/.venv`` from the workspace's ``pyproject.toml``.

    Returns ``""`` on success or benign no-op (no pyproject, uv missing,
    provisioning disabled, already in sync), otherwise the ``uv sync`` error
    output — which doubles as the "generated project is not installable"
    QA signal for new-project jobs.
    """
    build_dir = Path(workspace_dir).resolve()
    pyproject = build_dir / "pyproject.toml"
    if not provisioning_enabled() or not pyproject.is_file():
        return ""
    uv = shutil.which("uv")
    if uv is None:
        log.warning("uv not on PATH; QA falls back to the orchestrator's interpreter")
        return ""

    digest = hashlib.sha256(pyproject.read_bytes()).hexdigest()
    marker = build_dir / ".venv" / _HASH_MARKER
    try:
        if (
            marker.is_file()
            and marker.read_text(encoding="utf-8").strip() == digest
            and _venv_python(build_dir).is_file()
        ):
            return ""
    except OSError:
        pass

    # uv project commands ignore an inherited VIRTUAL_ENV (the orchestrator's
    # venv when launched via `uv run`) but warn loudly; drop it for clean output.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    try:
        proc = subprocess.run(
            [uv, "sync", "--no-progress"],
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            timeout=SYNC_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"uv sync timed out after {SYNC_TIMEOUT_SECONDS}s"
    except OSError as exc:
        log.warning("uv sync could not be spawned: %s", exc)
        return ""

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return output or f"uv sync exit {proc.returncode}"

    if _venv_python(build_dir).is_file():
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(digest, encoding="utf-8")
        except OSError:
            pass
    return ""
