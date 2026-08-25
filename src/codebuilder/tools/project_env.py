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

SYNC_TIMEOUT_SECONDS = 2400
_HASH_MARKER = ".codebuilder-pyproject-hash"
# Index configuration belonging to codebuilder's own deployment, never to the
# customer project it is building.
_INDEX_ENV = frozenset(
    {
        "UV_INDEX",
        "UV_DEFAULT_INDEX",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_FIND_LINKS",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
    }
)


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


def ensure_project_env(workspace_dir: str, *, locked: bool = False) -> str:
    """Sync ``<workspace>/.venv`` from the workspace's ``pyproject.toml``.

    ``locked=True`` validates the committed lock without allowing uv to change it.

    Returns ``""`` on success or benign no-op (no pyproject, provisioning
    disabled, already in sync), otherwise the ``uv sync`` error
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
        return "uv is not installed; project environment could not be synchronized"

    lockfile = build_dir / "uv.lock"
    digest_input = pyproject.read_bytes()
    if lockfile.is_file():
        digest_input += lockfile.read_bytes()
    digest = hashlib.sha256(digest_input).hexdigest()
    marker_value = f"locked:{digest}" if locked else digest
    marker = build_dir / ".venv" / _HASH_MARKER
    try:
        if marker.is_file() and _venv_python(build_dir).is_file():
            current = marker.read_text(encoding="utf-8").strip()
            if current == marker_value or (
                not locked and current == f"locked:{digest}"
            ):
                return ""
    except OSError:
        pass

    # uv project commands ignore an inherited VIRTUAL_ENV (the orchestrator's
    # venv when launched via `uv run`) but warn loudly; drop it for clean output.
    # Drop our own package-index config too: inherited wholesale it silently
    # redirects the *customer* project's resolution at codebuilder's private
    # mirror, so any package that mirror does not carry 401s instead of
    # resolving from public PyPI.
    # ponytail: strip, don't override — the customer's own pyproject/uv.lock
    # index declaration stays authoritative, which an explicit default would
    # outrank.
    dropped = {k for k in os.environ if k == "VIRTUAL_ENV" or k in _INDEX_ENV}
    if dropped - {"VIRTUAL_ENV"}:
        log.info(
            "dropped orchestrator index config for customer uv sync: %s",
            sorted(dropped - {"VIRTUAL_ENV"}),
        )
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    try:
        command = [uv, "sync", "--no-progress"]
        if locked:
            command.append("--locked")
        proc = subprocess.run(
            command,
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
        return f"uv sync could not be spawned: {exc}"

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return output or f"uv sync exit {proc.returncode}"

    if _venv_python(build_dir).is_file():
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(marker_value, encoding="utf-8")
        except OSError:
            pass
    return ""
