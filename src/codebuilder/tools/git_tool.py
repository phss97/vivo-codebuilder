from pathlib import Path

from git import Repo

# QA-harness artifacts that must never enter baselines or patch diffs. The
# project env provisioning (`uv sync`) drops .venv/ and uv.lock into the build
# dir, and `diff()`/`init_and_commit()` stage with `git add -A`, which would
# otherwise sweep thousands of interpreter files into the deliverable diff.
# Written to .git/info/exclude (not .gitignore) so the user's files stay
# untouched; already-tracked paths are unaffected.
_HARNESS_EXCLUDES = (
    ".venv/",
    "uv.lock",
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".DS_Store",
)


def _ensure_harness_excludes(repo: Repo) -> None:
    exclude_path = Path(repo.git_dir) / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    present = {line.strip() for line in existing.splitlines()}
    missing = [entry for entry in _HARNESS_EXCLUDES if entry not in present]
    if not missing:
        return
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    exclude_path.write_text(existing + prefix + "\n".join(missing) + "\n", encoding="utf-8")


def clone(url: str, dest: str) -> str:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    Repo.clone_from(url, dest_path)
    return str(dest_path)


def init_and_commit(workspace_dir: str, message: str = "codebuilder baseline") -> str:
    repo = Repo.init(workspace_dir)
    _ensure_harness_excludes(repo)
    repo.git.add(A=True)
    if repo.is_dirty(untracked_files=True):
        repo.index.commit(message)
    return str(Path(workspace_dir).resolve())


def diff(repo_dir: str) -> str:
    repo = Repo(repo_dir)
    _ensure_harness_excludes(repo)
    repo.git.add(A=True)
    staged = repo.git.diff("--cached")
    unstaged = repo.git.diff()
    return "\n".join(p for p in (staged, unstaged) if p)
