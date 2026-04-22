from pathlib import Path

from git import Repo


def clone(url: str, dest: str) -> str:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    Repo.clone_from(url, dest_path)
    return str(dest_path)


def init_and_commit(workspace_dir: str, message: str = "codebuilder baseline") -> str:
    repo = Repo.init(workspace_dir)
    repo.git.add(A=True)
    if repo.is_dirty(untracked_files=True):
        repo.index.commit(message)
    return str(Path(workspace_dir).resolve())


def diff(repo_dir: str) -> str:
    repo = Repo(repo_dir)
    repo.git.add(A=True)
    staged = repo.git.diff("--cached")
    unstaged = repo.git.diff()
    return "\n".join(p for p in (staged, unstaged) if p)
