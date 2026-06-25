from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


DEFAULT_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
DEFAULT_SKIP_FILES = {".DS_Store"}
DEFAULT_SKIP_SUFFIXES = {".pyc", ".pyo", ".zip"}
DEFAULT_MAX_LIST_ENTRIES = 400
DEFAULT_MAX_LIST_CHARS = 20_000
# Cap a single file read so the writer's agentic loop can't pull an arbitrarily
# large file into context (a runaway token sink). The real dependency APIs the
# writer needs are injected into its prompt via the symbol contract, so a full
# raw read is rarely necessary; when it is, this keeps it bounded.
MAX_READ_CHARS = 20_000


def resolve_within(workspace_dir: str, rel_path: str) -> Path:
    root = Path(workspace_dir).resolve()
    target = (root / rel_path).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Path '{rel_path}' escapes workspace '{workspace_dir}'")
    return target


class _ReadInput(BaseModel):
    path: str = Field(description="Relative path within the workspace")


class WorkspaceReadTool(BaseTool):
    name: str = "workspace_read"
    description: str = (
        "Read a file from the job workspace. Path is relative to the workspace root."
    )
    args_schema: Type[BaseModel] = _ReadInput
    workspace_dir: str

    def _run(self, path: str) -> str:
        target = resolve_within(self.workspace_dir, path)
        if not target.is_file():
            return f"ERROR: file not found: {path}"
        content = target.read_text(encoding="utf-8", errors="replace")
        if len(content) > MAX_READ_CHARS:
            omitted = len(content) - MAX_READ_CHARS
            return f"{content[:MAX_READ_CHARS]}\n\n[truncated {omitted} chars]"
        return content


class _WriteInput(BaseModel):
    path: str = Field(description="Relative path within the workspace")
    content: str = Field(description="Full file contents to write")


class WorkspaceWriteTool(BaseTool):
    name: str = "workspace_write"
    description: str = (
        "Write a file into the job workspace. Creates parent directories as needed. "
        "Path is relative to the workspace root."
    )
    args_schema: Type[BaseModel] = _WriteInput
    workspace_dir: str

    def _run(self, path: str, content: str) -> str:
        target = resolve_within(self.workspace_dir, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {target.relative_to(Path(self.workspace_dir).resolve())} ({len(content)} chars)"


class _ListInput(BaseModel):
    path: str = Field(default=".", description="Relative directory path")


class WorkspaceListTool(BaseTool):
    name: str = "workspace_list"
    description: str = (
        "List files under a directory in the job workspace, recursively. "
        "Generated directories and large archive/cache files are omitted."
    )
    args_schema: Type[BaseModel] = _ListInput
    workspace_dir: str

    def _run(self, path: str = ".") -> str:
        target = resolve_within(self.workspace_dir, path)
        if not target.exists():
            return f"ERROR: path not found: {path}"
        root = Path(self.workspace_dir).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return f"ERROR: path escapes workspace: {path}"

        entries: list[str] = []
        char_count = 0
        truncated = False
        for p in self._iter_visible_files(target, root):
            rel = p.relative_to(root).as_posix()
            added_chars = len(rel) + (1 if entries else 0)
            if (
                len(entries) >= DEFAULT_MAX_LIST_ENTRIES
                or char_count + added_chars > DEFAULT_MAX_LIST_CHARS
            ):
                truncated = True
                break
            entries.append(rel)
            char_count += added_chars

        if not entries:
            return "(empty)"
        if truncated:
            entries.append(
                "[truncated: listing capped at "
                f"{DEFAULT_MAX_LIST_ENTRIES} entries / {DEFAULT_MAX_LIST_CHARS} chars; "
                "narrow the path for more detail]"
            )
        return "\n".join(entries)

    def _iter_visible_files(self, directory: Path, root: Path):
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError:
            return

        for child in children:
            rel = child.relative_to(root)
            if _is_skipped_path(rel):
                continue
            if child.is_dir() and not child.is_symlink():
                yield from self._iter_visible_files(child, root)
            elif child.is_file():
                yield child


def _is_skipped_path(rel_path: Path) -> bool:
    if any(part in DEFAULT_SKIP_DIRS for part in rel_path.parts):
        return True
    if rel_path.name in DEFAULT_SKIP_FILES:
        return True
    return rel_path.suffix in DEFAULT_SKIP_SUFFIXES
