from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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
        return target.read_text(encoding="utf-8", errors="replace")


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
    description: str = "List files under a directory in the job workspace, recursively."
    args_schema: Type[BaseModel] = _ListInput
    workspace_dir: str

    def _run(self, path: str = ".") -> str:
        target = resolve_within(self.workspace_dir, path)
        if not target.exists():
            return f"ERROR: path not found: {path}"
        root = Path(self.workspace_dir).resolve()
        entries = []
        for p in sorted(target.rglob("*")):
            if p.is_file():
                entries.append(str(p.relative_to(root)))
        return "\n".join(entries) if entries else "(empty)"
