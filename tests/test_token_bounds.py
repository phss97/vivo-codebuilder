"""Token-bound guards: cap unbounded workspace reads and repair context.

These were unbounded sinks — a writer could read arbitrarily large files into
its agentic loop, and the repair context dumped every artifact's directory
listing. Both are now capped.
"""

import codebuilder.main as main
from codebuilder.schemas import CodeArtifact
from codebuilder.tools.workspace_tool import MAX_READ_CHARS, WorkspaceReadTool


def test_workspace_read_caps_large_files(tmp_path) -> None:
    (tmp_path / "big.py").write_text("x = 1  # padding line\n" * 5000, encoding="utf-8")
    out = WorkspaceReadTool(workspace_dir=str(tmp_path))._run("big.py")
    assert len(out) <= MAX_READ_CHARS + 200
    assert "[truncated" in out


def test_workspace_read_small_file_unchanged(tmp_path) -> None:
    (tmp_path / "s.py").write_text("a = 1\n", encoding="utf-8")
    assert WorkspaceReadTool(workspace_dir=str(tmp_path))._run("s.py") == "a = 1\n"


def test_repair_context_caps_files(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    artifacts = []
    for i in range(60):
        rel = f"src/m{i}.py"
        (tmp_path / rel).write_text("x = 1\n", encoding="utf-8")
        artifacts.append(CodeArtifact(subtask_id="s", file_path=rel, language="python"))
    ctx = main._repair_workspace_context(str(tmp_path), None, artifacts)
    planned = [line for line in ctx.splitlines() if line.startswith("- src/m")]
    assert 0 < len(planned) <= main.MAX_REPAIR_CONTEXT_FILES
