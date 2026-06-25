"""A `modify` artifact must not silently lose content — a truncated read written
back as the file, or a large file rewritten shorter, is rejected in review."""

from pathlib import Path

from codebuilder.runtime_qa import MODIFY_PREIMAGE_LIMIT, run_deterministic_review
from codebuilder.schemas import CodeArtifact, FileSkeleton, SubTask


class FakeTool:
    def __init__(self, out: str):
        self.out = out

    def _run(self, path: str = ".") -> str:
        return self.out


def _modify_subtask(path: str) -> SubTask:
    return SubTask(
        id="s1",
        title="t",
        description="d",
        files=[FileSkeleton(path=path, purpose="p", change_type="modify")],
        test_criteria="c",
    )


def _write(tmp_path: Path, path: str, content: str) -> CodeArtifact:
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return CodeArtifact(subtask_id="s1", file_path=path, content=content, language="python")


def test_review_rejects_truncation_marker(tmp_path: Path) -> None:
    content = "def f():\n    return 1\n\n[truncated 9999 chars]\n"
    art = _write(tmp_path, "mod.py", content)
    review = run_deterministic_review(
        _modify_subtask("mod.py"),
        art,
        str(tmp_path),
        existing_snapshot="def f():\n    return 0\n",
        lint_runner=FakeTool("PASS"),
    )
    assert not review.result.passed
    assert any("truncat" in i.lower() for i in review.result.issues)


def test_review_rejects_large_modify_shrink(tmp_path: Path) -> None:
    preimage = "x = 1\n" * (MODIFY_PREIMAGE_LIMIT // 3)  # > limit chars
    assert len(preimage) > MODIFY_PREIMAGE_LIMIT
    art = _write(tmp_path, "big.py", "x = 1\n" * 50)  # much shorter
    review = run_deterministic_review(
        _modify_subtask("big.py"),
        art,
        str(tmp_path),
        existing_snapshot=preimage,
        lint_runner=FakeTool("PASS"),
    )
    assert not review.result.passed
    assert any(
        ("lost" in i.lower() or "preserv" in i.lower() or "large" in i.lower())
        for i in review.result.issues
    )


def test_review_allows_normal_modify_shrink(tmp_path: Path) -> None:
    # small file (< limit) that legitimately shrank must NOT be flagged.
    art = _write(tmp_path, "ok.py", "y = 2\n" * 10)
    review = run_deterministic_review(
        _modify_subtask("ok.py"),
        art,
        str(tmp_path),
        existing_snapshot="x = 1\n" * 50,
        lint_runner=FakeTool("PASS"),
    )
    assert review.result.passed, review.result.issues
