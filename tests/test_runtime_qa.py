from pathlib import Path

from codebuilder.main import run_deterministic_review, run_final_qa
from codebuilder.schemas import ArtifactRef, CodeArtifact, QAReport, SubTask


class FakeTool:
    def __init__(self, output: str):
        self.output = output
        self.calls: list[str] = []

    def _run(self, path: str = ".") -> str:
        self.calls.append(path)
        return self.output


def _subtask(path: str = "hello.py") -> SubTask:
    return SubTask(
        id="s1",
        title="Write file",
        description="Create the requested file.",
        file_path=path,
        test_criteria="Deterministic checks pass.",
    )


def _artifact(tmp_path: Path, path: str, content: str) -> CodeArtifact:
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return CodeArtifact(
        subtask_id="s1",
        file_path=path,
        content=content,
        language="python",
    )


def _artifact_ref_schema(schema: dict) -> dict:
    items = schema["properties"]["artifact_urls"]["items"]
    ref = items.get("$ref")
    if not ref:
        return items
    _, _, name = ref.rpartition("/")
    return schema["$defs"][name]


def test_qa_report_artifact_urls_schema_is_closed() -> None:
    schema = QAReport.model_json_schema()

    assert schema["additionalProperties"] is False
    assert _artifact_ref_schema(schema)["additionalProperties"] is False


def test_deterministic_review_passes_good_file(tmp_path: Path) -> None:
    content = 'def greet() -> str:\n    return "hello"\n'
    artifact = _artifact(tmp_path, "hello.py", content)
    lint = FakeTool("PASS")

    review = run_deterministic_review(
        _subtask("hello.py"),
        artifact,
        str(tmp_path),
        lint_runner=lint,
    )

    assert review.result.passed is True
    assert lint.calls == ["hello.py"]


def test_deterministic_review_rejects_wrong_path(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "other.py", 'print("wrong")\n')

    review = run_deterministic_review(_subtask("hello.py"), artifact, str(tmp_path))

    assert review.result.passed is False
    assert "does not match planned path" in review.result.issues[0]


def test_deterministic_review_rejects_missing_file(tmp_path: Path) -> None:
    artifact = CodeArtifact(
        subtask_id="s1",
        file_path="hello.py",
        content='print("missing")\n',
        language="python",
    )

    review = run_deterministic_review(_subtask("hello.py"), artifact, str(tmp_path))

    assert review.result.passed is False
    assert "was not written" in "\n".join(review.result.issues)


def test_deterministic_review_rejects_placeholder_content(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "hello.py", "# TODO: implement\n")

    review = run_deterministic_review(_subtask("hello.py"), artifact, str(tmp_path))

    assert review.result.passed is False
    assert "placeholder" in "\n".join(review.result.issues)


def test_deterministic_review_rejects_lint_failure(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "hello.py", 'print("hello")\n')

    review = run_deterministic_review(
        _subtask("hello.py"),
        artifact,
        str(tmp_path),
        lint_runner=FakeTool("F821 undefined name 'x'"),
    )

    assert review.result.passed is False
    assert "ruff failed" in "\n".join(review.result.issues)


def test_deterministic_review_rejects_test_file_pytest_failure(tmp_path: Path) -> None:
    artifact = _artifact(
        tmp_path,
        "tests/test_bad.py",
        "def test_bad():\n    assert False\n",
    )
    test_tool = FakeTool("FAILED tests/test_bad.py::test_bad")

    review = run_deterministic_review(
        _subtask("tests/test_bad.py"),
        artifact,
        str(tmp_path),
        lint_runner=FakeTool("PASS"),
        test_runner=test_tool,
    )

    assert review.result.passed is False
    assert test_tool.calls == ["tests/test_bad.py"]
    assert "pytest failed" in "\n".join(review.result.issues)


def test_deterministic_review_marks_skip_for_fallback(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "hello.py", 'print("hello")\n')

    review = run_deterministic_review(
        _subtask("hello.py"),
        artifact,
        str(tmp_path),
        lint_runner=FakeTool("SKIP: ruff not installed in the runtime"),
    )

    assert review.needs_fallback is True
    assert review.result.passed is False
    assert "SKIP:" in "\n".join(review.result.issues)


def test_final_qa_builds_report_and_preserves_artifact_refs() -> None:
    qa = run_final_qa(
        "/unused",
        lint_runner=FakeTool("PASS"),
        test_runner=FakeTool("PASS\n1 passed"),
        artifact_urls=[{"file_path": "hello.py", "size": 12, "url": "https://example.test/hello.py"}],
    )

    assert qa.passed is True
    assert qa.artifact_urls == [
        ArtifactRef(file_path="hello.py", size=12, url="https://example.test/hello.py")
    ]
    assert "Deterministic QA" in qa.integration_notes
