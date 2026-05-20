import base64
import zipfile
from pathlib import Path
from types import SimpleNamespace

import codebuilder.main as main
from codebuilder.main import CodebuilderFlow
from codebuilder.runtime_qa import (
    run_deterministic_review,
    run_final_qa,
    run_full_architecture_gate,
    run_rpa_deterministic_gate,
)
from codebuilder.schemas import ArtifactRef, CodeArtifact, Plan, QAReport, SubTask
from codebuilder.tools import attachment_tool
from codebuilder.tools.lint_runner_tool import LintRunnerTool, TestRunnerTool as CodebuilderTestRunnerTool


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


def test_deterministic_review_accepts_workspace_content_without_echo(tmp_path: Path) -> None:
    target = tmp_path / "hello.py"
    target.write_text('def greet() -> str:\n    return "hello"\n', encoding="utf-8")
    artifact = CodeArtifact(
        subtask_id="s1",
        file_path="hello.py",
        language="python",
    )

    review = run_deterministic_review(
        _subtask("hello.py"),
        artifact,
        str(tmp_path),
        lint_runner=FakeTool("PASS"),
    )

    assert review.result.passed is True


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


def test_deterministic_review_rejects_lint_skip_for_production_file(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "hello.py", 'print("hello")\n')

    review = run_deterministic_review(
        _subtask("hello.py"),
        artifact,
        str(tmp_path),
        lint_runner=FakeTool("SKIP: ruff not installed in the runtime"),
    )

    assert review.needs_fallback is False
    assert review.result.passed is False
    assert "required quality gate skipped" in "\n".join(review.result.issues)


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


def test_deterministic_review_rejects_test_skip_for_test_file(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "tests/test_ok.py", "def test_ok():\n    assert True\n")

    review = run_deterministic_review(
        _subtask("tests/test_ok.py"),
        artifact,
        str(tmp_path),
        lint_runner=FakeTool("PASS"),
        test_runner=FakeTool("SKIP: no tests collected under this path."),
    )

    assert review.needs_fallback is False
    assert review.result.passed is False
    assert "required quality gate skipped" in "\n".join(review.result.issues)


def test_final_qa_fails_when_pytest_collects_no_tests() -> None:
    qa = run_final_qa(
        "/unused",
        lint_runner=FakeTool("PASS"),
        test_runner=FakeTool("SKIP: no tests collected under this path."),
    )

    assert qa.passed is False
    assert "Tests were not executed" in qa.integration_notes


def test_architecture_gate_fails_missing_rpa_structure(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (tmp_path / "src/demo").mkdir(parents=True)
    (tmp_path / "src/demo/__init__.py").write_text("", encoding="utf-8")

    result = run_rpa_deterministic_gate(str(tmp_path))

    assert result.passed is False
    assert "orchestrator" in "\n".join(result.issues)
    assert "producer" in "\n".join(result.issues)
    assert "consumer" in "\n".join(result.issues)


def _empty_plan(domain: str = "") -> Plan:
    return Plan(
        project_name="demo",
        mode="new_project",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="t",
                description="d",
                file_path="src/demo/__init__.py",
                test_criteria="exists",
            )
        ],
        domain=domain,
    )


def test_full_architecture_gate_skips_when_plan_has_no_domain(tmp_path: Path) -> None:
    result = run_full_architecture_gate(str(tmp_path), _empty_plan(domain=""))

    assert result.passed is True
    assert result.subtask_id == "architecture_gate"
    assert any("did not declare a domain" in s for s in result.suggestions)


def test_full_architecture_gate_skips_unknown_domain(tmp_path: Path) -> None:
    result = run_full_architecture_gate(str(tmp_path), _empty_plan(domain="python-package"))

    assert result.passed is True
    assert any("No architecture gate registered" in s for s in result.suggestions)


def test_full_architecture_gate_runs_rpa_when_domain_matches(tmp_path: Path) -> None:
    # Empty workspace + rpa domain → deterministic gate runs and fails fast.
    result = run_full_architecture_gate(str(tmp_path), _empty_plan(domain="rpa"))

    assert result.passed is False
    assert result.subtask_id == "architecture_gate"
    assert any("orchestrator" in issue for issue in result.issues)


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


def test_emit_progress_posts_configured_webhook(monkeypatch) -> None:
    flow = CodebuilderFlow()
    flow.state.project_name = "demo"
    flow.state.project_key = "demo-key"
    calls: list[dict] = []

    def fake_post(url: str, json: dict, headers: dict, timeout: int):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return SimpleNamespace(status_code=200)

    monkeypatch.setenv("CODEBUILDER_PROGRESS_WEBHOOK", "https://example.test/progress")
    monkeypatch.setenv("CODEBUILDER_PROGRESS_WEBHOOK_SECRET", "secret")
    monkeypatch.setattr(main.requests, "post", fake_post)

    main._emit_progress(flow.state, "subtask_started", subtask_id="s1")

    assert calls == [
        {
            "url": "https://example.test/progress",
            "json": {
                "event_type": "subtask_started",
                "session_id": flow.state.session_id,
                "flow_id": flow.state.id,
                "job_id": flow.state.id,
                "project_name": "demo",
                "project_key": "demo-key",
                "subtask_id": "s1",
            },
            "headers": {
                "Content-Type": "application/json",
                "X-Codebuilder-Progress-Secret": "secret",
            },
            "timeout": main.PROGRESS_WEBHOOK_TIMEOUT_SECONDS,
        }
    ]


def test_zip_build_excludes_tool_cache_dirs(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "app.py").write_text("print('ok')\n", encoding="utf-8")
    cache_dir = build_dir / ".ruff_cache"
    cache_dir.mkdir()
    (cache_dir / "CACHEDIR.TAG").write_text("cache", encoding="utf-8")

    zip_path = main._zip_build(str(build_dir), tmp_path, "demo")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "demo/app.py" in names
    assert all(".ruff_cache" not in name for name in names)


def test_attachment_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.py", "print('bad')\n")

    attachment = {
        "kind": "zip",
        "name": "bad.zip",
        "content_b64": base64.b64encode(archive.read_bytes()).decode(),
    }

    try:
        attachment_tool.materialize([attachment], str(tmp_path))
    except ValueError as exc:
        assert "Unsafe zip member" in str(exc)
    else:
        raise AssertionError("zip traversal attachment was accepted")


def test_attachment_file_name_cannot_escape_inputs_dir(tmp_path: Path) -> None:
    attachment = {
        "kind": "image",
        "name": "../escape.png",
        "content_b64": base64.b64encode(b"not really png").decode(),
    }

    records = attachment_tool.materialize([attachment], str(tmp_path))

    assert records[0]["path"] == "inputs/escape.png"
    assert (tmp_path / "inputs/escape.png").is_file()
    assert not (tmp_path / "escape.png").exists()


def test_lint_and_test_runner_reject_paths_outside_workspace(tmp_path: Path) -> None:
    lint = LintRunnerTool(workspace_dir=str(tmp_path))
    tests = CodebuilderTestRunnerTool(workspace_dir=str(tmp_path))

    assert "escapes workspace" in lint._run("../outside.py")
    assert "escapes workspace" in tests._run("../outside.py")


def test_finalize_repairs_failed_final_qa_and_returns_payload(monkeypatch, tmp_path: Path) -> None:
    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "executing"
    monkeypatch.setenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", "1")

    qa_calls: list[str] = []

    def fake_run_final_qa(build_dir: str) -> QAReport:
        qa_calls.append(build_dir)
        if len(qa_calls) == 1:
            return QAReport(
                passed=False,
                lint_output="PASS",
                test_output="FAILED tests/test_app.py::test_demo",
                integration_notes="Tests failed.",
            )
        return QAReport(passed=True, lint_output="PASS", test_output="PASS", integration_notes="Clean.")

    class FakeWriterCrew:
        def __init__(self, workspace_dir: str):
            self.workspace_dir = workspace_dir

        def repair_crew(self):
            workspace_dir = self.workspace_dir

            class FakeCrew:
                def kickoff(self, inputs: dict):
                    target = Path(workspace_dir) / "app.py"
                    target.write_text('def fixed() -> bool:\n    return True\n', encoding="utf-8")
                    return SimpleNamespace(
                        pydantic=CodeArtifact(
                            subtask_id="final_qa_repair",
                            file_path="app.py",
                            language="python",
                        )
                    )

            return FakeCrew()

    monkeypatch.setattr(main, "run_final_qa", fake_run_final_qa)
    monkeypatch.setattr(main, "WriterCrew", FakeWriterCrew)
    monkeypatch.setattr(
        main,
        "upload_workspace",
        lambda build_dir, prefix: [
            {"file_path": "app.py", "size": 37, "url": "https://example.test/app.py"}
        ],
    )
    monkeypatch.setattr(main, "upload_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.history, "record", lambda state: None)

    payload = flow.finalize(None)

    assert qa_calls == [str(tmp_path), str(tmp_path)]
    assert flow.state.status == "done"
    assert flow.state.final_qa_repair_attempts == 1
    assert payload["qa_report"]["passed"] is True
    assert payload["artifact_urls"] == [
        {"file_path": "app.py", "size": 37, "url": "https://example.test/app.py"}
    ]


def test_finalize_returns_artifacts_when_final_qa_still_fails(monkeypatch, tmp_path: Path) -> None:
    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.status = "executing"
    monkeypatch.setenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", "1")

    def fake_run_final_qa(build_dir: str) -> QAReport:
        return QAReport(
            passed=False,
            lint_output="PASS",
            test_output="FAILED tests/test_app.py::test_demo",
            integration_notes="Tests failed.",
        )

    class FakeWriterCrew:
        def __init__(self, workspace_dir: str):
            self.workspace_dir = workspace_dir

        def repair_crew(self):
            workspace_dir = self.workspace_dir

            class FakeCrew:
                def kickoff(self, inputs: dict):
                    target = Path(workspace_dir) / "app.py"
                    target.write_text('def still_broken() -> bool:\n    return False\n', encoding="utf-8")
                    return SimpleNamespace(
                        pydantic=CodeArtifact(
                            subtask_id="final_qa_repair",
                            file_path="app.py",
                            language="python",
                        )
                    )

            return FakeCrew()

    monkeypatch.setattr(main, "run_final_qa", fake_run_final_qa)
    monkeypatch.setattr(main, "WriterCrew", FakeWriterCrew)
    monkeypatch.setattr(
        main,
        "upload_workspace",
        lambda build_dir, prefix: [
            {"file_path": "app.py", "size": 45, "url": "https://example.test/app.py"}
        ],
    )
    monkeypatch.setattr(main, "upload_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.history, "record", lambda state: None)

    payload = flow.finalize(None)

    assert flow.state.status == "failed"
    assert flow.state.final_qa_repair_attempts == 1
    assert payload["qa_report"]["passed"] is False
    assert "still failing after 1 writer repair attempt" in payload["qa_report"]["integration_notes"]
    assert payload["artifact_urls"] == [
        {"file_path": "app.py", "size": 45, "url": "https://example.test/app.py"}
    ]
