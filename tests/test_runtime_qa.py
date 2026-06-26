import base64
import json
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

import codebuilder.main as main
from codebuilder import history
from codebuilder.main import CodebuilderFlow
from codebuilder.runtime_qa import (
    persist_bundle_artifact,
    run_deterministic_review,
    run_bundle_deterministic_review,
    run_final_qa,
    run_full_architecture_gate,
    run_rpa_deterministic_gate,
)
from codebuilder.schemas import (
    Attachment,
    ArtifactRef,
    CodeArtifact,
    CodeBundleArtifact,
    CodebuilderState,
    FileSkeleton,
    Plan,
    QAReport,
    ReviewResult,
    SubTask,
)
from codebuilder.tools import attachment_tool
from codebuilder.tools.lint_runner_tool import LintRunnerTool, TestRunnerTool as CodebuilderTestRunnerTool
from codebuilder.tools.workspace_tool import WorkspaceListTool


class FakeTool:
    def __init__(self, output: str):
        self.output = output
        self.calls: list[str] = []

    def _run(self, path: str = ".") -> str:
        self.calls.append(path)
        return self.output


class FakeToolByPath:
    def __init__(self, outputs: dict[str, str], default: str = "PASS"):
        self.outputs = outputs
        self.default = default
        self.calls: list[str] = []

    def _run(self, path: str = ".") -> str:
        self.calls.append(path)
        return self.outputs.get(path, self.default)


def _subtask(path: str = "hello.py") -> SubTask:
    return SubTask(
        id="s1",
        title="Write file",
        description="Create the requested file.",
        files=[FileSkeleton(path=path, purpose="Requested file.")],
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


def test_workspace_list_filters_generated_dirs_and_caps_output(tmp_path: Path) -> None:
    for path in (
        ".git/objects/aa/blob",
        ".venv/lib/site.py",
        "node_modules/pkg/index.js",
        "__pycache__/mod.pyc",
        ".pytest_cache/v/cache",
        "dist/app.exe",
        "build/temp.o",
        "project.zip",
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("noise", encoding="utf-8")

    src = tmp_path / "src"
    src.mkdir()
    for i in range(450):
        (src / f"file_{i:03d}.py").write_text("VALUE = 1\n", encoding="utf-8")

    listing = WorkspaceListTool(workspace_dir=str(tmp_path))._run(".")

    assert "src/file_000.py" in listing
    assert ".git" not in listing
    assert ".venv" not in listing
    assert "node_modules" not in listing
    assert "__pycache__" not in listing
    assert ".pytest_cache" not in listing
    assert "dist/" not in listing
    assert "build/" not in listing
    assert "project.zip" not in listing
    assert "[truncated" in listing
    assert len(listing) <= 20500


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


def test_persist_bundle_rejects_missing_extra_and_duplicate_paths(tmp_path: Path) -> None:
    subtask = SubTask(
        id="s1",
        title="Bundle",
        description="Create bundle.",
        files=[
            FileSkeleton(path="hello.py", purpose="Hello module."),
            FileSkeleton(path="goodbye.py", purpose="Goodbye module."),
        ],
        test_criteria="Deterministic checks pass.",
    )
    bundle = CodeBundleArtifact(
        subtask_id="s1",
        artifacts=[
            CodeArtifact(
                subtask_id="s1",
                file_path="hello.py",
                content='print("hello")\n',
                language="python",
            ),
            CodeArtifact(
                subtask_id="s1",
                file_path="hello.py",
                content='print("dupe")\n',
                language="python",
            ),
            CodeArtifact(
                subtask_id="s1",
                file_path="extra.py",
                content='print("extra")\n',
                language="python",
            ),
        ],
    )

    issues = persist_bundle_artifact(bundle, subtask, str(tmp_path))

    joined = "\n".join(issues)
    assert "missing planned artifact" in joined
    assert "unexpected artifact path" in joined
    assert "duplicate artifact path" in joined
    assert not (tmp_path / "hello.py").exists()
    assert not (tmp_path / "extra.py").exists()


def test_bundle_deterministic_review_passes_each_file(tmp_path: Path) -> None:
    subtask = SubTask(
        id="s1",
        title="Bundle",
        description="Create bundle.",
        files=[
            FileSkeleton(path="hello.py", purpose="Hello module."),
            FileSkeleton(path="tests/test_hello.py", purpose="Hello tests."),
        ],
        test_criteria="Deterministic checks pass.",
    )
    bundle = CodeBundleArtifact(
        subtask_id="s1",
        artifacts=[
            CodeArtifact(
                subtask_id="s1",
                file_path="hello.py",
                content='def greet() -> str:\n    return "hello"\n',
                language="python",
            ),
            CodeArtifact(
                subtask_id="s1",
                file_path="tests/test_hello.py",
                content="from hello import greet\n\n\ndef test_greet():\n    assert greet() == 'hello'\n",
                language="python",
                tests_included=True,
            ),
        ],
    )
    persist_issues = persist_bundle_artifact(bundle, subtask, str(tmp_path))
    assert persist_issues == []
    lint = FakeTool("PASS")
    tests = FakeTool("PASS\n1 passed")

    review = run_bundle_deterministic_review(
        subtask,
        bundle,
        str(tmp_path),
        lint_runner=lint,
        test_runner=tests,
    )

    assert review.result.passed is True, review.result.issues
    assert lint.calls == ["hello.py", "tests/test_hello.py"]
    assert tests.calls == ["tests/test_hello.py"]


def test_final_qa_fails_when_pytest_collects_no_tests() -> None:
    qa = run_final_qa(
        "/unused",
        lint_runner=FakeTool("PASS"),
        test_runner=FakeTool("SKIP: no tests collected under this path."),
    )

    assert qa.passed is False
    assert "Tests were not executed" in qa.integration_notes


def test_patch_final_qa_treats_no_tests_as_warning() -> None:
    qa = run_final_qa(
        "/unused",
        lint_runner=FakeTool("PASS"),
        test_runner=FakeTool("SKIP: no tests collected under this path."),
        allow_no_tests=True,
    )

    assert qa.passed is True
    assert (
        "No pytest tests were collected in the existing project; "
        "QA validated changed files with ruff only."
    ) in qa.integration_notes
    assert "Tests were not executed" not in qa.integration_notes


def test_patch_final_qa_still_fails_lint_even_when_no_tests_collected() -> None:
    qa = run_final_qa(
        "/unused",
        lint_runner=FakeTool("F821 undefined name 'x'"),
        test_runner=FakeTool("SKIP: no tests collected under this path."),
        allow_no_tests=True,
    )

    assert qa.passed is False
    assert "Lint failed." in qa.integration_notes


def test_architecture_gate_fails_missing_rpa_structure(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (tmp_path / "src/demo").mkdir(parents=True)
    (tmp_path / "src/demo/__init__.py").write_text("", encoding="utf-8")

    result = run_rpa_deterministic_gate(str(tmp_path))

    assert result.passed is False
    assert "orchestrator" in "\n".join(result.issues)
    assert "producer" in "\n".join(result.issues)
    assert "consumer" in "\n".join(result.issues)


def _empty_plan(domain: str = "", mode: str = "new_project") -> Plan:
    return Plan(
        project_name="demo",
        mode=mode,
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="t",
                description="d",
                files=[FileSkeleton(path="src/demo/__init__.py", purpose="Package init.")],
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
        artifact_urls=[
            {"file_path": "hello.py", "size": 12, "url": "https://example.test/hello.py"}
        ],
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


def test_planner_inputs_use_materialized_attachment_records_not_full_tree(tmp_path: Path) -> None:
    repo = tmp_path / "inputs" / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "important.py").write_text("VALUE = 1\n", encoding="utf-8")

    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.attachment_records = [
        {
            "kind": "git",
            "name": "repo",
            "path": "inputs/repo",
            "summary": "git repo cloned from https://example.test/repo.git",
        }
    ]

    inputs = main._planner_inputs(flow.state)

    assert "git repo cloned" in inputs["attachment_records"]
    assert "inputs/repo/.git/config" not in inputs["attachment_records"]
    assert "inputs/repo/src/important.py" not in inputs["attachment_records"]


def test_patch_preflight_qa_is_passed_to_planner(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "inputs" / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_existing.py").write_text(
        "def test_existing():\n    assert True\n",
        encoding="utf-8",
    )

    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.attachments = [Attachment(kind="zip", name="repo.zip")]
    calls: list[dict] = []

    def fake_run_final_qa(build_dir: str, **kwargs):
        calls.append({"build_dir": build_dir, **kwargs})
        return QAReport(
            passed=False,
            lint_output="E999",
            test_output="FAILED tests/test_existing.py::test_existing",
        )

    monkeypatch.setattr(main, "run_final_qa", fake_run_final_qa)

    main._run_patch_preflight_if_available(flow.state)
    inputs = main._planner_inputs(flow.state)

    assert calls[0]["build_dir"] == str(repo)
    assert calls[0]["test_paths"] is None
    assert "FAILED tests/test_existing.py::test_existing" in inputs["preflight_qa_report"]


def test_writer_context_is_scoped_to_planned_file_parent(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "src" / "pkg" / "neighbor.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "unrelated.md").write_text("# docs\n", encoding="utf-8")

    subtask = SubTask(
        id="s1",
        title="Patch target",
        description="Modify target only.",
        files=[FileSkeleton(path="src/pkg/target.py", purpose="Target module.", change_type="modify")],
        test_criteria="Scoped context is enough.",
    )

    context = main._subtask_workspace_context(str(tmp_path), subtask)

    assert "src/pkg/target.py" in context
    assert "src/pkg/neighbor.py" in context
    assert "docs/unrelated.md" not in context
    assert ".git/config" not in context


def test_build_subtask_persists_bundle_artifacts(monkeypatch, tmp_path: Path) -> None:
    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    subtask = SubTask(
        id="s1",
        title="Bundle",
        description="Create bundle.",
        files=[
            FileSkeleton(path="a.py", purpose="A module."),
            FileSkeleton(path="b.py", purpose="B module."),
        ],
        test_criteria="Deterministic checks pass.",
    )

    class FakeWriterCrew:
        def __init__(self, workspace_dir: str):
            self.workspace_dir = workspace_dir

        def crew(self):
            class FakeCrew:
                def kickoff(self, inputs: dict):
                    return SimpleNamespace(
                        pydantic=CodeBundleArtifact(
                            subtask_id="s1",
                            artifacts=[
                                CodeArtifact(
                                    subtask_id="s1",
                                    file_path="a.py",
                                    content="A = 1\n",
                                    language="python",
                                ),
                                CodeArtifact(
                                    subtask_id="s1",
                                    file_path="b.py",
                                    content="B = 2\n",
                                    language="python",
                                ),
                            ],
                        )
                    )

            return FakeCrew()

    class FakeReviewerCrew:
        def __init__(self, workspace_dir: str):
            self.workspace_dir = workspace_dir

    monkeypatch.setattr(main, "WriterCrew", FakeWriterCrew)
    monkeypatch.setattr(main, "ReviewerCrew", FakeReviewerCrew)

    flow._build_subtask(subtask, str(tmp_path), index=1, total=1)

    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "B = 2\n"
    assert [a.file_path for a in flow.state.artifacts] == ["a.py", "b.py"]
    assert flow.state.review_results[-1].passed is True


def test_build_stops_after_first_failed_subtask(monkeypatch, tmp_path: Path) -> None:
    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.plan = Plan(
        project_name="demo",
        mode="new_project",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="Bad package",
                description="Fails review.",
                files=[FileSkeleton(path="a.py", purpose="A module.")],
                test_criteria="Passes review.",
            ),
            SubTask(
                id="s2",
                title="Should not run",
                description="Must be skipped after s1 fails.",
                files=[FileSkeleton(path="b.py", purpose="B module.")],
                test_criteria="Passes review.",
            ),
        ],
    )
    calls: list[str] = []

    def fake_build_subtask(subtask: SubTask, build_dir: str, *, index: int, total: int):
        calls.append(subtask.id)
        return ReviewResult(
            subtask_id=subtask.id,
            passed=False,
            issues=["deterministic review failed"],
        )

    monkeypatch.setattr(main.git_tool, "init_and_commit", lambda *args, **kwargs: None)
    monkeypatch.setattr(flow, "_build_subtask", fake_build_subtask)

    flow.build(SimpleNamespace(feedback=""))

    assert calls == ["s1"]
    assert flow.state.status == "failed"
    assert flow.state.qa_report is not None
    assert flow.state.qa_report.passed is False
    assert "s1" in flow.state.qa_report.integration_notes


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


def test_zip_build_excludes_archive_when_written_inside_build_dir(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")

    zip_path = main._zip_build(str(tmp_path), tmp_path, "demo")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "demo/app.py" in names
    assert "demo/demo.zip" not in names


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


def test_attachment_zip_skips_generated_noise(tmp_path: Path) -> None:
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("app/src/main.py", "VALUE = 1\n")
        zf.writestr("app/.venv/lib/site.py", "noise\n")
        zf.writestr("app/.pytest_cache/v/cache", "noise\n")
        zf.writestr("app/dist/app.exe", "noise\n")
        zf.writestr("app/build/temp.o", "noise\n")
        zf.writestr("app/nested.zip", "noise\n")

    attachment = {
        "kind": "zip",
        "name": "input.zip",
        "content_b64": base64.b64encode(archive.read_bytes()).decode(),
    }

    records = attachment_tool.materialize([attachment], str(tmp_path))

    extracted = tmp_path / records[0]["path"]
    assert (extracted / "app/src/main.py").is_file()
    assert not (extracted / "app/.venv").exists()
    assert not (extracted / "app/.pytest_cache").exists()
    assert not (extracted / "app/dist").exists()
    assert not (extracted / "app/build").exists()
    assert not (extracted / "app/nested.zip").exists()


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


def test_final_qa_reports_pytest_failures_even_when_deterministic_gate_fails(tmp_path: Path) -> None:
    tests = FakeTool("FAILED tests/test_app.py::test_app")

    qa = run_final_qa(
        str(tmp_path),
        lint_runner=FakeTool("bad.py:1:1: F401 unused import"),
        type_runner=FakeTool("PASS"),
        test_runner=tests,
    )

    assert qa.passed is False
    assert tests.calls == ["."]
    assert "FAILED tests/test_app.py::test_app" in qa.test_output
    assert "Lint failed." in qa.integration_notes
    assert "Tests failed." in qa.integration_notes


def test_final_qa_runs_targeted_test_paths(tmp_path: Path) -> None:
    tests = FakeToolByPath({"tests/test_changed.py": "PASS\n1 passed"})

    qa = run_final_qa(
        str(tmp_path),
        lint_runner=FakeTool("PASS"),
        type_runner=FakeTool("PASS"),
        test_runner=tests,
        test_paths=["tests/test_changed.py"],
    )

    assert qa.passed is True
    assert tests.calls == ["tests/test_changed.py"]
    assert "pytest scoped to 1 path(s)" in qa.integration_notes


def test_finalize_repairs_failed_final_qa_and_returns_payload(monkeypatch, tmp_path: Path) -> None:
    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "executing"
    monkeypatch.setenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", "1")

    qa_calls: list[str] = []

    def fake_run_final_qa(build_dir: str, *, lint_paths=None, **_kwargs) -> QAReport:
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
        {
            "file_path": "app.py",
            "size": 37,
            "url": "https://example.test/app.py",
            "kind": "file",
        }
    ]


def test_finalize_returns_project_archive_as_primary_artifact(monkeypatch, tmp_path: Path) -> None:
    build_dir = tmp_path / "inputs" / "repo"
    build_dir.mkdir(parents=True)
    (build_dir / "changed.py").write_text("VALUE = 2\n", encoding="utf-8")
    (build_dir / "untouched.py").write_text("VALUE = 1\n", encoding="utf-8")

    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "executing"
    flow.state.plan = Plan(
        project_name="demo",
        mode="patch_existing",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="Patch changed file",
                description="Modify only changed.py.",
                files=[FileSkeleton(path="changed.py", purpose="Changed module.", change_type="modify")],
                test_criteria="Archive contains the repaired project.",
            )
        ],
    )
    flow.state.artifacts = [
        CodeArtifact(
            subtask_id="s1",
            file_path="changed.py",
            content="VALUE = 2\n",
            language="python",
        )
    ]
    flow._build_dir = str(build_dir)

    monkeypatch.setattr(
        main,
        "run_final_qa",
        lambda build_dir, **_kwargs: QAReport(
            passed=True, lint_output="PASS", test_output="PASS", integration_notes="Clean."
        ),
    )
    def fail_upload_workspace(build_dir, prefix):
        raise AssertionError("patch jobs should not upload every file by default")

    monkeypatch.setattr(main, "upload_workspace", fail_upload_workspace)
    monkeypatch.setattr(
        main,
        "upload_file",
        lambda local_path, key: {
            "file_path": Path(local_path).name,
            "size": Path(local_path).stat().st_size,
            "url": "https://example.test/demo.zip",
        },
    )
    monkeypatch.setattr(main.history, "record", lambda state: None)

    payload = flow.finalize(None)

    assert payload["project_archive"]["kind"] == "project_archive"
    assert payload["project_archive"]["file_path"] == "demo.zip"
    assert payload["project_archive"]["url"] == "https://example.test/demo.zip"
    assert payload["artifact_urls"][0]["kind"] == "project_archive"
    assert payload["artifact_urls"][0]["file_path"] == "demo.zip"
    assert len(payload["artifact_urls"]) == 1
    with zipfile.ZipFile(payload["zip_path"]) as zf:
        names = zf.namelist()
    assert "demo/changed.py" in names
    assert "demo/untouched.py" in names


def test_patch_finalize_uploads_file_artifacts_only_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    build_dir = tmp_path / "inputs" / "repo"
    build_dir.mkdir(parents=True)
    (build_dir / "changed.py").write_text("VALUE = 2\n", encoding="utf-8")

    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "executing"
    flow.state.plan = Plan(
        project_name="demo",
        mode="patch_existing",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="Patch changed file",
                description="Modify only changed.py.",
                files=[FileSkeleton(path="changed.py", purpose="Changed module.", change_type="modify")],
                test_criteria="Archive contains the repaired project.",
            )
        ],
    )
    flow._build_dir = str(build_dir)

    monkeypatch.setenv("CODEBUILDER_UPLOAD_FILE_ARTIFACTS", "true")
    monkeypatch.setattr(
        main,
        "run_final_qa",
        lambda build_dir, **_kwargs: QAReport(
            passed=True, lint_output="PASS", test_output="PASS", integration_notes="Clean."
        ),
    )
    monkeypatch.setattr(
        main,
        "upload_workspace",
        lambda build_dir, prefix: [
            {"file_path": "changed.py", "size": 10, "url": "https://example.test/changed.py"}
        ],
    )
    monkeypatch.setattr(
        main,
        "upload_file",
        lambda local_path, key: {
            "file_path": Path(local_path).name,
            "size": Path(local_path).stat().st_size,
            "url": "https://example.test/demo.zip",
        },
    )
    monkeypatch.setattr(main.history, "record", lambda state: None)

    payload = flow.finalize(None)

    assert [ref["kind"] for ref in payload["artifact_urls"]] == ["project_archive", "file"]


def test_patch_finalize_no_tests_warning_does_not_trigger_repair(
    monkeypatch, tmp_path: Path
) -> None:
    build_dir = tmp_path / "inputs" / "repo"
    build_dir.mkdir(parents=True)
    (build_dir / "changed.py").write_text("VALUE = 2\n", encoding="utf-8")

    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "executing"
    flow.state.plan = Plan(
        project_name="demo",
        mode="patch_existing",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="Patch changed file",
                description="Modify only changed.py.",
                files=[FileSkeleton(path="changed.py", purpose="Changed module.", change_type="modify")],
                test_criteria="Existing project may not include tests.",
            )
        ],
    )
    flow.state.artifacts = [
        CodeArtifact(
            subtask_id="s1",
            file_path="changed.py",
            content="VALUE = 2\n",
            language="python",
        )
    ]
    flow._build_dir = str(build_dir)

    def fake_run_final_qa(build_dir, **kwargs):
        return QAReport(
            passed=bool(kwargs.get("allow_no_tests")),
            lint_output="PASS",
            test_output="SKIP: no tests collected under this path.",
            integration_notes=(
                "No pytest tests were collected in the existing project; "
                "QA validated changed files with ruff only."
            ),
        )

    class FailingWriterCrew:
        def __init__(self, workspace_dir: str):
            self.workspace_dir = workspace_dir

        def repair_crew(self):
            raise AssertionError("patch no-tests warning should not trigger final QA repair")

    monkeypatch.setenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", "1")
    monkeypatch.delenv("CODEBUILDER_ARTIFACT_BUCKET", raising=False)
    monkeypatch.setattr(main, "run_final_qa", fake_run_final_qa)
    monkeypatch.setattr(main, "WriterCrew", FailingWriterCrew)
    monkeypatch.setattr(main, "upload_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.history, "record", lambda state: None)

    payload = flow.finalize(None)

    assert payload["status"] == "done"
    assert payload["qa_report"]["passed"] is True
    assert flow.state.final_qa_repair_attempts == 0
    assert "No pytest tests were collected" in payload["qa_report"]["integration_notes"]


def test_patch_test_paths_select_changed_and_related_tests(tmp_path: Path) -> None:
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_settings.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "unit" / "test_unrelated.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )

    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.plan = Plan(
        project_name="demo",
        mode="patch_existing",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="Patch settings",
                description="Patch settings.",
                files=[
                    FileSkeleton(
                        path="src/demo/settings.py",
                        purpose="Settings module.",
                        change_type="modify",
                    )
                ],
                test_criteria="Related settings tests pass.",
            )
        ],
    )
    flow.state.artifacts = [
        CodeArtifact(subtask_id="s1", file_path="src/demo/settings.py", language="python"),
        CodeArtifact(subtask_id="s1", file_path="tests/unit/test_explicit.py", language="python"),
    ]

    assert flow._final_qa_test_paths(str(tmp_path)) is None


def test_patch_final_qa_runs_full_suite_when_tests_exist(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_unrelated.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )

    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.plan = _empty_plan(mode="patch_existing")
    flow.state.artifacts = [
        CodeArtifact(subtask_id="s1", file_path="src/demo/settings.py", language="python")
    ]

    assert flow._final_qa_test_paths(str(tmp_path)) is None


def test_patch_test_paths_full_scope_env_returns_none(monkeypatch, tmp_path: Path) -> None:
    flow = CodebuilderFlow()
    flow.state.plan = _empty_plan(mode="patch_existing")
    monkeypatch.setenv("CODEBUILDER_PATCH_TEST_SCOPE", "full")

    assert flow._final_qa_test_paths(str(tmp_path)) is None


def test_mode_aware_final_qa_repair_defaults(monkeypatch) -> None:
    monkeypatch.delenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", raising=False)

    patch_flow = CodebuilderFlow()
    patch_flow.state.plan = _empty_plan(mode="patch_existing")
    assert patch_flow._max_final_qa_repairs() == 1

    new_flow = CodebuilderFlow()
    new_flow.state.plan = _empty_plan(mode="new_project")
    assert new_flow._max_final_qa_repairs() == 2

    monkeypatch.setenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", "4")
    assert patch_flow._max_final_qa_repairs() == 4


def test_finalize_fails_when_configured_project_archive_upload_fails(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")

    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.project_name = "demo"
    flow.state.status = "executing"
    flow.state.plan = _empty_plan(domain="")

    monkeypatch.setenv("CODEBUILDER_ARTIFACT_BUCKET", "artifact-bucket")
    monkeypatch.setattr(
        main,
        "run_final_qa",
        lambda build_dir, **_kwargs: QAReport(
            passed=True, lint_output="PASS", test_output="PASS", integration_notes="Clean."
        ),
    )
    monkeypatch.setattr(main, "upload_workspace", lambda build_dir, prefix: [])
    monkeypatch.setattr(main, "upload_file", lambda local_path, key: None)
    monkeypatch.setattr(main.history, "record", lambda state: None)

    payload = flow.finalize(None)

    assert payload["status"] == "failed"
    assert "project_archive" not in payload
    assert "zip_path" not in payload
    assert "artifact_urls" not in payload
    assert "Project archive upload failed" in payload["qa_report"]["integration_notes"]


def test_history_record_strips_artifact_urls_and_truncates_patch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setenv("CODEBUILDER_HISTORY_ENABLED", "true")

    state = CodebuilderState()
    state.project_key = "demo"
    state.project_name = "demo"
    state.plan = _empty_plan(domain="")
    state.status = "done"
    state.qa_report = QAReport(
        passed=True,
        lint_output="PASS",
        test_output="PASS",
        integration_notes="Clean.",
        artifact_urls=[
            ArtifactRef(
                file_path=f"file_{i}.py",
                size=10,
                url=f"https://example.test/file_{i}.py",
            )
            for i in range(500)
        ],
    )
    state.patch = "x" * 60000

    history.record(state)

    with sqlite3.connect(tmp_path / "history.db") as conn:
        qa_blob, patch = conn.execute(
            "select qa_report_json, patch from project_history"
        ).fetchone()

    stored_qa = json.loads(qa_blob)
    assert stored_qa["artifact_urls"] == []
    assert len(patch) < 51000
    assert "[truncated" in patch


def test_failed_qa_omits_runnable_archive_fields(monkeypatch, tmp_path: Path) -> None:
    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.status = "executing"
    monkeypatch.setenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", "1")

    def fake_run_final_qa(build_dir: str, *, lint_paths=None, **_kwargs) -> QAReport:
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
    assert "artifact_urls" not in payload
    assert "zip_path" not in payload
    assert "zip_url" not in payload
    assert "project_archive" not in payload
    assert payload["qa_report"]["artifact_urls"] == []
