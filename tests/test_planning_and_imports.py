"""Tests for the patch-existing + import-completeness + skeleton planner changes."""

from pathlib import Path

from codebuilder.runtime_qa import (
    run_deterministic_review,
    run_import_completeness_gate,
    validate_plan,
)
from codebuilder.schemas import (
    CodeBundleArtifact,
    CodeArtifact,
    FileSkeleton,
    Plan,
    PlanSkeleton,
    SubTask,
)


class _FakeLint:
    def __init__(self, output: str = "PASS") -> None:
        self.output = output

    def _run(self, path: str = ".") -> str:
        return self.output


def _modify_subtask(file_path: str = "src/pkg/svc.py") -> SubTask:
    return SubTask(
        id="s1",
        title="Modify file",
        description="Apply transformation.",
        files=[
            FileSkeleton(
                path=file_path,
                purpose="Existing service module to modify.",
                change_type="modify",
            )
        ],
        tech_notes="Extract method foo into helper.",
        test_criteria="Deterministic checks pass.",
    )


def _write_artifact(tmp_path: Path, path: str, content: str) -> CodeArtifact:
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return CodeArtifact(
        subtask_id="s1",
        file_path=path,
        content=content,
        language="python",
    )


def test_subtask_groups_planned_files() -> None:
    subtask = SubTask(
        id="s1",
        title="x",
        description="x",
        files=[FileSkeleton(path="x.py", purpose="x")],
        test_criteria="x",
    )
    assert subtask.files[0].change_type == "create"


def test_code_bundle_artifact_roundtrip() -> None:
    bundle = CodeBundleArtifact(
        subtask_id="s1",
        artifacts=[
            CodeArtifact(
                subtask_id="s1",
                file_path="src/pkg/a.py",
                content="A = 1\n",
                language="python",
            ),
            CodeArtifact(
                subtask_id="s1",
                file_path="src/pkg/b.py",
                content="B = 2\n",
                language="python",
            ),
        ],
    )

    restored = CodeBundleArtifact.model_validate_json(bundle.model_dump_json())

    assert [a.file_path for a in restored.artifacts] == ["src/pkg/a.py", "src/pkg/b.py"]


def test_deterministic_review_fails_when_modify_produces_noop(tmp_path: Path) -> None:
    original = 'def greet() -> str:\n    return "hello"\n'
    artifact = _write_artifact(tmp_path, "svc.py", original)

    review = run_deterministic_review(
        _modify_subtask("svc.py"),
        artifact,
        str(tmp_path),
        existing_snapshot=original,
        lint_runner=_FakeLint(),
    )

    assert review.result.passed is False
    assert any("no change" in issue.lower() for issue in review.result.issues), review.result.issues


def test_deterministic_review_passes_when_modify_changes_content(tmp_path: Path) -> None:
    original = 'def greet() -> str:\n    return "hello"\n'
    modified = 'def greet(name: str) -> str:\n    return f"hello {name}"\n'
    artifact = _write_artifact(tmp_path, "svc.py", modified)

    review = run_deterministic_review(
        _modify_subtask("svc.py"),
        artifact,
        str(tmp_path),
        existing_snapshot=original,
        lint_runner=_FakeLint(),
    )

    assert review.result.passed is True, review.result.issues


def test_import_gate_flags_missing_own_package_module(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "myproj"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "use_cases.py").write_text(
        "from myproj.domain.entities import Job\n",
        encoding="utf-8",
    )

    plan = Plan(
        project_name="x",
        mode="new_project",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="t",
                description="d",
                files=[FileSkeleton(path="src/myproj/use_cases.py", purpose="Use cases.")],
                test_criteria="t",
            )
        ],
    )

    missing, stubs = run_import_completeness_gate(str(tmp_path), plan)

    assert any("entities" in p for p in missing), missing
    assert len(stubs) >= 1
    stub = stubs[0]
    assert "entities" in stub.files[0].path
    assert stub.files[0].change_type == "create"
    assert "Job" in stub.description


def test_import_gate_skips_external_packages(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "myproj"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "foo.py").write_text(
        "from app_internal_sdk.bar import Baz\n",
        encoding="utf-8",
    )

    plan = Plan(
        project_name="x",
        mode="new_project",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="t",
                description="d",
                files=[FileSkeleton(path="src/myproj/foo.py", purpose="Foo module.")],
                test_criteria="t",
            )
        ],
        external_packages=["app_internal_sdk"],
    )

    missing, stubs = run_import_completeness_gate(str(tmp_path), plan)
    assert missing == []
    assert stubs == []


def test_import_gate_ignores_unknown_top_packages(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "myproj"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "foo.py").write_text(
        "from totally_unknown_pkg.bar import Baz\n",
        encoding="utf-8",
    )

    plan = Plan(
        project_name="x",
        mode="new_project",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="t",
                description="d",
                files=[FileSkeleton(path="src/myproj/foo.py", purpose="Foo module.")],
                test_criteria="t",
            )
        ],
    )

    missing, stubs = run_import_completeness_gate(str(tmp_path), plan)
    assert missing == []
    assert stubs == []


def test_import_gate_caps_stubs_at_eight(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "myproj"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    importer_lines = [
        f"from myproj.missing_{i} import Sym{i}" for i in range(12)
    ]
    (pkg / "use_cases.py").write_text("\n".join(importer_lines) + "\n", encoding="utf-8")

    plan = Plan(
        project_name="x",
        mode="new_project",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="t",
                description="d",
                files=[FileSkeleton(path="src/myproj/use_cases.py", purpose="Use cases.")],
                test_criteria="t",
            )
        ],
    )

    missing, stubs = run_import_completeness_gate(str(tmp_path), plan)
    assert len(missing) == 12
    assert len(stubs) == 8


def test_validate_plan_accepts_up_to_24_work_packages() -> None:
    subtasks = [
        SubTask(
            id=f"s{i:02d}",
            title="t",
            description="d",
            files=[FileSkeleton(path=f"src/pkg/f{i}.py", purpose="Module.")],
            test_criteria="t",
        )
        for i in range(24)
    ]
    plan = Plan(
        project_name="x",
        mode="new_project",
        tech_stack=["python"],
        subtasks=subtasks,
    )
    assert validate_plan(plan) is plan


def test_validate_plan_rejects_oversized_work_package() -> None:
    subtask = SubTask(
        id="s1",
        title="Too large",
        description="d",
        files=[
            FileSkeleton(path=f"src/pkg/f{i}.py", purpose="Module.")
            for i in range(9)
        ],
        test_criteria="t",
    )
    plan = Plan(project_name="x", mode="new_project", tech_stack=["python"], subtasks=[subtask])

    try:
        validate_plan(plan)
    except ValueError as exc:
        assert "at most 8 files" in str(exc)
    else:
        raise AssertionError("oversized work package was accepted")


def test_guardrail_force_splits_oversized_work_package() -> None:
    from codebuilder.crews.planner_crew.planner_crew import _require_nonempty_plan

    class _Out:
        def __init__(self, plan: Plan) -> None:
            self.pydantic = plan

    def _plan(n_files: int) -> Plan:
        subtask = SubTask(
            id="s03",
            title="Bundle",
            description="d",
            files=[
                FileSkeleton(path=f"src/pkg/f{i}.py", purpose="Module.")
                for i in range(n_files)
            ],
            test_criteria="t",
        )
        return Plan(
            project_name="x", mode="new_project", tech_stack=["python"], subtasks=[subtask]
        )

    # 9 files > cap (8): guardrail fails so the crew re-prompts the planner to split.
    ok, msg = _require_nonempty_plan(_Out(_plan(9)))
    assert ok is False
    assert "s03" in msg and "split" in msg.lower()

    # 8 files == cap: accepted, returns the Plan unchanged.
    ok, result = _require_nonempty_plan(_Out(_plan(8)))
    assert ok is True
    assert isinstance(result, Plan)


def test_validate_plan_rejects_duplicate_planned_file_paths() -> None:
    plan = Plan(
        project_name="x",
        mode="new_project",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="a",
                description="d",
                files=[FileSkeleton(path="src/pkg/shared.py", purpose="Module.")],
                test_criteria="t",
            ),
            SubTask(
                id="s2",
                title="b",
                description="d",
                files=[FileSkeleton(path="src/pkg/shared.py", purpose="Module.")],
                test_criteria="t",
            ),
        ],
    )

    try:
        validate_plan(plan)
    except ValueError as exc:
        assert "duplicate planned file path" in str(exc)
    else:
        raise AssertionError("duplicate planned path was accepted")


def test_validate_plan_rejects_diagnostic_only_plan() -> None:
    plan = Plan(
        project_name="TBD",
        mode="patch_existing",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="Diagnose failures",
                description="Inspect the project and write diagnostic notes.",
                files=[
                    FileSkeleton(
                        path="FILES_TO_BE_DETERMINED_BY_DIAGNOSTICS.md",
                        purpose="Placeholder diagnostic report.",
                    )
                ],
                test_criteria="TESTS_TO_BE_DETERMINED_BY_DIAGNOSTICS",
            )
        ],
    )

    try:
        validate_plan(plan)
    except ValueError as exc:
        message = str(exc)
        assert "placeholder" in message.lower()
        assert "diagnostic-only" in message.lower()
    else:
        raise AssertionError("diagnostic-only plan was accepted")


def test_plan_skeleton_roundtrip() -> None:
    skeleton = PlanSkeleton(
        project_name="x",
        mode="new_project",
        domain="rpa",
        tech_stack=["python"],
        files=[
            FileSkeleton(path="src/pkg/__init__.py", purpose="package init", change_type="create"),
            FileSkeleton(
                path="src/pkg/domain/entities/job.py",
                purpose="Job entity",
                change_type="create",
            ),
        ],
        external_packages=["app_internal_sdk"],
    )
    dumped = skeleton.model_dump_json()
    restored = PlanSkeleton.model_validate_json(dumped)
    assert restored.files[1].path == "src/pkg/domain/entities/job.py"
    assert restored.external_packages == ["app_internal_sdk"]
