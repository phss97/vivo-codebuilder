"""Tests for the patch-existing + import-completeness + skeleton planner changes."""

from pathlib import Path

from codebuilder.runtime_qa import (
    run_deterministic_review,
    run_import_completeness_gate,
    validate_plan,
)
from codebuilder.schemas import (
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
        file_path=file_path,
        change_type="modify",
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


def test_subtask_default_change_type_is_create() -> None:
    subtask = SubTask(
        id="s1",
        title="x",
        description="x",
        file_path="x.py",
        test_criteria="x",
    )
    assert subtask.change_type == "create"


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
            SubTask(id="s1", title="t", description="d", file_path="src/myproj/use_cases.py", test_criteria="t")
        ],
    )

    missing, stubs = run_import_completeness_gate(str(tmp_path), plan)

    assert any("entities" in p for p in missing), missing
    assert len(stubs) >= 1
    stub = stubs[0]
    assert "entities" in stub.file_path
    assert stub.change_type == "create"
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
            SubTask(id="s1", title="t", description="d", file_path="src/myproj/foo.py", test_criteria="t")
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
            SubTask(id="s1", title="t", description="d", file_path="src/myproj/foo.py", test_criteria="t")
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
            SubTask(id="s1", title="t", description="d", file_path="src/myproj/use_cases.py", test_criteria="t")
        ],
    )

    missing, stubs = run_import_completeness_gate(str(tmp_path), plan)
    assert len(missing) == 12
    assert len(stubs) == 8


def test_validate_plan_accepts_up_to_60_subtasks() -> None:
    subtasks = [
        SubTask(
            id=f"s{i:02d}",
            title="t",
            description="d",
            file_path=f"src/pkg/f{i}.py",
            test_criteria="t",
        )
        for i in range(50)
    ]
    plan = Plan(
        project_name="x",
        mode="new_project",
        tech_stack=["python"],
        subtasks=subtasks,
    )
    assert validate_plan(plan) is plan


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
