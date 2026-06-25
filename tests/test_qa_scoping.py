"""Patch jobs must scope the mypy gate to changed Python files, mirroring the
lint scoping — otherwise pre-existing type debt in untouched files fails the job.
"""

from codebuilder.main import CodebuilderFlow
from codebuilder.schemas import CodeArtifact, FileSkeleton, Plan, SubTask


def _plan(mode: str) -> Plan:
    return Plan(
        project_name="p",
        mode=mode,
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s1",
                title="t",
                description="d",
                files=[FileSkeleton(path="src/p/a.py", purpose="x")],
                test_criteria="c",
            )
        ],
    )


def _flow(mode: str, paths: list[str]) -> CodebuilderFlow:
    flow = CodebuilderFlow()
    flow.state.plan = _plan(mode)
    flow.state.artifacts = [
        CodeArtifact(subtask_id="s1", file_path=p, language="python") for p in paths
    ]
    return flow


def test_type_paths_patch_scopes_changed_python() -> None:
    flow = _flow("patch_existing", ["src/p/a.py", "README.md", "src/p/b.py"])
    assert flow._final_qa_type_paths() == ["src/p/a.py", "src/p/b.py"]


def test_type_paths_new_project_is_none() -> None:
    flow = _flow("new_project", ["src/p/a.py"])
    assert flow._final_qa_type_paths() is None


def test_type_paths_patch_without_python_is_none() -> None:
    flow = _flow("patch_existing", ["README.md", ".env.example"])
    assert flow._final_qa_type_paths() is None
