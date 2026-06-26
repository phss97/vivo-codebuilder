"""Tests for the zip-based ``patch_existing`` failure fixes:

1. ``_resolve_patch_root`` — zip-extracted dirs under ``inputs/`` resolve as the
   patch root (previously only ``inputs/repo*`` git clones did, so zip jobs
   built at the workspace root and duplicated the project tree).
2. ``_strip_patch_root_prefix`` — workspace-relative planned paths
   (``inputs/<dir>/src/x.py``) normalize to patch-root-relative paths.
3. ``build()`` — wires both together and git-inits zip patch roots so
   ``finalize``'s ``git diff`` captures the repair.
4. ``TestRunnerTool`` — a target project's ``addopts`` (e.g. ``--cov`` without
   pytest-cov installed) must not crash pytest at arg parsing.
5. ``run_final_qa(lint_paths=...)`` — patch jobs lint only changed files, so
   pre-existing lint debt in untouched user code cannot fail the job.
"""

from pathlib import Path

from git import Repo

from codebuilder.main import (
    CodebuilderFlow,
    _resolve_patch_root,
    _strip_patch_root_prefix,
)
from codebuilder.runtime_qa import run_final_qa
from codebuilder.schemas import FileSkeleton, Plan, ReviewResult, SubTask
from codebuilder.tools.lint_runner_tool import TestRunnerTool

# build is wrapped by @listen("approved"); unwrap to call the body directly
# (same pattern as test_revise_plan_recovery).
_BUILD_BODY = CodebuilderFlow.build.unwrap()


def _patch_plan(paths: list[str]) -> Plan:
    return Plan(
        project_name="demo",
        mode="patch_existing",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s01",
                title="repair",
                description="d",
                files=[FileSkeleton(path=p, purpose="Repair.") for p in paths],
                test_criteria="t",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# 1 — _resolve_patch_root                                                      #
# --------------------------------------------------------------------------- #
def test_patch_root_prefers_git_clone(tmp_path: Path) -> None:
    (tmp_path / "inputs" / "repo").mkdir(parents=True)
    (tmp_path / "inputs" / "app-x").mkdir()

    assert _resolve_patch_root(str(tmp_path)) == str(tmp_path / "inputs" / "repo")


def test_patch_root_resolves_zip_extracted_dir(tmp_path: Path) -> None:
    extracted = tmp_path / "inputs" / "app-faturamento-terra"
    (extracted / "src").mkdir(parents=True)
    (extracted / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    # Non-dir inputs (PDFs, images) must not be considered.
    (tmp_path / "inputs" / "spec.pdf").write_bytes(b"%PDF")

    assert _resolve_patch_root(str(tmp_path)) == str(extracted)


def test_patch_root_descends_zip_wrapper_dirs(tmp_path: Path) -> None:
    # zip-of-a-folder: inputs/app-x/ contains only app-x/ which holds the project
    inner = tmp_path / "inputs" / "app-x" / "app-x"
    (inner / "src").mkdir(parents=True)
    (inner / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert _resolve_patch_root(str(tmp_path)) == str(inner)


def test_patch_root_prefers_marked_dir_over_unmarked(tmp_path: Path) -> None:
    (tmp_path / "inputs" / "assets").mkdir(parents=True)
    marked = tmp_path / "inputs" / "zz-project"
    marked.mkdir()
    (marked / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert _resolve_patch_root(str(tmp_path)) == str(marked)


def test_patch_root_none_when_no_input_dirs(tmp_path: Path) -> None:
    assert _resolve_patch_root(str(tmp_path)) is None  # no inputs/ at all
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "spec.pdf").write_bytes(b"%PDF")
    assert _resolve_patch_root(str(tmp_path)) is None  # files only


# --------------------------------------------------------------------------- #
# 2 — _strip_patch_root_prefix                                                 #
# --------------------------------------------------------------------------- #
def test_strip_patch_root_prefix_normalizes_workspace_relative_paths() -> None:
    plan = _patch_plan(["inputs/app-x/src/mod.py", "src/other.py"])

    _strip_patch_root_prefix(plan, "inputs/app-x")

    assert [f.path for f in plan.subtasks[0].files] == ["src/mod.py", "src/other.py"]


# --------------------------------------------------------------------------- #
# 3 — build() wiring: patch root + baseline git init                           #
# --------------------------------------------------------------------------- #
def test_build_uses_zip_patch_root_and_inits_git(tmp_path: Path, monkeypatch) -> None:
    extracted = tmp_path / "inputs" / "app-x"
    (extracted / "src").mkdir(parents=True)
    (extracted / "src" / "mod.py").write_text("X = 1\n", encoding="utf-8")

    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.plan = _patch_plan(["inputs/app-x/src/mod.py"])

    built: list[tuple[str, str]] = []
    def fake_build_subtask(self, subtask, build_dir, *, index, total):
        built.append((subtask.files[0].path, build_dir))
        return ReviewResult(subtask_id=subtask.id, passed=True)

    monkeypatch.setattr(
        CodebuilderFlow,
        "_build_subtask",
        fake_build_subtask,
    )

    _BUILD_BODY(flow, prior=None)

    assert flow._build_dir == str(extracted)
    assert built == [("src/mod.py", str(extracted))]
    # Baseline commit exists so finalize's git diff captures exactly the repair.
    repo = Repo(str(extracted))
    assert not repo.bare
    assert repo.head.commit.message.startswith("codebuilder baseline")
    (extracted / "src" / "mod.py").write_text("X = 2\n", encoding="utf-8")
    from codebuilder.tools import git_tool

    assert "X = 2" in git_tool.diff(str(extracted))


# --------------------------------------------------------------------------- #
# 4 — TestRunnerTool neutralizes project addopts                               #
# --------------------------------------------------------------------------- #
def test_test_runner_survives_project_addopts(tmp_path: Path) -> None:
    # An addopts flag no installed plugin provides crashes pytest at arg
    # parsing ("unrecognized arguments") before any test runs — the exact
    # failure mode of `--cov=src` without pytest-cov in the runner.
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "--definitely-not-a-real-flag"\n',
        encoding="utf-8",
    )
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    output = TestRunnerTool(workspace_dir=str(tmp_path))._run(".")

    assert output.startswith("PASS"), output


# --------------------------------------------------------------------------- #
# 5 — run_final_qa lint scoping                                                #
# --------------------------------------------------------------------------- #
class _FakeLint:
    """Per-path canned lint results; '.' simulates the whole-dir sweep."""

    def __init__(self, results: dict[str, str]) -> None:
        self.results = results
        self.calls: list[str] = []

    def _run(self, path: str = ".") -> str:
        self.calls.append(path)
        return self.results.get(path, "PASS")


class _FakeTest:
    def _run(self, path: str = ".") -> str:
        return "PASS\n1 passed"


def test_final_qa_scoped_lint_ignores_preexisting_debt(tmp_path: Path) -> None:
    lint = _FakeLint({".": "legacy.py:1:1: F401 unused import", "src/changed.py": "PASS"})

    scoped = run_final_qa(
        str(tmp_path), lint_paths=["src/changed.py"], lint_runner=lint, test_runner=_FakeTest()
    )
    unscoped = run_final_qa(str(tmp_path), lint_runner=lint, test_runner=_FakeTest())

    assert scoped.passed is True
    assert "scoped to 1 changed file(s)" in scoped.integration_notes
    assert unscoped.passed is False


def test_final_qa_scoped_lint_aggregates_failures_and_skips(tmp_path: Path) -> None:
    failing = _FakeLint({"a.py": "a.py:1:1: E999 bad", "b.py": "PASS"})
    failed = run_final_qa(
        str(tmp_path), lint_paths=["a.py", "b.py"], lint_runner=failing, test_runner=_FakeTest()
    )
    assert failed.passed is False
    assert "E999" in failed.lint_output

    skipping = _FakeLint({"a.py": "SKIP: ruff not installed in the runtime; review logic manually."})
    skipped = run_final_qa(
        str(tmp_path), lint_paths=["a.py"], lint_runner=skipping, test_runner=_FakeTest()
    )
    assert skipped.passed is False
    assert "Lint was not executed" in skipped.integration_notes


def test_final_qa_lint_paths_helper_scopes_only_patch_jobs() -> None:
    from codebuilder.schemas import CodeArtifact

    flow = CodebuilderFlow()
    flow.state.plan = _patch_plan(["src/mod.py"])
    flow.state.artifacts = [
        CodeArtifact(subtask_id="s01", file_path="src/mod.py", content="X = 1\n", language="python"),
        CodeArtifact(subtask_id="s01", file_path="src/mod.py", content="X = 2\n", language="python"),
    ]
    assert flow._final_qa_lint_paths() == ["src/mod.py"]

    flow.state.plan.mode = "new_project"
    assert flow._final_qa_lint_paths() is None
