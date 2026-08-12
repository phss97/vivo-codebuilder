"""Focused safety checks for structured package execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

import codebuilder.main as main
from codebuilder.package_workspace import snapshot_files, stage_tree
from codebuilder.runtime_qa import plan_spec_hash, validate_plan
from codebuilder.schemas import (
    AuthoritativeAsset,
    CommandResult,
    IntakeAssessment,
    PackageResult,
    Plan,
    ProductionReview,
    QAIssue,
    QAReport,
    VerificationCommand,
)


def _package(package_id: str, *, depends_on: list[str] | None = None) -> dict:
    slug = package_id.replace("-", "_")
    source = f"src/demo_pkg/{slug}.py"
    test = f"tests/test_{slug}.py"
    criterion = f"{package_id}-criterion"
    return {
        "id": package_id,
        "title": f"Package {package_id}",
        "what_to_build": f"Build the {package_id} behavior.",
        "expected_behavior": f"The {package_id} behavior works.",
        "success_criteria": [
            {"id": criterion, "description": f"{package_id} is implemented."}
        ],
        "tests": [
            {
                "id": f"{package_id}-test",
                "criterion_ids": [criterion],
                "path": test,
                "test_name": f"test_{slug}",
                "expected_behavior": f"Checks {package_id}.",
            }
        ],
        "files": [
            {
                "path": source,
                "purpose": f"Implements {package_id}.",
                "kind": "source",
                "public_api": [f"build_{slug}() -> str"],
            },
            {
                "path": test,
                "purpose": f"Tests {package_id}.",
                "kind": "test",
                "public_api": [],
            },
        ],
        "depends_on": depends_on or [],
    }


def _plan(*packages: dict) -> Plan:
    return validate_plan(
        Plan.model_validate(
            {
                "project_name": "demo",
                "mode": "new_project",
                "tech_stack": ["python"],
                "package_name": "demo_pkg",
                "work_packages": list(packages or [_package("wp-1")]),
                "verification_commands": [
                    {
                        "id": "tests",
                        "category": "test",
                        "argv": ["python", "-m", "unittest", "discover"],
                    }
                ],
            }
        )
    )


@pytest.fixture
def flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> main.CodebuilderFlow:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        main.git_tool, "init_and_commit", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(main, "_install_skills", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main.history, "record", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("CODEBUILDER_ARTIFACT_BUCKET", raising=False)
    monkeypatch.setenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", "0")
    instance = main.CodebuilderFlow()
    instance.state.workspace_dir = str(workspace)
    instance.state.project_name = "demo"
    instance.state.brief = "Build the approved demo."
    return instance


def _green_qa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "check_spec_contract", lambda *_args, **_kwargs: "PASS")
    monkeypatch.setattr(
        main,
        "run_verification_commands",
        lambda *_args, **_kwargs: [
            CommandResult(command_id="tests", passed=True, returncode=0)
        ],
    )

    async def reviewer(**_kwargs) -> ProductionReview:
        return ProductionReview(passed=True)

    monkeypatch.setattr(main.cc_agent, "run_reviewer", reviewer)


def test_test_author_out_of_scope_change_is_restored_and_pauses_for_test_owner(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"))
    flow.state.plan = plan

    async def test_author(*, cwd, declared_test_files, **_kwargs) -> None:
        test = Path(cwd) / declared_test_files[0]
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("def test_wp_1(): pass\n")
        (Path(cwd) / "UNAPPROVED.md").write_text("agent drift\n")

    async def executor(**_kwargs) -> None:
        raise AssertionError(
            "implementation must not run after a test contract failure"
        )

    monkeypatch.setattr(main.cc_agent, "run_test_author", test_author)
    monkeypatch.setattr(main.cc_agent, "run_executor", executor)

    result = asyncio.run(flow._build_structured(plan))

    assert result == {"route": "qa_exhausted"}
    assert flow.state.current_failure is not None
    assert flow.state.current_failure.issues[0].owner == "test"
    assert not (Path(flow.state.current_stage_dir) / "UNAPPROVED.md").exists()


def test_executor_cannot_promote_mutated_tests_or_unapproved_files(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"))
    flow.state.plan = plan
    _green_qa(monkeypatch)

    async def test_author(*, cwd, declared_test_files, **_kwargs) -> None:
        test = Path(cwd) / declared_test_files[0]
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("FROZEN = True\n\ndef test_wp_1(): pass\n")

    async def executor(*, cwd, **_kwargs) -> None:
        root = Path(cwd)
        (root / "tests/test_wp_1.py").write_text("FROZEN = False\n")
        (root / "src/demo_pkg").mkdir(parents=True, exist_ok=True)
        (root / "src/demo_pkg/wp_1.py").write_text("def build_wp_1(): return 'ok'\n")
        (root / "rogue.py").write_text("SHOULD_NOT_ESCAPE = True\n")

    monkeypatch.setattr(main.cc_agent, "run_test_author", test_author)
    monkeypatch.setattr(main.cc_agent, "run_executor", executor)

    result = asyncio.run(flow._build_structured(plan))
    stage = Path(flow.state.current_stage_dir)
    canonical = Path(flow.state.canonical_build_dir)

    assert result == {"route": "qa_exhausted"}
    assert (stage / "tests/test_wp_1.py").read_text() == (
        "FROZEN = True\n\ndef test_wp_1(): pass\n"
    )
    assert not (stage / "rogue.py").exists()
    assert snapshot_files(canonical) == {}
    assert {issue.owner for issue in flow.state.current_failure.issues} == {"code"}


def test_verification_commands_cannot_mutate_frozen_evidence(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"))
    flow.state.plan = plan
    monkeypatch.setattr(main, "check_spec_contract", lambda *_args, **_kwargs: "PASS")
    calls = {"count": 0}

    def verification(build_dir, _commands):
        calls["count"] += 1
        if calls["count"] > 1:
            root = Path(build_dir)
            (root / "tests/test_wp_1.py").write_text("MUTATED = True\n")
            (root / "rogue.txt").write_text("mutation\n")
        return [CommandResult(command_id="tests", passed=True, returncode=0)]

    async def test_author(*, cwd, declared_test_files, **_kwargs) -> None:
        test = Path(cwd) / declared_test_files[0]
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("FROZEN = True\n\ndef test_wp_1(): pass\n")

    async def executor(*, cwd, **_kwargs) -> None:
        source = Path(cwd) / "src/demo_pkg/wp_1.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("def build_wp_1(): return 'ok'\n")

    monkeypatch.setattr(main, "run_verification_commands", verification)
    monkeypatch.setattr(main.cc_agent, "run_test_author", test_author)
    monkeypatch.setattr(main.cc_agent, "run_executor", executor)

    result = asyncio.run(flow._build_structured(plan))
    stage = Path(flow.state.current_stage_dir)

    assert result == {"route": "qa_exhausted"}
    assert (stage / "tests/test_wp_1.py").read_text() == (
        "FROZEN = True\n\ndef test_wp_1(): pass\n"
    )
    assert not (stage / "rogue.txt").exists()
    assert snapshot_files(flow.state.canonical_build_dir) == {}
    assert flow.state.current_failure.issues[0].owner == "spec"


def test_build_mutations_are_discarded_before_the_next_command(
    flow: main.CodebuilderFlow,
) -> None:
    build = Path(flow.state.workspace_dir) / "build"
    build.mkdir()
    (build / "sentinel.txt").write_text("safe", encoding="utf-8")
    plan = _plan(_package("wp-1"))
    plan.verification_commands = [
        VerificationCommand(
            id="build",
            category="build",
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path; Path('sentinel.txt').write_text('poison')",
            ],
        ),
        VerificationCommand(
            id="test",
            category="test",
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path; assert Path('sentinel.txt').read_text() == 'safe'",
            ],
        ),
    ]

    results, issues = flow._run_protected_verification(build, plan, "isolation")

    if any(result.returncode == 127 for result in results):
        pytest.skip("verification sandbox unavailable")
    assert [result.passed for result in results] == [True, True]
    assert not issues
    assert (build / "sentinel.txt").read_text(encoding="utf-8") == "safe"


def test_preimplementation_verification_cannot_poison_last_green_or_tests(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"))
    flow.state.plan = plan

    async def test_author(*, cwd, declared_test_files, **_kwargs) -> None:
        test = Path(cwd) / declared_test_files[0]
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("FROZEN = True\n\ndef test_wp_1(): pass\n")

    def verification(build_dir, _commands):
        root = Path(build_dir)
        (root / "tests/test_wp_1.py").write_text("POISONED = True\n")
        canonical = Path(flow.state.canonical_build_dir)
        (canonical / "poison.py").write_text("POISON = True\n")
        return [CommandResult(command_id="tests", passed=False, returncode=1)]

    async def executor(**_kwargs) -> None:
        raise AssertionError("executor must not run after red evidence mutation")

    monkeypatch.setattr(main.cc_agent, "run_test_author", test_author)
    monkeypatch.setattr(main, "run_verification_commands", verification)
    monkeypatch.setattr(main.cc_agent, "run_executor", executor)

    result = asyncio.run(flow._build_structured(plan))

    assert result == {"route": "qa_exhausted"}
    assert snapshot_files(flow.state.canonical_build_dir) == {}
    assert (
        Path(flow.state.current_stage_dir) / "tests/test_wp_1.py"
    ).read_text() == "FROZEN = True\n\ndef test_wp_1(): pass\n"
    assert {issue.owner for issue in flow.state.current_failure.issues} == {"spec"}


@pytest.mark.parametrize(
    ("returncode", "owner"), [(124, "environment"), (125, "spec"), (127, "environment")]
)
def test_preimplementation_sandbox_failure_never_runs_executor(
    flow: main.CodebuilderFlow,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    owner: str,
) -> None:
    plan = _plan(_package("wp-1"))
    flow.state.plan = plan

    async def test_author(*, cwd, declared_test_files, **_kwargs) -> None:
        test = Path(cwd) / declared_test_files[0]
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("def test_wp_1(): pass\n")

    async def executor(**_kwargs) -> None:
        raise AssertionError("executor must not run after a sandbox failure")

    monkeypatch.setattr(main.cc_agent, "run_test_author", test_author)
    monkeypatch.setattr(
        main,
        "run_verification_commands",
        lambda *_args, **_kwargs: [
            CommandResult(command_id="tests", passed=False, returncode=returncode)
        ],
    )
    monkeypatch.setattr(main.cc_agent, "run_executor", executor)

    result = asyncio.run(flow._build_structured(plan))

    assert result == {"route": "qa_exhausted"}
    assert flow.state.current_failure.issues[0].owner == owner


def test_invalid_declared_test_stays_with_test_owner_and_never_runs_executor(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"))
    flow.state.plan = plan

    async def test_author(*, cwd, declared_test_files, **_kwargs) -> None:
        test = Path(cwd) / declared_test_files[0]
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("def test_wp_1(:\n")

    async def executor(**_kwargs) -> None:
        raise AssertionError("executor must not run with an invalid declared test")

    monkeypatch.setattr(main.cc_agent, "run_test_author", test_author)
    monkeypatch.setattr(main.cc_agent, "run_executor", executor)

    result = asyncio.run(flow._build_structured(plan))

    assert result == {"route": "qa_exhausted"}
    assert flow.state.current_failure is not None
    assert {issue.owner for issue in flow.state.current_failure.issues} == {"test"}
    assert (
        "cannot inspect declared test" in flow.state.current_failure.issues[0].evidence
    )


def test_dependent_package_qa_validates_an_independent_partial_spec(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"), _package("wp-2", depends_on=["wp-1"]))
    workspace = Path(flow.state.workspace_dir)
    canonical = workspace / "output"
    canonical.mkdir()
    stage = workspace / "stages/wp-2"
    (stage / "src/demo_pkg").mkdir(parents=True)
    (stage / "tests").mkdir()
    (stage / "src/demo_pkg/wp_2.py").write_text("def build_wp_2(): return 'ok'\n")
    (stage / "tests/test_wp_2.py").write_text("def test_wp_2(): pass\n")
    trusted = workspace / "trusted-tests/wp-2"
    stage_tree(workspace, stage, trusted)
    flow.state.plan = plan
    flow.state.canonical_build_dir = str(canonical)
    flow.state.approved_spec_hash = plan_spec_hash(plan)
    captured = []

    def contract(_build_dir, partial):
        validate_plan(partial)
        captured.append(partial)
        return "PASS"

    monkeypatch.setattr(main, "check_spec_contract", contract)
    monkeypatch.setattr(
        main,
        "run_verification_commands",
        lambda *_args, **_kwargs: [
            CommandResult(command_id="tests", passed=True, returncode=0)
        ],
    )

    async def reviewer(**_kwargs) -> ProductionReview:
        return ProductionReview(passed=True)

    monkeypatch.setattr(main.cc_agent, "run_reviewer", reviewer)

    report = asyncio.run(flow._structured_qa(stage, plan, plan.work_packages[1]))

    assert report.passed
    assert captured[0].work_packages[0].depends_on == []


def test_green_package_promotes_exact_files_and_reaches_final_success(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"))
    flow.state.plan = plan
    _green_qa(monkeypatch)

    async def test_author(*, cwd, declared_test_files, **_kwargs) -> None:
        test = Path(cwd) / declared_test_files[0]
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("def test_wp_1(): assert True\n")

    async def executor(*, cwd, **_kwargs) -> None:
        source = Path(cwd) / "src/demo_pkg/wp_1.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("def build_wp_1(): return 'ok'\n")

    monkeypatch.setattr(main.cc_agent, "run_test_author", test_author)
    monkeypatch.setattr(main.cc_agent, "run_executor", executor)

    result = asyncio.run(flow._build_structured(plan))

    assert result == {"route": "execution_complete"}
    assert set(snapshot_files(flow.state.canonical_build_dir)) == {
        "src/demo_pkg/wp_1.py",
        "tests/test_wp_1.py",
    }
    assert [(item.package_id, item.status) for item in flow.state.package_results] == [
        ("wp-1", "passed")
    ]
    assert flow.state.qa_report is not None
    assert flow.state.qa_report.passed
    assert flow.state.qa_report.package_id == "__final__"


def test_green_test_owner_retry_continues_to_remaining_packages(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"))
    flow.state.plan = plan
    flow.state.current_package_id = "wp-1"
    flow.state.current_failure = QAReport(
        passed=False,
        issues=[
            QAIssue(
                source="contract",
                owner="test",
                message="Test contract failed.",
            )
        ],
    )
    calls = []

    async def rerun(*_args, **_kwargs):
        return {"route": "package_complete"}

    async def continue_build(_plan):
        calls.append("continued")
        return {"route": "execution_complete"}

    monkeypatch.setattr(flow, "_run_package", rerun)
    monkeypatch.setattr(flow, "_continue_structured_build", continue_build)

    result = asyncio.run(
        flow.retry_failed_qa(type("Feedback", (), {"feedback": "retry"})())
    )

    assert result == {"route": "execution_complete"}
    assert flow.state.package_cursor == 1
    assert calls == ["continued"]


def test_structured_revision_failure_regates_the_unchanged_spec(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"))
    flow.state.plan = plan

    async def legacy_revision(**_kwargs):
        return Plan(
            project_name="demo",
            mode="new_project",
            plan_markdown="# legacy downgrade",
        )

    monkeypatch.setattr(main.cc_agent, "run_planner", legacy_revision)

    result = asyncio.run(
        main.CodebuilderFlow.revise_plan.__wrapped__(
            flow, type("Feedback", (), {"feedback": "change one detail"})()
        )
    )

    assert flow.state.plan is not None and flow.state.plan.is_structured
    assert flow.state.plan.open_questions == []
    assert "legacy plan" in result["revision_error"]


def test_authoritative_assets_are_rehashed_from_intake_evidence_before_approval(
    flow: main.CodebuilderFlow,
) -> None:
    workspace = Path(flow.state.workspace_dir)
    source = workspace / "inputs/repo"
    (source / "src").mkdir(parents=True)
    (source / "contract.sql").write_text("CREATE TABLE customer (id INT);\n")
    plan = _plan(_package("wp-1")).model_copy(
        update={
            "mode": "patch_existing",
            "authoritative_assets": [
                AuthoritativeAsset(path="contract.sql", sha256="0" * 64)
            ],
        }
    )
    intake = IntakeAssessment(
        ready=True,
        authoritative_assets=[AuthoritativeAsset(path="contract.sql")],
    )

    main._bind_authoritative_asset_hashes(plan, str(workspace), intake)

    assert plan.authoritative_assets[0].sha256 != "0" * 64
    assert len(plan.authoritative_assets[0].sha256) == 64


def test_workspace_prefixed_authoritative_asset_is_canonicalized_before_approval(
    flow: main.CodebuilderFlow,
) -> None:
    workspace = Path(flow.state.workspace_dir)
    source = workspace / "inputs/app-faturamento-automatico-terra"
    (source / "src").mkdir(parents=True)
    lock = source / "uv.lock"
    lock.write_text("version = 1\n")
    prefixed = "inputs/app-faturamento-automatico-terra/uv.lock"
    plan = _plan(_package("wp-1")).model_copy(
        update={
            "mode": "patch_existing",
            "authoritative_assets": [AuthoritativeAsset(path=prefixed)],
        }
    )
    intake = IntakeAssessment(
        ready=True,
        authoritative_assets=[AuthoritativeAsset(path=prefixed)],
    )

    main._bind_authoritative_asset_hashes(plan, str(workspace), intake)

    assert intake.authoritative_assets[0].path == "uv.lock"
    assert plan.authoritative_assets[0].path == "uv.lock"
    assert (
        plan.authoritative_assets[0].sha256
        == hashlib.sha256(lock.read_bytes()).hexdigest()
    )


def test_new_project_does_not_use_unrelated_attachment_directories_as_baseline(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Path(flow.state.workspace_dir)
    reference = workspace / "inputs/reference-material"
    reference.mkdir(parents=True)
    (reference / "do-not-copy.txt").write_text("reference only\n")
    plan = _plan(_package("wp-1"))
    flow.state.plan = plan
    _green_qa(monkeypatch)

    async def test_author(*, cwd, declared_test_files, **_kwargs) -> None:
        test = Path(cwd) / declared_test_files[0]
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("def test_wp_1(): assert True\n")

    async def executor(*, cwd, **_kwargs) -> None:
        source = Path(cwd) / "src/demo_pkg/wp_1.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("def build_wp_1(): return 'ok'\n")

    monkeypatch.setattr(main.cc_agent, "run_test_author", test_author)
    monkeypatch.setattr(main.cc_agent, "run_executor", executor)

    assert asyncio.run(flow._build_structured(plan)) == {"route": "execution_complete"}
    assert not (Path(flow.state.canonical_build_dir) / "reference-material").exists()


def test_authoritative_asset_symlink_escape_is_rejected_before_approval(
    flow: main.CodebuilderFlow, tmp_path: Path
) -> None:
    workspace = Path(flow.state.workspace_dir)
    inputs = workspace / "inputs"
    inputs.mkdir()
    outside = tmp_path / "outside.sql"
    outside.write_text("CREATE TABLE escaped (id INT);\n")
    (inputs / "contract.sql").symlink_to(outside)
    plan = _plan(_package("wp-1")).model_copy(
        update={
            "authoritative_assets": [AuthoritativeAsset(path="contract.sql")],
        }
    )

    with pytest.raises(ValueError, match="could not be resolved before approval"):
        main._bind_authoritative_asset_hashes(plan, str(workspace))


def test_executor_authoritative_asset_change_is_restored_and_reported(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Path(flow.state.workspace_dir)
    inputs = workspace / "inputs"
    inputs.mkdir()
    original = "CREATE TABLE customer (id INT);\n"
    (inputs / "contract.sql").write_text(original)
    plan = _plan(_package("wp-1")).model_copy(
        update={
            "authoritative_assets": [
                AuthoritativeAsset(
                    path="contract.sql",
                    sha256=hashlib.sha256(original.encode()).hexdigest(),
                )
            ],
        }
    )
    flow.state.plan = plan
    _green_qa(monkeypatch)

    async def test_author(*, cwd, declared_test_files, **_kwargs) -> None:
        test = Path(cwd) / declared_test_files[0]
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("def test_wp_1(): assert True\n")

    async def executor(*, cwd, **_kwargs) -> None:
        root = Path(cwd)
        (root / "contract.sql").write_text("MUTATED\n")
        source = root / "src/demo_pkg/wp_1.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("def build_wp_1(): return 'ok'\n")

    monkeypatch.setattr(main.cc_agent, "run_test_author", test_author)
    monkeypatch.setattr(main.cc_agent, "run_executor", executor)

    assert asyncio.run(flow._build_structured(plan)) == {"route": "qa_exhausted"}
    assert (Path(flow.state.current_stage_dir) / "contract.sql").read_text() == original
    assert (
        Path(flow.state.canonical_build_dir) / "contract.sql"
    ).read_text() == original
    assert flow.state.current_failure is not None
    issue = next(
        issue
        for issue in flow.state.current_failure.issues
        if "authoritative assets" in issue.message
    )
    assert issue.owner == "code"
    assert issue.evidence == "contract.sql"


def test_empty_setup_failure_creates_report_only_quarantine(
    flow: main.CodebuilderFlow,
) -> None:
    plan = _plan(_package("wp-1")).model_copy(update={"mode": "patch_existing"})
    flow.state.plan = plan

    assert asyncio.run(flow._build_structured(plan)) == {"route": "qa_exhausted"}
    assert flow.state.canonical_build_dir == ""
    assert flow.state.current_stage_dir == ""
    assert flow.terminate_failed_qa() == {"route": "quarantine_ready"}
    assert flow.state.project_archive is None
    assert flow.state.quarantine_archive is not None

    with zipfile.ZipFile(flow.state.quarantine_archive.local_path) as archive:
        names = set(archive.namelist())
    assert names == {
        "evidence/approved-spec.json",
        "evidence/QA.md",
        "evidence/failed-stage-files.json",
    }


def test_staging_failure_mid_dag_names_the_package_it_failed_on(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # stage_tree runs unguarded inside _run_package, so this really does escape
    # to build()'s handler. It used to escape with the *previous* package's id
    # still in state, and the gate reads that for can_skip / _package_by_id —
    # so the human would be offered to skip a package that had already passed.
    plan = _plan(_package("wp-1"), _package("wp-2", depends_on=["wp-1"]))
    flow.state.plan = plan
    flow.state.canonical_build_dir = str(Path(flow.state.workspace_dir) / "output")
    flow.state.current_package_id = "wp-0-already-green"

    def boom(*_args, **_kwargs):
        raise main.package_workspace.WorkspaceSafetyError("symlink escapes the stage")

    monkeypatch.setattr(main.package_workspace, "stage_tree", boom)

    with pytest.raises(main.package_workspace.WorkspaceSafetyError):
        asyncio.run(flow._continue_structured_build(plan))

    assert flow.state.current_package_id == "wp-1"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is POSIX-only")
def test_special_file_in_patch_baseline_fails_at_setup_without_blocking(
    flow: main.CodebuilderFlow,
) -> None:
    source = Path(flow.state.workspace_dir) / "inputs/repo"
    (source / "src").mkdir(parents=True)
    os.mkfifo(source / "pipe")
    plan = _plan(_package("wp-1")).model_copy(update={"mode": "patch_existing"})
    flow.state.plan = plan

    assert asyncio.run(flow._build_structured(plan)) == {"route": "qa_exhausted"}
    assert flow.state.current_package_id == "__setup__"
    assert flow.state.current_failure is not None
    assert "special filesystem node" in flow.state.current_failure.issues[0].evidence


def test_termination_creates_quarantine_with_last_green_spec_and_qa_only(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"), _package("wp-2", depends_on=["wp-1"]))
    workspace = Path(flow.state.workspace_dir)
    canonical = workspace / "output"
    (canonical / "src/demo_pkg").mkdir(parents=True)
    (canonical / "tests").mkdir()
    (canonical / "src/demo_pkg/wp_1.py").write_text("GREEN = True\n")
    (canonical / "tests/test_wp_1.py").write_text("def test_green(): pass\n")
    failed_stage = stage_tree(workspace, canonical, workspace / "stages/wp-2")
    (failed_stage / "src/demo_pkg/wp_2.py").write_text("BROKEN = True\n")
    (failed_stage / "tests/test_wp_2.py").write_text(
        "def test_broken(): assert False\n"
    )
    issue = QAIssue(
        source="command",
        owner="code",
        message="Approved test failed.",
        evidence="1 failed",
    )
    failure = QAReport(passed=False, package_id="wp-2", issues=[issue])
    spec_hash = plan_spec_hash(plan)
    flow.state.plan = plan
    flow.state.canonical_build_dir = str(canonical)
    flow.state.current_stage_dir = str(failed_stage)
    flow.state.current_package_id = "wp-2"
    flow.state.approved_spec_hash = spec_hash
    flow.state.current_failure = failure
    flow.state.qa_report = failure
    flow.state.package_results = [
        PackageResult(package_id="wp-1", spec_hash=spec_hash, status="passed"),
        PackageResult(
            package_id="wp-2",
            spec_hash=spec_hash,
            status="failed",
            issues=[issue],
        ),
    ]
    monkeypatch.setattr(main, "upload_file", lambda *_args, **_kwargs: None)

    assert flow.terminate_failed_qa() == {"route": "quarantine_ready"}
    completion = asyncio.run(flow.finalize())

    assert flow.state.quarantine_archive is not None
    assert flow.state.project_archive is None
    assert "project_archive" not in completion
    with zipfile.ZipFile(flow.state.quarantine_archive.local_path) as archive:
        names = set(archive.namelist())
        assert "last-green/src/demo_pkg/wp_1.py" in names
        assert "failed-stage/src/demo_pkg/wp_2.py" in names
        assert "evidence/approved-spec.json" in names
        assert "evidence/QA.md" in names
        assert json.loads(archive.read("evidence/approved-spec.json"))["revision"] == 1
        assert "Approved test failed." in archive.read("evidence/QA.md").decode()


def test_skip_routes_directly_to_quarantine_without_full_missing_file_qa(
    flow: main.CodebuilderFlow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(_package("wp-1"), _package("wp-2", depends_on=["wp-1"]))
    workspace = Path(flow.state.workspace_dir)
    canonical = workspace / "output"
    canonical.mkdir()
    failed_stage = stage_tree(workspace, canonical, workspace / "stages/wp-1")
    (failed_stage / "tests").mkdir()
    (failed_stage / "tests/test_wp_1.py").write_text("def test_wp_1(): assert False\n")
    issue = QAIssue(source="command", owner="code", message="failed")
    failure = QAReport(passed=False, package_id="wp-1", issues=[issue])
    flow.state.plan = plan
    flow.state.canonical_build_dir = str(canonical)
    flow.state.current_stage_dir = str(failed_stage)
    flow.state.current_package_id = "wp-1"
    flow.state.approved_spec_hash = plan_spec_hash(plan)
    flow.state.current_failure = failure
    flow.state.qa_report = failure
    flow.state.package_results = [
        PackageResult(
            package_id="wp-1",
            spec_hash=flow.state.approved_spec_hash,
            status="failed",
            issues=[issue],
        )
    ]
    monkeypatch.setattr(
        flow,
        "_structured_qa",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full QA must not run after a human skip")
        ),
    )

    result = asyncio.run(flow.skip_failed_package())

    assert result == {"route": "quarantine_ready"}
    assert flow.state.skipped_package_ids == ["wp-1", "wp-2"]
    assert flow.state.quarantine_archive is not None
