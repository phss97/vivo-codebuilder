"""Filesystem safety checks for package staging and promotion."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from codebuilder.package_workspace import (
    WorkspaceSafetyError,
    changed_paths,
    copy_clean_tree,
    create_quarantine_zip,
    harden_tree,
    promote_files,
    restore_files,
    snapshot_files,
    stage_tree,
    validate_changed_paths,
)


def _project(workspace, name="canonical"):
    project = workspace / name
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("VALUE = 1\n")
    (project / "tests").mkdir()
    (project / "tests" / "test_app.py").write_text("def test_app(): ...\n")
    return project


def test_stage_tree_resets_and_excludes_agent_cache_artifacts_and_secrets(tmp_path):
    canonical = _project(tmp_path)
    for directory in (".git", ".claude", ".venv", "artifacts", "dist"):
        path = canonical / directory
        path.mkdir()
        (path / "ignored.txt").write_text("ignored")
    (canonical / ".env").write_text("API_KEY=secret\n")
    (canonical / ".env.example").write_text("API_KEY=\n")
    (canonical / "old.zip").write_bytes(b"archive")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (canonical / "linked.txt").symlink_to(outside)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "stale.py").write_text("stale")

    result = stage_tree(tmp_path, canonical, stage)

    assert result == stage
    assert (stage / "src" / "app.py").read_text() == "VALUE = 1\n"
    assert (stage / ".env.example").is_file()
    assert not (stage / "stale.py").exists()
    for excluded in (
        ".git",
        ".claude",
        ".venv",
        "artifacts",
        "dist",
        ".env",
        "old.zip",
        "linked.txt",
    ):
        assert not (stage / excluded).exists()


def test_stage_tree_rejects_root_source_nested_outside_and_symlink_targets(tmp_path):
    canonical = _project(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    linked_stage = tmp_path / "linked-stage"
    linked_stage.symlink_to(other, target_is_directory=True)

    for unsafe in (
        tmp_path,
        canonical,
        canonical / "nested-stage",
        tmp_path.parent / "outside-stage",
        linked_stage,
    ):
        with pytest.raises(WorkspaceSafetyError):
            stage_tree(tmp_path, canonical, unsafe)

    assert (canonical / "src" / "app.py").is_file()
    assert other.is_dir()


def test_stage_tree_accepts_system_tmp_alias_but_rejects_job_symlink_escape():
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        workspace = Path(temporary)
        canonical = _project(workspace)

        stage = stage_tree(workspace, canonical, workspace / "stage")

        assert (stage / "src/app.py").is_file()
        escape = workspace / "escape"
        escape.symlink_to(workspace.parent, target_is_directory=True)
        with pytest.raises(WorkspaceSafetyError, match="symlink"):
            stage_tree(workspace, canonical, escape / "stage")


def test_snapshot_and_changed_paths_are_deterministic_and_ignore_noise(tmp_path):
    project = _project(tmp_path)
    before = snapshot_files(project)

    assert list(before) == sorted(before)
    assert list(before) == ["src/app.py", "tests/test_app.py"]

    (project / "src" / "app.py").write_text("VALUE = 2\n")
    (project / "src" / "new.py").write_text("NEW = True\n")
    (project / "tests" / "test_app.py").unlink()
    cache = project / ".pytest_cache"
    cache.mkdir()
    (cache / "noise").write_text("changes do not count")

    after = snapshot_files(project)

    assert changed_paths(before, after) == [
        "src/app.py",
        "src/new.py",
        "tests/test_app.py",
    ]


def test_snapshot_detects_chmod_only_change(tmp_path):
    project = _project(tmp_path)
    before = snapshot_files(project)

    (project / "src/app.py").chmod(0o755)

    assert changed_paths(before, snapshot_files(project)) == ["src/app.py"]


def test_validate_changed_paths_requires_exact_safe_allowlist():
    assert validate_changed_paths(
        ["tests/test_api.py"], ["tests/test_api.py", "tests/test_other.py"]
    ) == ["tests/test_api.py"]

    with pytest.raises(WorkspaceSafetyError, match="outside exact allowlist"):
        validate_changed_paths(["src/api.py"], ["src"])
    with pytest.raises(WorkspaceSafetyError, match="unsafe relative path"):
        validate_changed_paths(["../src/api.py"], ["../src/api.py"])
    with pytest.raises(WorkspaceSafetyError, match="sensitive"):
        validate_changed_paths([".env"], [".env"])


def test_restore_files_restores_content_and_removes_untrusted_additions(tmp_path):
    trusted = _project(tmp_path, "trusted")
    target = _project(tmp_path, "target")
    (target / "tests" / "test_app.py").write_text("weakened = True\n")
    (target / "tests" / "test_extra.py").write_text("untrusted\n")

    restored = restore_files(
        tmp_path,
        target,
        trusted,
        ["tests/test_app.py", "tests/test_extra.py"],
    )

    assert restored == ["tests/test_app.py", "tests/test_extra.py"]
    assert (target / "tests" / "test_app.py").read_text() == "def test_app(): ...\n"
    assert not (target / "tests" / "test_extra.py").exists()

    with pytest.raises(WorkspaceSafetyError):
        restore_files(tmp_path, tmp_path, trusted, ["tests/test_app.py"])


def test_restore_files_replaces_symlink_and_directory_path_drift(tmp_path):
    trusted = _project(tmp_path, "trusted")
    target = _project(tmp_path, "target")
    outside = tmp_path / "outside.py"
    outside.write_text("outside\n")
    (target / "src/app.py").unlink()
    (target / "src/app.py").symlink_to(outside)

    restore_files(tmp_path, target, trusted, ["src/app.py"])

    assert not (target / "src/app.py").is_symlink()
    assert (target / "src/app.py").read_text() == "VALUE = 1\n"
    assert outside.read_text() == "outside\n"

    (target / "src/app.py").unlink()
    (target / "src/app.py").mkdir()
    (target / "src/app.py/evil.py").write_text("evil\n")

    restore_files(tmp_path, target, trusted, ["src/app.py", "src/app.py/evil.py"])

    assert (target / "src/app.py").is_file()
    assert (target / "src/app.py").read_text() == "VALUE = 1\n"


def test_restore_files_recovers_directory_and_descendants_replaced_by_symlink(
    tmp_path,
):
    trusted = _project(tmp_path, "trusted")
    target = _project(tmp_path, "target")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "app.py").write_text("outside\n")
    before = snapshot_files(target)
    (target / "src/app.py").unlink()
    (target / "src").rmdir()
    (target / "src").symlink_to(outside, target_is_directory=True)

    drift = changed_paths(before, snapshot_files(target))
    restore_files(tmp_path, target, trusted, drift)

    assert drift == ["src", "src/app.py"]
    assert (target / "src").is_dir()
    assert not (target / "src").is_symlink()
    assert (target / "src/app.py").read_text() == "VALUE = 1\n"
    assert (outside / "app.py").read_text() == "outside\n"


def test_harden_tree_and_clean_copy_strip_write_bits_but_keep_execute(tmp_path):
    project = _project(tmp_path)
    executable = project / "src/app.py"
    executable.chmod(0o777)
    (project / "tests").chmod(0o777)

    copied = copy_clean_tree(project, tmp_path / "copy")
    hardened = harden_tree(project)

    assert hardened == project.resolve()
    for tree in (project, copied):
        assert (tree / "src/app.py").stat().st_mode & 0o777 == 0o755
        assert (tree / "tests").stat().st_mode & 0o777 == 0o755


def test_promote_files_fails_closed_then_copies_only_allowlisted_diff(tmp_path):
    canonical = _project(tmp_path)
    (canonical / "src" / "obsolete.py").write_text("OLD = True\n")
    stage = stage_tree(tmp_path, canonical, tmp_path / "stage")
    (stage / "src" / "app.py").write_text("VALUE = 2\n")
    (stage / "src" / "app.py").chmod(0o777)
    (stage / "src" / "obsolete.py").unlink()
    (stage / "README.md").write_text("unapproved\n")

    allowlist = ["src/app.py", "src/obsolete.py"]
    with pytest.raises(WorkspaceSafetyError, match="README.md"):
        promote_files(tmp_path, stage, canonical, allowlist)
    assert (canonical / "src" / "app.py").read_text() == "VALUE = 1\n"
    assert (canonical / "src" / "obsolete.py").is_file()

    (stage / "README.md").unlink()
    promoted = promote_files(tmp_path, stage, canonical, allowlist)

    assert promoted == ["src/app.py", "src/obsolete.py"]
    assert (canonical / "src" / "app.py").read_text() == "VALUE = 2\n"
    assert (canonical / "src" / "app.py").stat().st_mode & 0o777 == 0o755
    assert not (canonical / "src" / "obsolete.py").exists()


def test_quarantine_zip_contains_only_last_green_approved_failure_and_evidence(
    tmp_path,
):
    canonical = _project(tmp_path)
    (canonical / ".env").write_text("TOKEN=secret\n")
    cache = canonical / ".mypy_cache"
    cache.mkdir()
    (cache / "noise").write_text("cache")
    failed = stage_tree(tmp_path, canonical, tmp_path / "failed")
    (failed / "src" / "app.py").write_text("VALUE = 'broken'\n")
    (failed / "src" / "new.py").write_text("NEW = 'broken'\n")
    (failed / "unapproved.py").write_text("do not package\n")

    output = create_quarantine_zip(
        tmp_path,
        canonical,
        failed,
        ["src/app.py", "src/new.py", "src/missing.py"],
        '{"revision":2,"project":"demo"}',
        "# QA\n\nTests failed.\n",
        tmp_path / "quarantine.zip",
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "last-green/src/app.py" in names
        assert "failed-stage/src/app.py" in names
        assert "failed-stage/src/new.py" in names
        assert "failed-stage/unapproved.py" not in names
        assert all(".env" not in name and ".mypy_cache" not in name for name in names)
        assert archive.read("evidence/QA.md").decode() == "# QA\n\nTests failed.\n"
        assert json.loads(archive.read("evidence/approved-spec.json")) == {
            "project": "demo",
            "revision": 2,
        }
        assert json.loads(archive.read("evidence/failed-stage-files.json")) == [
            {"path": "src/app.py", "status": "included"},
            {"path": "src/missing.py", "status": "missing"},
            {"path": "src/new.py", "status": "included"},
        ]


def test_quarantine_zip_rejects_destination_inside_source_or_workspace_escape(
    tmp_path,
):
    canonical = _project(tmp_path)
    failed = stage_tree(tmp_path, canonical, tmp_path / "failed")

    for destination in (
        canonical / "quarantine.zip",
        failed / "quarantine.zip",
        tmp_path.parent / "quarantine.zip",
    ):
        with pytest.raises(WorkspaceSafetyError):
            create_quarantine_zip(
                tmp_path,
                canonical,
                failed,
                [],
                "{}",
                "QA failed",
                destination,
            )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is POSIX-only")
def test_special_files_are_recorded_without_opening_and_rejected_for_copy(tmp_path):
    canonical = _project(tmp_path)
    os.mkfifo(canonical / "pipe")

    assert snapshot_files(canonical)["pipe"].startswith("special:")
    with pytest.raises(WorkspaceSafetyError, match="special filesystem node"):
        stage_tree(tmp_path, canonical, tmp_path / "stage")
