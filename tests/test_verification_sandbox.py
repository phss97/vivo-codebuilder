from __future__ import annotations

import shutil
import socket
import sys
import time

import codebuilder.runtime_qa as runtime_qa
import pytest
from codebuilder.runtime_qa import (
    check_declared_tests,
    check_spec_contract,
    run_verification_command,
)
from codebuilder.schemas import Plan, VerificationCommand, WorkPackageSpec


def _package() -> WorkPackageSpec:
    return WorkPackageSpec.model_validate(
        {
            "id": "core",
            "title": "Core",
            "what_to_build": "Build the approved core behavior.",
            "expected_behavior": "The core behavior works.",
            "success_criteria": [{"id": "works", "description": "It works."}],
            "tests": [
                {
                    "id": "test-core",
                    "criterion_ids": ["works"],
                    "path": "tests/test_core.py",
                    "test_name": "test_core",
                    "expected_behavior": "The core behavior is verified.",
                }
            ],
            "files": [
                {
                    "path": "src/demo/core.py",
                    "purpose": "Core implementation.",
                    "public_api": [],
                },
                {
                    "path": "tests/test_core.py",
                    "purpose": "Core test.",
                    "kind": "test",
                    "public_api": [],
                },
            ],
        }
    )


def test_verification_sandbox_blocks_absolute_outside_write(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "canonical.txt"
    outside.write_text("safe", encoding="utf-8")
    command = VerificationCommand(
        id="escape",
        category="test",
        argv=[
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('bad')",
            str(outside),
        ],
    )

    result = run_verification_command(str(project), command)

    assert not result.passed
    assert outside.read_text(encoding="utf-8") == "safe"


def test_verification_sandbox_blocks_absolute_outside_read(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("do-not-expose", encoding="utf-8")
    command = VerificationCommand(
        id="read-escape",
        category="test",
        argv=[
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text())",
            str(outside),
        ],
    )

    result = run_verification_command(str(project), command)

    assert not result.passed
    assert "do-not-expose" not in result.stdout


def test_verification_sandbox_blocks_outside_metadata_read(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    result = run_verification_command(
        str(project),
        VerificationCommand(
            id="metadata-escape",
            category="test",
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; print(Path(sys.argv[1]).stat().st_size)",
                str(outside),
            ],
        ),
    )

    assert not result.passed
    assert result.stdout.strip() != "6"


def test_verification_sandbox_runs_normal_command_or_fails_closed(tmp_path):
    command = VerificationCommand(
        id="normal",
        category="test",
        argv=[sys.executable, "-c", "print('ok')"],
    )

    result = run_verification_command(str(tmp_path), command)

    if result.returncode == 127:
        assert "sandbox unavailable" in result.stderr.lower()
    else:
        assert result.passed, result.stderr
        assert result.stdout.strip() == "ok"


def test_verification_sandbox_runs_direct_pytest(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok(): assert True\n")
    result = run_verification_command(
        str(tmp_path),
        VerificationCommand(
            id="pytest",
            category="test",
            argv=[sys.executable, "-m", "pytest", "-q"],
        ),
    )

    if result.returncode == 127:
        assert "sandbox unavailable" in result.stderr.lower()
    else:
        assert result.passed, result.stderr


def test_verification_output_capture_is_bounded(tmp_path):
    result = run_verification_command(
        str(tmp_path),
        VerificationCommand(
            id="noisy",
            category="test",
            argv=[sys.executable, "-c", "print('x' * 500_000)"],
        ),
    )

    if result.returncode == 127:
        assert "sandbox unavailable" in result.stderr.lower()
    else:
        assert result.passed
        assert len(result.stdout) <= runtime_qa.MAX_QA_OUTPUT_CHARS
        assert "truncated" in result.stdout


def test_verification_timeout_is_reported(tmp_path):
    started = time.monotonic()
    result = run_verification_command(
        str(tmp_path),
        VerificationCommand(
            id="timeout",
            category="test",
            argv=[sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_seconds=1,
        ),
    )

    if result.returncode == 127:
        assert "sandbox unavailable" in result.stderr.lower()
    else:
        assert result.returncode == 124
        assert result.timed_out
        assert time.monotonic() - started < 3


def test_verification_project_is_read_only_including_dotenv(tmp_path):
    result = run_verification_command(
        str(tmp_path),
        VerificationCommand(
            id="dotenv-write",
            category="test",
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path; Path('.env').write_text('POISON=1')",
            ],
        ),
    )

    assert not result.passed
    assert not (tmp_path / ".env").exists()


def test_build_writes_only_to_disposable_project(tmp_path):
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = run_verification_command(
        str(tmp_path),
        VerificationCommand(
            id="compile",
            category="build",
            argv=[sys.executable, "-m", "compileall", "-q", "."],
        ),
    )

    if result.returncode == 127:
        assert "sandbox unavailable" in result.stderr.lower()
    else:
        assert result.passed, result.stderr
        assert any("__pycache__" in path for path in result.mutated_paths)
        assert not (tmp_path / "__pycache__").exists()


def test_verification_network_requires_explicit_spec_capability(tmp_path):
    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
    except PermissionError:
        listener.close()
        pytest.skip("host sandbox forbids loopback sockets")
    listener.listen()
    port = listener.getsockname()[1]
    argv = [
        sys.executable,
        "-c",
        "import socket,sys; socket.create_connection(('127.0.0.1', int(sys.argv[1])))",
        str(port),
    ]
    try:
        blocked = run_verification_command(
            str(tmp_path),
            VerificationCommand(id="offline", category="test", argv=argv),
        )
        allowed = run_verification_command(
            str(tmp_path),
            VerificationCommand(
                id="online", category="integration", argv=argv, network=True
            ),
        )
    finally:
        listener.close()

    assert not blocked.passed
    assert allowed.passed or allowed.returncode == 127


def test_linux_bwrap_drops_caps_and_does_not_mount_the_host_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    qa_home = tmp_path / "qa"
    project.mkdir()
    qa_home.mkdir()
    monkeypatch.setattr(runtime_qa.sys, "platform", "linux")
    original_which = shutil.which
    monkeypatch.setattr(
        runtime_qa.shutil,
        "which",
        lambda name: "/usr/bin/bwrap" if name == "bwrap" else original_which(name),
    )

    argv, *_rest = runtime_qa._verification_sandbox(
        project,
        project,
        qa_home,
        [sys.executable, "-c", "print('ok')"],
        network=False,
        writable=False,
    )

    assert ["--cap-drop", "ALL"] == argv[
        argv.index("--cap-drop") : argv.index("--cap-drop") + 2
    ]
    assert "--unshare-user" in argv
    assert "--unshare-pid" in argv
    assert "--unshare-net" in argv
    root_index = argv.index(str(project))
    assert argv[root_index - 1] == "--ro-bind"
    assert not any(
        argv[index : index + 3] == ["--ro-bind", "/", "/"]
        for index in range(len(argv) - 2)
    )


def test_macos_profile_denies_fork_and_network_by_default(tmp_path, monkeypatch):
    project = tmp_path / "project"
    qa_home = tmp_path / "qa"
    project.mkdir()
    qa_home.mkdir()
    monkeypatch.setattr(runtime_qa.sys, "platform", "darwin")
    original_which = shutil.which
    monkeypatch.setattr(
        runtime_qa.shutil,
        "which",
        lambda name: (
            "/usr/bin/sandbox-exec" if name == "sandbox-exec" else original_which(name)
        ),
    )

    argv, *_rest = runtime_qa._verification_sandbox(
        project,
        project,
        qa_home,
        [sys.executable, "-c", "print('ok')"],
        network=False,
        writable=False,
    )
    profile = argv[2]

    assert "(allow process-exec)" in profile
    assert "process-fork" not in profile
    assert "(allow network*)" not in profile
    assert "(allow file-read-metadata)" not in profile
    assert f'(literal "{project.parent}")' in profile
    assert f'(subpath "{project}")' not in profile.split("(allow file-write*", 1)[1]


@pytest.mark.parametrize("relative", ["source.py", ".env"])
def test_actual_execution_tree_mutation_fails_even_when_sandbox_uses_a_copy(
    tmp_path, monkeypatch, relative
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "source.py").write_text("SAFE = True\n", encoding="utf-8")

    def private_copy(root, cwd, qa_home, argv, *, network, writable):
        copied = qa_home / "private-project"
        shutil.copytree(root, copied)
        return argv, copied / cwd.relative_to(root), copied, {}, "test", ""

    monkeypatch.setattr(runtime_qa, "_verification_sandbox", private_copy)
    command = VerificationCommand(
        id="mutate-copy",
        category="test",
        argv=[
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('bad')",
            relative,
        ],
    )

    result = run_verification_command(str(project), command)

    assert not result.passed
    assert result.returncode == 126
    assert result.mutated_paths == [relative]
    assert (project / "source.py").read_text(encoding="utf-8") == "SAFE = True\n"
    assert not (project / ".env").exists()


def test_verification_reaps_background_process_group(tmp_path):
    marker = tmp_path / "late-write"
    child = (
        "import time; from pathlib import Path; "
        f"time.sleep(0.4); Path({str(marker)!r}).write_text('late')"
    )
    parent = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1]], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)"
    )
    run_verification_command(
        str(tmp_path),
        VerificationCommand(
            id="background",
            category="test",
            argv=[sys.executable, "-c", parent, child],
        ),
    )
    time.sleep(0.6)

    assert not marker.exists()


def test_declared_tests_reject_syntax_and_exact_name_drift(tmp_path):
    package = _package()
    tests = tmp_path / "tests"
    tests.mkdir()
    path = tests / "test_core.py"
    path.write_text("def test_core(:\n", encoding="utf-8")

    assert "cannot inspect declared test" in check_declared_tests(
        str(tmp_path), package
    )

    path.write_text("def test_other():\n    pass\n", encoding="utf-8")
    assert "exact test missing: test_core" in check_declared_tests(
        str(tmp_path), package
    )


def test_declared_non_python_test_requires_exact_approved_name(tmp_path):
    package = _package()
    package.tests[0].path = "tests/core.test.js"
    package.tests[0].test_name = "buildsCore"
    (tmp_path / "tests").mkdir()
    path = tmp_path / package.tests[0].path
    path.write_text("test('other', () => {});\n", encoding="utf-8")

    assert "exact test missing: buildsCore" in check_declared_tests(
        str(tmp_path), package
    )


def test_package_identifier_must_be_a_real_directory(tmp_path):
    package = _package()
    (tmp_path / "src/demo").mkdir(parents=True)
    (tmp_path / "src/demo/core.py").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_core.py").write_text(
        "def test_core():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "src/alias").symlink_to(tmp_path / "src/demo", target_is_directory=True)
    plan = Plan.model_validate(
        {
            "project_name": "Demo",
            "mode": "new_project",
            "tech_stack": ["python"],
            "package_name": "demo",
            "identifier_contract": {"packages": ["demo", "alias"]},
            "work_packages": [package.model_dump()],
            "verification_commands": [
                {
                    "id": "tests",
                    "category": "test",
                    "argv": ["pytest", "-q"],
                }
            ],
        }
    )
    plan.tech_stack = ["Python 3.12"]

    output = check_spec_contract(str(tmp_path), plan)

    assert "Package drift: expected exact package 'alias'" in output
