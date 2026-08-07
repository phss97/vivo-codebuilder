"""Deterministic, package-level QA for attached and generated Python projects."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import selectors
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from codebuilder.package_workspace import (
    WorkspaceSafetyError,
    changed_paths,
    copy_clean_tree,
    snapshot_files,
)
from codebuilder.schemas import (
    ArtifactRef,
    CommandResult,
    Plan,
    QAIssue,
    QAReport,
    VerificationCommand,
    WorkPackageSpec,
)
from codebuilder.tools import LintRunnerTool, TestRunnerTool, TypeCheckRunnerTool
from codebuilder.tools.project_env import (
    ensure_project_env,
    project_python,
    provisioning_enabled,
)

MAX_QA_OUTPUT_CHARS = 12000
MAX_PROMPT_SECTION_CHARS = 6000

_QA_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
_DEPENDENCY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_ENV_FENCE = re.compile(r"```(?:dotenv|env)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ENV_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)
_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PACKAGE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
# Only real DDL: identifier_contract.fields is non-empty on most plans and would
# demand a parity test with no schema to check against.
_SCHEMA_PATH = re.compile(r"(\.sql$|(^|/)migrations?/)", re.IGNORECASE)
_WORK_PACKAGE_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_MODULE_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ENV_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PLACEHOLDER = re.compile(
    r"\b(?:todo|tbd|placeholder|to be determined|diagnostic only|investigate only|analysis only)\b",
    re.IGNORECASE,
)
_PUBLIC_API_NAME = re.compile(
    r"^(?:(?:async\s+)?def\s+|class\s+)?([A-Za-z_][A-Za-z0-9_]*)"
)
_SAFE_ENV_KEYS = {
    "COMSPEC",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "VIRTUAL_ENV",
}


def is_pass(output: str) -> bool:
    normalized = output.strip()
    return normalized == "PASS" or normalized.startswith("PASS\n")


def is_skip(output: str) -> bool:
    return output.strip().startswith("SKIP:")


def truncate(value: str, limit: int = MAX_QA_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    for _ in range(2):
        marker = f"\n\n[truncated {omitted} chars]\n\n"
        available = max(0, limit - len(marker))
        head = available * 2 // 3
        tail = available - head
        omitted = len(value) - head - tail
    marker = f"\n\n[truncated {omitted} chars]\n\n"
    return f"{value[:head]}{marker}{value[-tail:] if tail else ''}"


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _QA_SKIP_DIRS]
        base = Path(directory)
        files.extend(base / name for name in filenames if name.endswith(".py"))
    return files


def _production_python_files(root: Path) -> list[Path]:
    """Return project source files, excluding tests and generated/build trees."""
    source_root = root / "src" if (root / "src").is_dir() else root
    return [
        path
        for path in _python_files(source_root)
        if "tests" not in path.relative_to(source_root).parts
    ]


def artifact_refs(refs: list[dict] | list[ArtifactRef] | None) -> list[ArtifactRef]:
    return [
        ref if isinstance(ref, ArtifactRef) else ArtifactRef(**ref)
        for ref in refs or []
    ]


def _safe_relative_path(value: str, *, allow_dot: bool = False) -> str | None:
    """Return a normalized POSIX path only when the input is already safe."""
    stripped = value.strip()
    if not stripped or "\\" in stripped or re.match(r"^[A-Za-z]:", stripped):
        return None
    path = PurePosixPath(stripped)
    if path.is_absolute() or ".." in path.parts:
        return None
    normalized = path.as_posix()
    if normalized == ".":
        return "." if allow_dot and stripped == "." else None
    return normalized if normalized == stripped else None


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            duplicates.append(value)
        seen.add(key)
    return duplicates


def _public_api_name(declaration: str) -> str | None:
    match = _PUBLIC_API_NAME.match(declaration.strip())
    return match.group(1) if match else None


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"


def _declared_signature(declaration: str) -> str | None:
    value = declaration.strip()
    if "(" not in value:
        return None
    source = value if value.startswith(("def ", "async def ")) else f"def {value}"
    try:
        node = ast.parse(
            source + (" ..." if source.rstrip().endswith(":") else ": ...")
        )
    except SyntaxError:
        return None
    function = node.body[0] if node.body else None
    return (
        _function_signature(function)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        else None
    )


def plan_spec_hash(plan: Plan) -> str:
    """Hash the canonical structured spec, excluding its derived Markdown view."""
    payload = plan.model_dump(mode="json", exclude={"plan_markdown"})
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


# ponytail: prose form of the rules enforced immediately below, inlined into the
# planner prompt by main._planner_prompt so there is one place to edit instead of
# two files that silently drift. It is adjacency, not derivation — a rule change
# still has to touch both halves, but they are now on the same screen. Encode the
# rules as data and generate this text only if the list keeps growing.
PLANNER_CONTRACT_RULES = """- `open_questions` must be empty. A plan that still carries one is rejected
  outright, so state an `assumptions` entry instead and plan around it.
- `work_packages`: unique ids, no self-dependency, no dependency on an id that
  does not exist, and no cycles. Titles and descriptions must be real, not
  `TODO`/`TBD` placeholders.
- Every owned file is declared exactly once across all packages, with a
  non-empty `purpose` and a relative path that never escapes the project root.
- Success-criterion ids are unique across the whole plan and every one of them
  is covered by at least one test case. Test ids are unique too.
- Every test must own a file declared in the same package with `kind="test"`,
  reference only criterion ids from that package, and set both `test_name` and
  `expected_behavior`.
- Every Python file that exports anything must list it in `public_api`, spelled
  exactly as the code will spell it — a bare name or a full signature
  (`build_invoice(path: str) -> Path`). That list is the only check that catches a
  renamed `create_registro` → `build_registro`, and a file with an empty
  `public_api` is exempt from it, so leave it empty only for a file that genuinely
  exports nothing. Never a synonym or a translation of the real name.
- `verification_commands`: unique ids, non-empty shell-free argv arrays, a
  relative `cwd`, and at least one entry with `required=true` and
  `category="test"`. Keep `network=false` unless the command truly needs it.
  Only `category="build"` may write to its disposable verification copy.
- When the plan declares or depends on a DDL/schema/migration file (`*.sql` or
  under `migrations/`), exactly one test case must assert the model/ORM field
  names match it column-for-column, with `verifies_schema` set to that path.
  This is what catches a translated `nome` living beside a correct `job_name`.
- `authoritative_assets`: safe relative paths, no repeats, and either a real
  64-character SHA-256 or none at all.
- `terminology` holds only repeated human-facing prose, one entry per canonical
  term, with non-empty translations. Machine identifiers, external literals,
  database fields, environment variables, and entry points belong in
  `identifier_contract` and are never translated.
- For `new_project`, every identifier must be valid English ASCII:
  `package_name` and `identifier_contract.packages` lowercase
  `[a-z][a-z0-9_]*`, `modules` dotted, `symbols`/`fields` plain identifiers,
  `environment_variables` `[A-Z][A-Z0-9_]*`, `entry_points` `name=module:attr`.
  Patch jobs preserve the existing names byte-for-byte instead."""


def validate_plan(plan: Plan | None) -> Plan:
    """Validate legacy plans or the complete revisioned specification contract."""
    if not isinstance(plan, Plan):
        raise ValueError("Planner did not return a valid Plan object.")
    issues: list[str] = []
    if not plan.is_structured and not plan.plan_markdown.strip():
        issues.append("plan_markdown is empty")
    if plan.mode not in ("new_project", "patch_existing"):
        issues.append(f"invalid mode: {plan.mode!r}")
    if not plan.is_structured:
        if issues:
            raise ValueError("Invalid plan: " + "; ".join(issues))
        return plan

    if plan.open_questions:
        issues.append("open_questions must be resolved before approval")
    if not plan.package_name.strip():
        issues.append("package_name is empty")
    if not plan.work_packages:
        issues.append("work_packages is empty")
    if not plan.verification_commands:
        issues.append("verification_commands is empty")

    package_ids = [package.id for package in plan.work_packages]
    duplicate_packages = _duplicates(package_ids)
    if duplicate_packages:
        issues.append("duplicate work package ids: " + ", ".join(duplicate_packages))
    package_id_set = set(package_ids)
    file_owners: dict[str, str] = {}
    schema_paths: set[str] = set()
    schema_tests: dict[str, str] = {}
    test_ids: list[str] = []
    all_criterion_ids: list[str] = []
    for package in plan.work_packages:
        text = " ".join(
            (package.title, package.what_to_build, package.expected_behavior)
        )
        if not _WORK_PACKAGE_ID.fullmatch(package.id) or not package.title.strip():
            issues.append(f"invalid work package id/title: {package.id!r}")
        if _PLACEHOLDER.search(text):
            issues.append(f"{package.id}: placeholder-only work package")
        if not package.what_to_build.strip() or not package.expected_behavior.strip():
            issues.append(f"{package.id}: build or behavior description is empty")
        if not package.files:
            issues.append(f"{package.id}: files is empty")
        for dependency in package.depends_on:
            if dependency == package.id:
                issues.append(f"{package.id}: cannot depend on itself")
            elif dependency not in package_id_set:
                issues.append(f"{package.id}: unknown dependency {dependency!r}")

        criterion_ids = [criterion.id for criterion in package.success_criteria]
        all_criterion_ids.extend(criterion_ids)
        if not criterion_ids:
            issues.append(f"{package.id}: success_criteria is empty")
        if duplicate_criteria := _duplicates(criterion_ids):
            issues.append(
                f"{package.id}: duplicate criterion ids: "
                + ", ".join(duplicate_criteria)
            )
        if any(not item.description.strip() for item in package.success_criteria):
            issues.append(f"{package.id}: criterion description is empty")

        criterion_id_set = set(criterion_ids)
        covered: set[str] = set()
        declared_paths: dict[str, str] = {}
        for file in package.files:
            normalized = _safe_relative_path(file.path)
            if normalized is None:
                issues.append(f"{package.id}: unsafe file path {file.path!r}")
                continue
            owner = file_owners.get(normalized.casefold())
            if owner:
                issues.append(
                    f"{normalized}: owned by both {owner!r} and {package.id!r}"
                )
            else:
                file_owners[normalized.casefold()] = package.id
            declared_paths[normalized] = file.kind
            if _SCHEMA_PATH.search(normalized):
                schema_paths.add(normalized.casefold())
            if not file.purpose.strip():
                issues.append(f"{normalized}: purpose is empty")
            # ponytail: no "public_api must be present" rule — it rejected a plan for
            # omitting a key rather than for anything semantic, and check_spec_contract
            # already skips files whose public_api is empty.
            for declaration in file.public_api:
                name = _public_api_name(declaration)
                if name is None:
                    issues.append(
                        f"{normalized}: invalid public API declaration {declaration!r}"
                    )
                elif plan.mode == "new_project" and not name.isascii():
                    issues.append(f"{normalized}: non-ASCII public API {name!r}")

        for test in package.tests:
            test_ids.append(test.id)
            normalized = _safe_relative_path(test.path)
            if normalized is None:
                issues.append(f"{package.id}: unsafe test path {test.path!r}")
            elif declared_paths.get(normalized) != "test":
                issues.append(
                    f"{package.id}: test {test.id!r} must own declared test file "
                    f"{test.path!r}"
                )
            if not test.criterion_ids:
                issues.append(f"{package.id}: test {test.id!r} covers no criteria")
            unknown = set(test.criterion_ids) - criterion_id_set
            if unknown:
                issues.append(
                    f"{package.id}: test {test.id!r} references unknown criteria: "
                    + ", ".join(sorted(unknown))
                )
            covered.update(test.criterion_ids)
            if test.verifies_schema.strip():
                schema = _safe_relative_path(test.verifies_schema)
                if schema is None:
                    issues.append(
                        f"{package.id}: test {test.id!r} has unsafe verifies_schema "
                        f"{test.verifies_schema!r}"
                    )
                else:
                    schema_tests[test.id] = schema.casefold()
            if not test.test_name.strip() or not test.expected_behavior.strip():
                issues.append(f"{package.id}: test {test.id!r} is incomplete")
        uncovered = criterion_id_set - covered
        if uncovered:
            issues.append(
                f"{package.id}: criteria without tests: " + ", ".join(sorted(uncovered))
            )

    if duplicate_tests := _duplicates(test_ids):
        issues.append("duplicate test ids: " + ", ".join(duplicate_tests))
    if duplicate_criteria := _duplicates(all_criterion_ids):
        issues.append(
            "duplicate criterion ids across work packages: "
            + ", ".join(duplicate_criteria)
        )

    # Kahn's algorithm is enough here and gives one deterministic cycle error.
    remaining = {package.id: set(package.depends_on) for package in plan.work_packages}
    while ready := {package_id for package_id, deps in remaining.items() if not deps}:
        for package_id in ready:
            remaining.pop(package_id)
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    if remaining:
        issues.append(
            "cyclic work package dependencies: " + ", ".join(sorted(remaining))
        )

    command_ids = [command.id for command in plan.verification_commands]
    if duplicate_commands := _duplicates(command_ids):
        issues.append(
            "duplicate verification command ids: " + ", ".join(duplicate_commands)
        )
    if not any(
        command.required and command.category == "test"
        for command in plan.verification_commands
    ):
        issues.append("at least one required test verification command is required")
    for command in plan.verification_commands:
        if (
            not command.id.strip()
            or not command.argv
            or any(not value.strip() for value in command.argv)
        ):
            issues.append(f"verification command {command.id!r} has empty argv")
        if _safe_relative_path(command.cwd, allow_dot=True) is None:
            issues.append(f"verification command {command.id!r} has unsafe cwd")

    asset_paths: list[str] = []
    for asset in plan.authoritative_assets:
        normalized = _safe_relative_path(asset.path)
        if normalized is None:
            issues.append(f"unsafe authoritative asset path {asset.path!r}")
        else:
            asset_paths.append(normalized)
            if _SCHEMA_PATH.search(normalized):
                schema_paths.add(normalized.casefold())
        if asset.sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", asset.sha256):
            issues.append(f"{asset.path}: invalid SHA-256 digest")
    if duplicate_assets := _duplicates(asset_paths):
        issues.append("duplicate authoritative assets: " + ", ".join(duplicate_assets))

    if schema_paths:
        for test_id, schema in schema_tests.items():
            if schema not in schema_paths:
                issues.append(
                    f"test {test_id!r}: verifies_schema {schema!r} is not a declared "
                    "schema file or authoritative asset"
                )
    if schema_paths and not set(schema_tests.values()) & schema_paths:
        issues.append(
            "the declared schema (" + ", ".join(sorted(schema_paths)) + ") needs one "
            "test asserting the model/ORM field names match it, with verifies_schema "
            "set to that path"
        )

    # ponytail: the contract is a presence set — check_spec_contract asserts each
    # value appears verbatim, so a repeat enforces the same thing twice. Dedupe
    # rather than discard the whole planner run. Exact match, not casefold:
    # FAILED and failed are distinct identifiers.
    for field_name in plan.identifier_contract.__class__.model_fields:
        values = getattr(plan.identifier_contract, field_name)
        seen: set[str] = set()
        values[:] = [v for v in values if not (v in seen or seen.add(v))]
    canonical_terms = [entry.canonical for entry in plan.terminology]
    if duplicate_terms := _duplicates(canonical_terms):
        issues.append("duplicate canonical terminology: " + ", ".join(duplicate_terms))
    for entry in plan.terminology:
        if not entry.canonical.strip() or any(
            not language.strip() or not translation.strip()
            for language, translation in entry.translations.items()
        ):
            issues.append("terminology entries and translations must be non-empty")

    if plan.mode == "new_project":
        if not _PACKAGE_ID.fullmatch(plan.package_name):
            issues.append("new-project package_name must be lowercase English ASCII")
        contract = plan.identifier_contract
        for value in contract.packages:
            if not _PACKAGE_ID.fullmatch(value):
                issues.append(f"invalid new-project package identifier {value!r}")
        for value in contract.modules:
            if not _MODULE_ID.fullmatch(value):
                issues.append(f"invalid new-project module identifier {value!r}")
        for value in [*contract.symbols, *contract.fields]:
            if not _ID.fullmatch(value):
                issues.append(f"invalid new-project code identifier {value!r}")
        for value in contract.environment_variables:
            if not _ENV_ID.fullmatch(value):
                issues.append(f"invalid environment variable {value!r}")
        for value in contract.entry_points:
            if (
                not value.isascii()
                or any(char.isspace() for char in value)
                or ":" not in value
            ):
                issues.append(f"invalid entry point {value!r}")

    if issues:
        raise ValueError("Invalid plan: " + "; ".join(dict.fromkeys(issues)))
    plan.plan_markdown = plan.render_markdown()
    return plan


def _bounded_communicate(
    process: subprocess.Popen[bytes], timeout: int
) -> tuple[str, str, bool]:
    """Drain both pipes without retaining attacker-controlled output in memory."""
    selector = selectors.DefaultSelector()
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    buffers = {name: bytearray() for name in streams}
    totals = {name: 0 for name in streams}
    capture_limit = max(0, MAX_QA_OUTPUT_CHARS - 80)
    for name, stream in streams.items():
        if stream is None:
            continue
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)

    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while selector.get_map() or process.poll() is None:
            running = process.poll() is None
            remaining = deadline - time.monotonic()
            if running and remaining <= 0:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    try:
                        process.kill()
                    except OSError:
                        pass
                running = False
            events = selector.select(min(0.1, max(0.0, remaining)) if running else 0)
            if not events:
                if process.poll() is not None:
                    break
                continue
            for key, _mask in events:
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                name = key.data
                totals[name] += len(chunk)
                available = capture_limit - len(buffers[name])
                if available > 0:
                    buffers[name].extend(chunk[:available])
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
        for key in list(selector.get_map().values()):
            selector.unregister(key.fileobj)
            key.fileobj.close()
        selector.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def output(name: str) -> str:
        value = bytes(buffers[name]).decode(errors="replace")
        omitted = totals[name] - len(buffers[name])
        return value + (f"\n\n[truncated {omitted} bytes]" if omitted else "")

    return output("stdout"), output("stderr"), timed_out


def _verification_sandbox(
    root: Path,
    cwd: Path,
    qa_home: Path,
    argv: list[str],
    *,
    network: bool,
    writable: bool,
) -> tuple[list[str], Path, Path, dict[str, Any], str, str]:
    """Return a filesystem-confined command, or an error when none is available."""
    raw_executable = Path(argv[0])
    invocation = (
        Path(os.path.abspath(cwd / raw_executable))
        if "/" in argv[0]
        else Path(found).absolute()
        if (found := shutil.which(argv[0]))
        else None
    )
    executable = invocation.resolve() if invocation and invocation.exists() else None
    sandboxed_command = [str(invocation), *argv[1:]] if invocation else argv

    if sys.platform == "darwin":
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            return argv, cwd, root, {}, "", "sandbox-exec is unavailable"
        readable = [
            Path("/"),
            root,
            qa_home.resolve(),
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            *(
                Path(path)
                for path in ("/System", "/Library", "/usr", "/bin", "/sbin", "/dev")
            ),
        ]
        if Path("/opt/homebrew").is_dir():
            readable.append(Path("/opt/homebrew"))
        if executable is not None:
            readable.append(executable)
        read_rules = " ".join(
            f"({'subpath' if path != Path('/') and path.is_dir() else 'literal'} "
            f"{json.dumps(str(path))})"
            for path in dict.fromkeys(readable)
            if path.exists()
        )
        metadata_rules = " ".join(
            f"(literal {json.dumps(str(path))})"
            for path in dict.fromkeys(
                [root, *root.parents, qa_home.resolve(), *qa_home.resolve().parents]
            )
            if path.exists()
        )
        write_rules = [
            f"(subpath {json.dumps(str(qa_home.resolve()))})",
            '(literal "/dev/null")',
        ]
        if writable:
            write_rules.insert(0, f"(subpath {json.dumps(str(root))})")
        profile = " ".join(
            [
                "(version 1)",
                "(deny default)",
                "(allow process-exec) (allow process-info*)",
                f"(allow file-read* {read_rules})",
                f"(allow file-read-metadata {metadata_rules})",
                "(allow sysctl-read)",
                "(allow mach-lookup)",
                "(allow signal (target self))",
                f"(allow file-write* {' '.join(write_rules)})",
                "(allow network*)" if network else "",
            ]
        )
        return (
            [sandbox_exec, "-p", profile, "--", *sandboxed_command],
            cwd,
            root,
            {},
            "sandbox-exec",
            "",
        )

    if sys.platform.startswith("linux"):
        bwrap = shutil.which("bwrap")
        if not bwrap:
            return argv, cwd, root, {}, "", "bubblewrap is unavailable"
        system_paths = [
            Path(path)
            for path in (
                "/usr",
                "/bin",
                "/sbin",
                "/lib",
                "/lib64",
                "/etc/ld.so.cache",
                "/etc/ssl",
                "/etc/pki",
                "/etc/ca-certificates",
                "/etc/localtime",
                "/etc/passwd",
                "/etc/group",
                "/etc/nsswitch.conf",
                "/etc/hosts",
                "/etc/resolv.conf",
                "/etc/services",
                "/etc/protocols",
            )
        ]
        readable = [
            *system_paths,
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            *([executable] if executable is not None else []),
        ]
        mounts: list[str] = []
        selected: list[Path] = []
        for path in dict.fromkeys(readable):
            if (
                path == Path("/")
                or not path.exists()
                or any(path == parent or parent in path.parents for parent in selected)
            ):
                continue
            selected.append(path)
            mounts.extend(["--ro-bind", str(path), str(path)])
        isolation = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--cap-drop",
            "ALL",
            *([] if network else ["--unshare-net"]),
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            *mounts,
            "--bind" if writable else "--ro-bind",
            str(root),
            str(root),
            "--bind",
            str(qa_home),
            str(qa_home),
            "--chdir",
            str(cwd),
            "--",
            *sandboxed_command,
        ]
        return isolation, cwd, root, {}, "bwrap", ""

    return argv, cwd, root, {}, "", f"unsupported platform: {sys.platform}"


def _sandbox_startup_failed(
    kind: str, process: subprocess.CompletedProcess[str]
) -> bool:
    stderr = (process.stderr or "").lower()
    return (kind == "sandbox-exec" and "sandbox_apply" in stderr) or (
        kind == "bwrap"
        and any(
            marker in stderr
            for marker in (
                "creating new namespace failed",
                "operation not permitted",
                "no permissions to create a new namespace",
            )
        )
    )


def _verification_changes(before: dict[str, str], root: Path) -> list[str]:
    try:
        return changed_paths(before, snapshot_files(root, include_excluded=True))
    except (OSError, WorkspaceSafetyError) as exc:
        return [f"<execution tree unavailable: {exc}>"]


def run_verification_command(
    build_dir: str, command: VerificationCommand
) -> CommandResult:
    """Run one approved argv in an OS filesystem sandbox without app secrets."""
    relative_cwd = _safe_relative_path(command.cwd, allow_dot=True)
    source_root = Path(build_dir).resolve()
    source_cwd = (
        source_root / ("" if relative_cwd == "." else relative_cwd or "")
    ).resolve()
    if (
        relative_cwd is None
        or not source_cwd.is_relative_to(source_root)
        or not source_cwd.is_dir()
    ):
        return CommandResult(
            command_id=command.id,
            passed=False,
            returncode=2,
            stderr=f"Invalid verification working directory: {command.cwd!r}",
        )
    if not command.argv or any(not part.strip() for part in command.argv):
        return CommandResult(
            command_id=command.id,
            passed=False,
            returncode=2,
            stderr="Verification argv is empty.",
        )
    environment = {
        key: value for key, value in os.environ.items() if key in _SAFE_ENV_KEYS
    }
    try:
        with (
            tempfile.TemporaryDirectory(prefix="codebuilder-run-") as run_home,
            tempfile.TemporaryDirectory(prefix="codebuilder-qa-") as qa_home,
        ):
            qa_root = Path(qa_home).resolve()
            root = copy_clean_tree(source_root, Path(run_home) / "project").resolve()
            cwd = root / ("" if relative_cwd == "." else relative_cwd)
            for directory in ("cache", "uv-cache", "cargo", "npm-cache", "tmp"):
                (qa_root / directory).mkdir()
            (
                sandboxed_argv,
                sandboxed_cwd,
                execution_root,
                process_options,
                sandbox_kind,
                sandbox_error,
            ) = _verification_sandbox(
                root,
                cwd,
                qa_root,
                command.argv,
                network=command.network,
                writable=command.category == "build",
            )
            if sandbox_error:
                return CommandResult(
                    command_id=command.id,
                    passed=False,
                    returncode=127,
                    stderr=f"Verification sandbox unavailable: {sandbox_error}",
                )
            before = snapshot_files(execution_root, include_excluded=True)
            environment.update(
                {
                    "HOME": str(qa_root),
                    "XDG_CACHE_HOME": str(qa_root / "cache"),
                    "UV_CACHE_DIR": str(qa_root / "uv-cache"),
                    "CARGO_HOME": str(qa_root / "cargo"),
                    "npm_config_cache": str(qa_root / "npm-cache"),
                    "TEMP": str(qa_root / "tmp"),
                    "TMP": str(qa_root / "tmp"),
                    "TMPDIR": str(qa_root / "tmp"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(execution_root),
                }
            )
            timed_out = False
            process = subprocess.Popen(
                sandboxed_argv,
                cwd=sandboxed_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                shell=False,
                start_new_session=True,
                **process_options,
            )
            stdout, stderr, timed_out = _bounded_communicate(
                process, command.timeout_seconds
            )
            completed = subprocess.CompletedProcess(
                sandboxed_argv,
                process.returncode,
                stdout,
                stderr,
            )
            mutations = _verification_changes(before, execution_root)
            if timed_out:
                return CommandResult(
                    command_id=command.id,
                    passed=False,
                    returncode=124,
                    stdout=truncate(stdout or ""),
                    stderr=truncate(stderr or ""),
                    timed_out=True,
                    mutated_paths=mutations,
                )
            if _sandbox_startup_failed(sandbox_kind, completed):
                return CommandResult(
                    command_id=command.id,
                    passed=False,
                    returncode=127,
                    stdout=truncate(stdout or ""),
                    stderr="Verification sandbox unavailable: "
                    + truncate(stderr or "sandbox startup failed"),
                    mutated_paths=mutations,
                )
    except (OSError, WorkspaceSafetyError) as exc:
        return CommandResult(
            command_id=command.id,
            passed=False,
            returncode=127,
            stderr=str(exc),
        )
    stderr = stderr or ""
    denial_output = f"{stdout}\n{stderr}".lower()
    policy_denied = (
        completed.returncode != 0
        and sandbox_kind in {"sandbox-exec", "bwrap"}
        and any(
            marker in denial_output
            for marker in (
                "operation not permitted",
                "permission denied",
                "read-only file system",
            )
        )
    )
    if policy_denied:
        stderr = "Verification sandbox denied an unapproved capability.\n" + stderr
    if mutations:
        stderr = "\n".join(
            part
            for part in (
                stderr,
                "Verification command mutated its execution tree: "
                + ", ".join(mutations),
            )
            if part
        )
    mutation_failure = command.category != "build" and bool(mutations)
    return CommandResult(
        command_id=command.id,
        passed=completed.returncode == 0 and not mutation_failure,
        returncode=(
            126
            if completed.returncode == 0 and mutation_failure
            else 125
            if policy_denied
            else completed.returncode
        ),
        stdout=truncate(stdout or ""),
        stderr=truncate(stderr),
        mutated_paths=mutations,
    )


def run_verification_commands(
    build_dir: str, commands: list[VerificationCommand]
) -> list[CommandResult]:
    return [run_verification_command(build_dir, command) for command in commands]


def _module_exists(root: Path, module: str) -> bool:
    relative = Path(*module.split("."))
    for base in (root, root / "src"):
        if (base / relative).with_suffix(".py").is_file():
            return True
        if (base / relative / "__init__.py").is_file():
            return True
    return False


def _defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            names.update(
                alias.asname or alias.name for alias in node.names if alias.name != "*"
            )
    return names


def check_declared_tests(build_dir: str, package: WorkPackageSpec) -> str:
    """Validate the test author's exact approved files before code generation."""
    root = Path(build_dir).resolve()
    failures: list[str] = []
    for test in package.tests:
        relative = _safe_relative_path(test.path)
        path = root / (relative or test.path)
        try:
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError):
            resolved = Path("/")
        if (
            relative is None
            or not resolved.is_relative_to(root)
            or path.is_symlink()
            or any(
                parent.is_symlink()
                for parent in path.parents
                if parent != root and parent.is_relative_to(root)
            )
            or not path.is_file()
        ):
            failures.append(f"Missing declared test: {test.path}")
            continue
        if path.suffix != ".py":
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                failures.append(f"{test.path}: cannot inspect declared test: {exc}")
                continue
            if test.test_name not in content:
                failures.append(f"{test.path}: exact test missing: {test.test_name}.")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            failures.append(f"{test.path}: cannot inspect declared test: {exc}")
            continue
        if test.test_name not in _defined_names(tree):
            failures.append(f"{test.path}: exact test missing: {test.test_name}.")
    return "PASS" if not failures else "\n".join(failures)


def check_spec_contract(build_dir: str, plan: Plan) -> str:
    """Check exact paths, package/module names, exports, and protected assets."""
    try:
        validate_plan(plan)
    except ValueError as exc:
        return str(exc)
    if not plan.is_structured:
        return "SKIP: legacy Markdown plan"

    root = Path(build_dir).resolve()
    failures: list[str] = []
    all_identifiers: set[str] = set()
    defined_symbols: set[str] = set()
    for source_path in _production_python_files(root):
        try:
            source_tree = ast.parse(
                source_path.read_text(encoding="utf-8"), filename=str(source_path)
            )
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in ast.walk(source_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined_symbols.add(node.name)
                all_identifiers.add(node.name)
            elif isinstance(node, ast.Name):
                all_identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                all_identifiers.add(node.attr)
            elif isinstance(node, ast.arg):
                all_identifiers.add(node.arg)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _ID.fullmatch(node.value)
            ):
                all_identifiers.add(node.value)
    python_project = (
        any("python" in stack.casefold() for stack in plan.tech_stack)
        or bool(plan.identifier_contract.modules)
        or any(
            file.path.endswith(".py")
            for package in plan.work_packages
            for file in package.files
        )
    )
    package_names = [*plan.identifier_contract.packages]
    if python_project:
        package_names.insert(0, plan.package_name)
    for package_name in dict.fromkeys(package_names):
        package_locations = (
            root / package_name,
            root / "src" / package_name,
        )
        if not any(
            path.is_dir()
            and not any(
                parent.is_symlink()
                for parent in [path, *path.parents]
                if parent != root and parent.is_relative_to(root)
            )
            for path in package_locations
        ):
            failures.append(f"Package drift: expected exact package {package_name!r}.")
    if python_project:
        for module in plan.identifier_contract.modules:
            if not _module_exists(root, module):
                failures.append(f"Module drift: expected exact module {module!r}.")

    for package in plan.work_packages:
        for file in package.files:
            path = root / file.path
            if not path.resolve().is_relative_to(root.resolve()):
                failures.append(f"Declared file escapes project root: {file.path}")
                continue
            if path.is_symlink() or not path.is_file():
                failures.append(f"Missing declared file: {file.path}")
                continue
            if not file.public_api or path.suffix != ".py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeError) as exc:
                failures.append(f"{file.path}: cannot inspect public API: {exc}")
                continue
            defined = _defined_names(tree)
            expected = {
                name
                for declaration in file.public_api
                if (name := _public_api_name(declaration)) is not None
            }
            missing = sorted(expected - defined)
            if missing:
                failures.append(
                    f"{file.path}: exact public API missing: {', '.join(missing)}."
                )
            functions = {
                node.name: node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for declaration in file.public_api:
                expected_signature = _declared_signature(declaration)
                name = _public_api_name(declaration)
                if expected_signature is None or name not in functions:
                    continue
                actual_signature = _function_signature(functions[name])
                if actual_signature != expected_signature:
                    failures.append(
                        f"{file.path}: signature drift for {name}: expected "
                        f"{expected_signature!r}, found {actual_signature!r}."
                    )
        declared_tests = check_declared_tests(str(root), package)
        if not is_pass(declared_tests):
            failures.extend(declared_tests.splitlines())

    for symbol in plan.identifier_contract.symbols:
        if symbol not in defined_symbols:
            failures.append(f"Symbol drift: expected exact symbol {symbol!r}.")
    for field in plan.identifier_contract.fields:
        if field not in all_identifiers:
            failures.append(f"Field drift: expected exact field {field!r}.")

    env_path = root / ".env.example"
    env_names: set[str] = set()
    if env_path.is_file():
        try:
            env_names = {
                line.split("=", 1)[0].strip()
                for line in env_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#") and "=" in line
            }
        except (OSError, UnicodeError):
            pass
    for variable in plan.identifier_contract.environment_variables:
        if variable not in env_names:
            failures.append(
                f"Environment variable drift: {variable!r} is missing from .env.example."
            )

    if plan.identifier_contract.entry_points:
        pyproject, error = _load_pyproject(root)
        scripts = ((pyproject or {}).get("project") or {}).get("scripts") or {}
        actual_entry_points = {
            str(target) for target in scripts.values() if isinstance(target, str)
        }
        actual_entry_points.update(
            f"{name}={target}"
            for name, target in scripts.items()
            if isinstance(name, str) and isinstance(target, str)
        )
        for entry_point in plan.identifier_contract.entry_points:
            if entry_point not in actual_entry_points:
                failures.append(
                    f"Entry-point drift: expected exact entry point {entry_point!r}."
                    + (f" ({error})" if error else "")
                )

    for asset in plan.authoritative_assets:
        if not asset.immutable or not asset.sha256:
            continue
        path = root / asset.path
        if not path.resolve().is_relative_to(root.resolve()):
            failures.append(f"Authoritative asset escapes project root: {asset.path}")
            continue
        if path.is_symlink() or not path.is_file():
            failures.append(f"Missing authoritative asset: {asset.path}")
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            failures.append(f"{asset.path}: cannot verify authoritative asset: {exc}")
            continue
        if digest != asset.sha256.lower():
            failures.append(f"Authoritative asset changed: {asset.path}")

    return "PASS" if not failures else "\n".join(failures)


def qa_report_for_prompt(report: QAReport | None) -> str:
    """Bound each category independently before passing preflight evidence to an agent."""
    if report is None:
        return "(not run: no attached project was resolved)"
    sections = [
        f"Status: {'PASS' if report.passed else 'FAIL'}",
        f"### Lint and format\n{truncate(report.lint_output or '(no output)', MAX_PROMPT_SECTION_CHARS)}",
        f"### MyPy\n{truncate(report.type_output or '(no output)', MAX_PROMPT_SECTION_CHARS)}",
        f"### Install, configuration, dependencies, and entry points\n"
        f"{truncate(report.integration_notes or '(no output)', MAX_PROMPT_SECTION_CHARS)}",
        f"### Tests\n{truncate(report.test_output or '(no output)', MAX_PROMPT_SECTION_CHARS)}",
    ]
    return "\n\n".join(sections)


def qa_report_for_repair(report: QAReport) -> str:
    """Compact JSON view of a failed QA report for a repair pass."""
    payload = report.model_dump()
    payload["artifact_urls"] = []
    for key in ("lint_output", "type_output", "test_output", "integration_notes"):
        payload[key] = truncate(payload.get(key) or "")
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _literal_string(node: ast.AST | None) -> str | None:
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _settings_fields(tree: ast.Module) -> list[str]:
    """Return environment names declared by Pydantic BaseSettings classes."""
    found: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            base.id if isinstance(base, ast.Name) else base.attr
            for base in node.bases
            if isinstance(base, (ast.Name, ast.Attribute))
        }
        if "BaseSettings" not in bases:
            continue

        prefix = ""
        aliases: dict[str, str] = {}
        field_names: list[str] = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                targets = [
                    target.id for target in item.targets if isinstance(target, ast.Name)
                ]
                if "model_config" in targets and isinstance(item.value, ast.Call):
                    for keyword in item.value.keywords:
                        if keyword.arg == "env_prefix":
                            prefix = _literal_string(keyword.value) or prefix
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                name = item.target.id
                if name.startswith("_") or name == "model_config":
                    continue
                if isinstance(item.annotation, ast.Subscript):
                    annotation_name = getattr(item.annotation.value, "id", "")
                    if annotation_name == "ClassVar":
                        continue
                field_names.append(name)
                if isinstance(item.value, ast.Call):
                    for keyword in item.value.keywords:
                        if keyword.arg in {"alias", "validation_alias"}:
                            alias = _literal_string(keyword.value)
                            if alias:
                                aliases[name] = alias
            elif isinstance(item, ast.ClassDef) and item.name == "Config":
                for config_item in item.body:
                    if not isinstance(config_item, ast.Assign):
                        continue
                    if any(
                        isinstance(target, ast.Name) and target.id == "env_prefix"
                        for target in config_item.targets
                    ):
                        prefix = _literal_string(config_item.value) or prefix

        found.extend(
            (aliases.get(name) or f"{prefix}{name}").upper() for name in field_names
        )
    return found


def _base_settings_field_names(tree: ast.Module) -> set[str]:
    """Return Python attribute names declared on Pydantic BaseSettings classes."""
    fields: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            base.id if isinstance(base, ast.Name) else base.attr
            for base in node.bases
            if isinstance(base, (ast.Name, ast.Attribute))
        }
        if "BaseSettings" not in bases:
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(
                item.target, ast.Name
            ):
                continue
            name = item.target.id
            if name.startswith("_") or name == "model_config":
                continue
            if (
                isinstance(item.annotation, ast.Subscript)
                and getattr(item.annotation.value, "id", "") == "ClassVar"
            ):
                continue
            fields.add(name)
    return fields


def _is_any(annotation: ast.AST | None) -> bool:
    return (isinstance(annotation, ast.Name) and annotation.id == "Any") or (
        isinstance(annotation, ast.Attribute) and annotation.attr == "Any"
    )


def _self_attribute(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def check_rpa_production_contract(build_dir: str) -> str:
    """Catch type-bypassing wiring and unused client lifecycle methods in RPA source."""
    root = Path(build_dir)
    parsed: list[tuple[Path, ast.Module]] = []
    settings_fields: set[str] = set()
    for path in _production_python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        parsed.append((path, tree))
        settings_fields.update(_base_settings_field_names(tree))

    failures: list[str] = []
    lifecycle_classes: list[tuple[Path, str, tuple[str, str]]] = []
    calls_by_file: dict[Path, set[str]] = {}
    lifecycle_pairs = (("login", "logout"), ("connect", "disconnect"))

    for path, tree in parsed:
        relative = path.relative_to(root)
        calls_by_file[path] = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not (
                isinstance(node.func, ast.Name) and node.func.id == "getattr"
            ):
                continue
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
                continue
            attribute = node.args[1].value
            if not isinstance(attribute, str):
                continue
            receiver = _self_attribute(node.args[0])
            if receiver in {"settings", "_settings"} and settings_fields:
                if attribute not in settings_fields:
                    failures.append(
                        f"{relative}:{node.lineno}: settings field {attribute!r} "
                        "is not declared by BaseSettings."
                    )

        for class_node in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            method_names = {
                item.name
                for item in class_node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if any(token in class_node.name.lower() for token in ("client", "adapter")):
                for pair in lifecycle_pairs:
                    if set(pair).issubset(method_names):
                        lifecycle_classes.append((path, class_node.name, pair))

            init = next(
                (
                    item
                    for item in class_node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "__init__"
                ),
                None,
            )
            if init is None:
                continue
            any_parameters = {
                argument.arg
                for argument in [*init.args.posonlyargs, *init.args.args]
                if argument.arg != "self" and _is_any(argument.annotation)
            }
            injected: dict[str, str] = {}
            for item in ast.walk(init):
                if not isinstance(item, ast.Assign) or not isinstance(
                    item.value, ast.Name
                ):
                    continue
                if item.value.id not in any_parameters:
                    continue
                for target in item.targets:
                    target_name = _self_attribute(target)
                    if target_name:
                        injected[target_name] = item.value.id
            for item in ast.walk(class_node):
                if not isinstance(item, ast.Call) or not (
                    isinstance(item.func, ast.Name) and item.func.id == "getattr"
                ):
                    continue
                dependency = _self_attribute(item.args[0]) if item.args else None
                if dependency in injected:
                    failures.append(
                        f"{relative}:{item.lineno}: injected dependency "
                        f"{injected[dependency]!r} is typed Any and accessed dynamically; "
                        "use its real Protocol or concrete contract."
                    )

    for defining_path, class_name, (open_method, close_method) in lifecycle_classes:
        external_calls = set().union(
            *(calls for path, calls in calls_by_file.items() if path != defining_path)
        )
        missing = [
            method
            for method in (open_method, close_method)
            if method not in external_calls
        ]
        if missing:
            relative = defining_path.relative_to(root)
            failures.append(
                f"{relative}: {class_name} defines {open_method}()/{close_method}() "
                f"but production source never calls: {', '.join(missing)}."
            )

    return "PASS" if not failures else "\n".join(dict.fromkeys(failures))


def check_env_example(build_dir: str) -> str:
    """Compare Settings and README env snippets with the canonical example."""
    root = Path(build_dir)
    settings: list[str] = []
    parse_errors: list[str] = []
    for path in _python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_errors.append(f"{path.relative_to(root)}: {exc}")
            continue
        settings.extend(_settings_fields(tree))

    if not settings:
        return (
            "PASS"
            if not parse_errors
            else "AST scan warnings:\n" + "\n".join(parse_errors)
        )

    env_path = root / ".env.example"
    if not env_path.is_file():
        return "Missing .env.example for BaseSettings configuration."
    documented = {
        line.split("=", 1)[0].strip().upper()
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    missing = sorted(set(settings) - documented)
    readme_keys: set[str] = set()
    readme_path = root / "README.md"
    if readme_path.is_file():
        try:
            readme = readme_path.read_text(encoding="utf-8")
            readme_keys = {
                match.group(1)
                for block in _ENV_FENCE.findall(readme)
                for match in _ENV_ASSIGNMENT.finditer(block)
            }
        except (OSError, UnicodeError) as exc:
            parse_errors.append(f"README.md: {exc}")
    stale_readme = sorted(readme_keys - documented)
    if not missing and not stale_readme and not parse_errors:
        return "PASS"
    messages: list[str] = []
    if missing:
        messages.append(
            ".env.example is missing BaseSettings keys: " + ", ".join(missing)
        )
    if stale_readme:
        messages.append(
            "README.md documents environment keys not present in .env.example: "
            + ", ".join(stale_readme)
        )
    if parse_errors:
        messages.append("AST scan warnings:\n" + "\n".join(parse_errors))
    return "\n".join(messages)


def _runtime_dependency_values(pyproject: dict[str, Any]) -> list[str]:
    project = pyproject.get("project") or {}
    return [
        value for value in (project.get("dependencies") or []) if isinstance(value, str)
    ]


def _dependency_names(values: list[str]) -> set[str]:
    names: set[str] = set()
    for value in values:
        match = _DEPENDENCY_NAME.match(value.strip())
        if match:
            names.add(match.group(0).lower().replace("_", "-").replace(".", "-"))
    return names


def project_dependency_names(build_dir: str) -> list[str]:
    """Return declared dependency names across runtime, optional, and dev groups."""
    pyproject, _ = _load_pyproject(Path(build_dir))
    if pyproject is None:
        return []
    project = pyproject.get("project") or {}
    values = _runtime_dependency_values(pyproject)
    for groups in (
        project.get("optional-dependencies") or {},
        pyproject.get("dependency-groups") or {},
    ):
        if isinstance(groups, dict):
            for dependencies in groups.values():
                if isinstance(dependencies, list):
                    values.extend(
                        value for value in dependencies if isinstance(value, str)
                    )
    return sorted(_dependency_names(values))


def check_preserved_dependencies(build_dir: str, baseline: list[str]) -> str:
    missing = sorted(set(baseline) - set(project_dependency_names(build_dir)))
    return (
        "PASS"
        if not missing
        else "Existing dependencies removed during patch: " + ", ".join(missing)
    )


def _load_pyproject(root: Path) -> tuple[dict[str, Any] | None, str]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return None, "SKIP: no pyproject.toml"
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle), ""
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return None, f"Invalid pyproject.toml: {exc}"


def _source_signals(root: Path) -> tuple[bool, bool]:
    uses_pyodbc_url = False
    imports_win32com = False
    for path in _python_files(root):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        uses_pyodbc_url = uses_pyodbc_url or "mssql+pyodbc" in source.lower()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports_win32com = imports_win32com or any(
                    alias.name == "win32com" or alias.name.startswith("win32com.")
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports_win32com = imports_win32com or (
                    node.module == "win32com" or node.module.startswith("win32com.")
                )
    if not uses_pyodbc_url:
        for path in (root / ".env.example", root / "README.md"):
            if path.is_file():
                try:
                    uses_pyodbc_url = (
                        "mssql+pyodbc" in path.read_text(encoding="utf-8").lower()
                    )
                except (OSError, UnicodeError):
                    pass
    return uses_pyodbc_url, imports_win32com


def _check_entry_points(
    root: Path, pyproject: dict[str, Any], *, smoke: bool = False
) -> str:
    scripts = (pyproject.get("project") or {}).get("scripts") or {}
    if not scripts:
        return "SKIP: no [project.scripts] entries"
    if not isinstance(scripts, dict):
        return "Invalid [project.scripts]: expected a TOML table."
    code = (
        "import functools,importlib,sys; "
        "obj=functools.reduce(getattr, sys.argv[2].split('.'), "
        "importlib.import_module(sys.argv[1])); "
        "assert callable(obj), f'{sys.argv[1]}:{sys.argv[2]} is not callable'"
    )
    smoke_code = (
        "import functools,importlib,sys; "
        "name,module,target=sys.argv[1:4]; "
        "obj=functools.reduce(getattr,target.split('.'),importlib.import_module(module)); "
        "sys.argv=[name,'--help']; result=obj(); "
        "raise SystemExit(result if isinstance(result,int) else 0)"
    )
    failures: list[str] = []
    for name, target in scripts.items():
        if not isinstance(target, str) or ":" not in target:
            failures.append(f"{name}: invalid target {target!r}")
            continue
        module, callable_name = target.split(":", 1)
        callable_name = callable_name.split("[", 1)[0]
        try:
            process = subprocess.run(
                [project_python(str(root)), "-c", code, module, callable_name],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{name}: {exc}")
            continue
        if process.returncode != 0:
            output = ((process.stdout or "") + (process.stderr or "")).strip()
            failures.append(
                f"{name} ({target}): {output or f'exit {process.returncode}'}"
            )
            continue
        if smoke:
            try:
                with tempfile.TemporaryDirectory(
                    prefix="codebuilder-entrypoint-"
                ) as smoke_cwd:
                    env = dict(os.environ)
                    python_path = env.get("PYTHONPATH")
                    env["PYTHONPATH"] = str(root) + (
                        os.pathsep + python_path if python_path else ""
                    )
                    process = subprocess.run(
                        [
                            project_python(str(root)),
                            "-c",
                            smoke_code,
                            name,
                            module,
                            callable_name,
                        ],
                        cwd=smoke_cwd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=env,
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append(f"{name} --help: {exc}")
                continue
            if process.returncode != 0:
                output = ((process.stdout or "") + (process.stderr or "")).strip()
                failures.append(
                    f"{name} --help ({target}): "
                    f"{output or f'exit {process.returncode}'}"
                )
    return (
        "PASS"
        if not failures
        else "Console entry point failures:\n" + "\n".join(failures)
    )


def check_runtime_contract(build_dir: str, smoke_entry_points: bool = False) -> str:
    """Validate dependency declarations and installed console entry points."""
    root = Path(build_dir)
    pyproject, load_note = _load_pyproject(root)
    if pyproject is None:
        return load_note
    dependency_values = _runtime_dependency_values(pyproject)
    declared = _dependency_names(dependency_values)
    uses_pyodbc_url, imports_win32com = _source_signals(root)
    failures: list[str] = []
    if uses_pyodbc_url and "pyodbc" not in declared:
        failures.append("mssql+pyodbc is used but dependency `pyodbc` is not declared.")
    if imports_win32com:
        pywin32_specs = [
            value.lower()
            for value in dependency_values
            if "pywin32" in _dependency_names([value])
        ]
        if not pywin32_specs:
            failures.append(
                "win32com is imported but Windows-scoped dependency `pywin32` is not declared."
            )
        elif not any(
            ("sys_platform" in value and "==" in value and "win32" in value)
            or ("platform_system" in value and "==" in value and "windows" in value)
            for value in pywin32_specs
        ):
            failures.append(
                "`pywin32` must be scoped to Windows with a `sys_platform == 'win32'` marker."
            )
    entry_points = _check_entry_points(root, pyproject, smoke=smoke_entry_points)
    if not is_pass(entry_points) and not is_skip(entry_points):
        failures.append(entry_points)
    return "PASS" if not failures else "\n".join(failures)


def _run_with_example_env(build_dir: str) -> str:
    """Run tests with .env.example active, restoring the workspace afterward."""
    root = Path(build_dir)
    env_path = root / ".env"
    example_path = root / ".env.example"
    if env_path.exists():
        return "SKIP: existing .env was already active during the main pytest run"
    if not example_path.is_file():
        return "SKIP: no .env.example"
    output = ""
    cleanup_error = ""
    try:
        shutil.copyfile(example_path, env_path)
        output = TestRunnerTool(
            workspace_dir=build_dir,
            provision_environment=False,
        )._run(".")
    except OSError as exc:
        output = f"Could not activate .env.example for tests: {exc}"
    finally:
        try:
            env_path.unlink(missing_ok=True)
        except OSError as exc:
            cleanup_error = f"Could not remove temporary .env after tests: {exc}"
    return "\n".join(part for part in (output, cleanup_error) if part)


def run_final_qa(
    build_dir: str,
    *,
    artifact_urls: list[dict] | list[ArtifactRef] | None = None,
    require_installable: bool = False,
    require_typecheck: bool = False,
    locked_sync: bool = True,
    baseline_dependencies: list[str] | None = None,
    plan: Plan | None = None,
    package_id: str = "",
    spec_hash: str = "",
) -> QAReport:
    """Run every applicable deterministic check and aggregate all failures."""
    has_pyproject = (Path(build_dir) / "pyproject.toml").is_file()
    if has_pyproject and require_installable and not provisioning_enabled():
        sync_output = "SKIP: project environment provisioning is disabled"
    else:
        sync_output = ensure_project_env(build_dir, locked=locked_sync)
    sync_ok = not sync_output or not require_installable

    lint_output = LintRunnerTool(
        workspace_dir=build_dir,
        provision_environment=False,
    )._run(".")
    lint_ok = is_pass(lint_output)

    type_output = TypeCheckRunnerTool(
        workspace_dir=build_dir,
        native_config=True,
        provision_environment=False,
    )._run(".")
    type_ok = is_pass(type_output) or (is_skip(type_output) and not require_typecheck)

    env_output = check_env_example(build_dir)
    env_ok = is_pass(env_output)
    runtime_output = check_runtime_contract(build_dir, require_typecheck)
    runtime_ok = is_pass(runtime_output) or is_skip(runtime_output)
    dependency_output = check_preserved_dependencies(
        build_dir, baseline_dependencies or []
    )
    dependency_ok = is_pass(dependency_output)

    plan_valid = False
    if plan:
        try:
            validate_plan(plan)
            plan_valid = True
        except ValueError:
            pass
    contract_output = check_spec_contract(build_dir, plan) if plan else "SKIP: no spec"
    contract_ok = is_pass(contract_output) or is_skip(contract_output)
    command_results = (
        run_verification_commands(build_dir, plan.verification_commands)
        if plan and plan.is_structured and plan_valid
        else []
    )
    required_commands = {
        command.id: command.required
        for command in (plan.verification_commands if plan else [])
    }
    commands_ok = all(
        result.passed or not required_commands.get(result.command_id, True)
        for result in command_results
    )

    wiring_output = (
        check_rpa_production_contract(build_dir)
        if require_typecheck
        else "SKIP: not RPA"
    )
    wiring_ok = is_pass(wiring_output) or is_skip(wiring_output)

    test_output = TestRunnerTool(
        workspace_dir=build_dir,
        provision_environment=False,
    )._run(".")
    test_ok = is_pass(test_output)
    example_env_test_output = (
        _run_with_example_env(build_dir) if require_typecheck else "SKIP: not RPA"
    )
    example_env_test_ok = is_pass(example_env_test_output) or is_skip(
        example_env_test_output
    )
    combined_test_output = test_output
    if require_typecheck:
        combined_test_output = (
            f"{test_output}\n\n.env.example environment:\n{example_env_test_output}"
        )

    checks = [
        f"uv sync --locked: {'PASS' if not sync_output else 'FAIL'}",
        f"ruff check + format: {'PASS' if lint_ok else 'FAIL'}",
        f"mypy: {'PASS' if type_ok else 'FAIL'}",
        f".env.example consistency: {'PASS' if env_ok else 'FAIL'}",
        f"runtime dependencies and entry points: {'PASS' if runtime_ok else 'FAIL'}",
        f"existing dependencies preserved: {'PASS' if dependency_ok else 'FAIL'}",
        f"approved spec contract: {'PASS' if contract_ok else 'FAIL'}",
        f"approved verification commands: {'PASS' if commands_ok else 'FAIL'}",
        f"RPA production wiring: {'PASS' if wiring_ok else 'FAIL'}",
        f"pytest: {'PASS' if test_ok else 'FAIL'}",
        f"pytest with .env.example: {'PASS' if example_env_test_ok else 'FAIL'}",
    ]
    details: list[str] = []
    if sync_output:
        details.append(f"uv sync --locked output:\n{truncate(sync_output)}")
    if is_skip(type_output) and not require_typecheck:
        details.append(f"MyPy was not applicable: {type_output}")
    if not env_ok:
        details.append(f"Configuration contract:\n{env_output}")
    if not runtime_ok:
        details.append(f"Runtime contract:\n{runtime_output}")
    if not dependency_ok:
        details.append(f"Dependency preservation:\n{dependency_output}")
    if not contract_ok:
        details.append(f"Approved spec contract:\n{contract_output}")
    for result in command_results:
        if result.passed:
            continue
        details.append(
            f"Approved command {result.command_id} (exit {result.returncode}):\n"
            + truncate(
                "\n".join(part for part in (result.stdout, result.stderr) if part)
            )
        )
    if not wiring_ok:
        details.append(f"RPA production wiring:\n{wiring_output}")
    if not example_env_test_ok:
        details.append(
            "Pytest with .env.example:\n" + truncate(example_env_test_output)
        )
    notes = "\n".join([*checks, *details])

    contract_issues = (
        []
        if contract_ok
        else [
            QAIssue(
                source="contract",
                owner="spec" if contract_output.startswith("Invalid plan:") else "code",
                message="Approved specification contract failed.",
                evidence=truncate(contract_output),
                repair_instruction="Restore the exact approved paths and identifiers.",
            )
        ]
    )
    command_issues = [
        QAIssue(
            source="command",
            owner=(
                "spec"
                if result.returncode in {125, 126}
                else "environment"
                if result.returncode in {124, 127}
                else "code"
            ),
            message=f"Verification command {result.command_id!r} failed.",
            evidence=truncate(
                "\n".join(part for part in (result.stdout, result.stderr) if part)
            ),
            repair_instruction="Fix the implementation without changing approved tests.",
            blocking=required_commands.get(result.command_id, True),
        )
        for result in command_results
        if not result.passed
    ]
    return QAReport(
        passed=(
            sync_ok
            and lint_ok
            and type_ok
            and env_ok
            and runtime_ok
            and dependency_ok
            and contract_ok
            and commands_ok
            and wiring_ok
            and test_ok
            and example_env_test_ok
        ),
        lint_output=truncate(lint_output),
        type_output=truncate(type_output),
        test_output=truncate(combined_test_output),
        integration_notes=truncate(notes),
        artifact_urls=artifact_refs(artifact_urls),
        spec_hash=spec_hash
        or (plan_spec_hash(plan) if plan and plan.is_structured else ""),
        package_id=package_id,
        command_results=command_results,
        issues=[*contract_issues, *command_issues],
        contract_issues=contract_issues,
    )
