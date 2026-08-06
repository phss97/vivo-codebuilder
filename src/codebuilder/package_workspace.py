"""Safe staging and promotion helpers for spec-driven package execution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from codebuilder.tools.s3_artifacts import SKIP_DIRS, SKIP_FILES


_EXCLUDED_DIRS = frozenset(
    SKIP_DIRS
    | {
        ".artifacts",
        ".aws",
        ".secrets",
        ".ssh",
        ".uv-cache",
        "artifacts",
        "build",
        "dist",
        "htmlcov",
    }
)
_EXCLUDED_FILES = frozenset(
    SKIP_FILES
    | {
        ".coverage",
        ".envrc",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "service-account.json",
        "service_account.json",
    }
)
_SECRET_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_ARCHIVE_SUFFIXES = frozenset({".gz", ".tar", ".tgz", ".zip"})
_PUBLIC_ENV_SUFFIXES = (".example", ".sample", ".template")


class WorkspaceSafetyError(ValueError):
    """A filesystem operation escaped or weakened the workspace boundary."""


def _workspace_root(workspace_root: str | Path) -> Path:
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise WorkspaceSafetyError(f"workspace root is not a directory: {root}")
    return root


def _inside_workspace(
    root: Path, path: str | Path, *, allow_root: bool = False
) -> Path:
    raw = Path(path)
    absolute = root / raw if not raw.is_absolute() else Path(os.path.abspath(raw))
    # macOS exposes these system roots through /private. Normalize only those
    # fixed aliases; all symlinks at or below the job root remain forbidden.
    for alias in (Path("/tmp"), Path("/var")):
        if not alias.is_symlink():
            continue
        target = alias.resolve()
        if target != root and target not in root.parents:
            continue
        try:
            absolute = target / absolute.relative_to(alias)
        except ValueError:
            continue
        break
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise WorkspaceSafetyError(f"path escapes workspace: {path}") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise WorkspaceSafetyError(f"symlink target is not allowed: {path}")
    candidate = absolute.resolve(strict=False)
    if root != candidate and root not in candidate.parents:
        raise WorkspaceSafetyError(f"path escapes workspace: {path}")
    if candidate == root and not allow_root:
        raise WorkspaceSafetyError("operation cannot target the workspace root")
    return candidate


def _tree_root(root: Path, path: str | Path) -> Path:
    tree = _inside_workspace(root, path)
    if not tree.is_dir():
        raise WorkspaceSafetyError(f"tree root is not a directory: {tree}")
    return tree


def _independent(first: Path, second: Path) -> None:
    if first == second or first in second.parents or second in first.parents:
        raise WorkspaceSafetyError(f"paths must be separate: {first} and {second}")


def _normalize_relative(path: str | Path) -> str:
    value = str(path).strip().replace("\\", "/")
    pure = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or value.startswith("/")
        or (len(value) > 1 and value[1] == ":")
        or ".." in pure.parts
        or pure.as_posix() == "."
    ):
        raise WorkspaceSafetyError(f"unsafe relative path: {path!r}")
    normalized = pure.as_posix()
    if _is_excluded(PurePosixPath(normalized)):
        raise WorkspaceSafetyError(f"excluded or sensitive path: {normalized}")
    return normalized


def _normalize_paths(paths: Iterable[str | Path]) -> list[str]:
    return sorted({_normalize_relative(path) for path in paths})


def _is_excluded(relative: PurePosixPath) -> bool:
    if any(part in _EXCLUDED_DIRS for part in relative.parts):
        return True
    name = relative.name
    if name in _EXCLUDED_FILES or relative.suffix.lower() in _SECRET_SUFFIXES:
        return True
    if name == ".env":
        return True
    if name.startswith(".env.") and not name.endswith(_PUBLIC_ENV_SUFFIXES):
        return True
    return relative.suffix.lower() in _ARCHIVE_SUFFIXES


def _safe_member(root: Path, relative: str) -> Path:
    member = root
    for part in PurePosixPath(relative).parts:
        member /= part
        if member.is_symlink():
            raise WorkspaceSafetyError(f"symlink is not allowed: {relative}")
    resolved = member.resolve()
    if root != resolved and root not in resolved.parents:
        raise WorkspaceSafetyError(f"path escapes tree: {relative}")
    return member


def _restore_member(root: Path, relative: str) -> Path:
    """Resolve a target lexically, replacing untrusted path-type drift safely."""
    parts = PurePosixPath(relative).parts
    current = root
    for part in parts[:-1]:
        current /= part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            if current.is_dir() and not current.is_symlink():
                shutil.rmtree(current)
            else:
                current.unlink()
        current.mkdir(exist_ok=True)
    return current / parts[-1]


def _iter_files(
    root: Path, *, reject_special: bool = True, include_excluded: bool = False
):
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        relative_dir = current.relative_to(root)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not (current / name).is_symlink()
            and (
                include_excluded
                or not _is_excluded(PurePosixPath(relative_dir.as_posix(), name))
            )
        )
        for name in sorted(filenames):
            path = current / name
            relative = path.relative_to(root)
            if path.is_symlink() or (
                not include_excluded
                and _is_excluded(PurePosixPath(relative.as_posix()))
            ):
                continue
            if not stat.S_ISREG(path.lstat().st_mode) and reject_special:
                raise WorkspaceSafetyError(
                    f"special filesystem node is not allowed: {relative.as_posix()}"
                )
            yield path, relative.as_posix()


def _iter_symlinks(root: Path, *, include_excluded: bool = False):
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        relative_dir = current.relative_to(root)
        for name in sorted([*dirnames, *filenames]):
            path = current / name
            relative = PurePosixPath(relative_dir.as_posix(), name)
            if path.is_symlink() and (include_excluded or not _is_excluded(relative)):
                yield path, relative.as_posix()
        dirnames[:] = [
            name
            for name in dirnames
            if not (current / name).is_symlink()
            and (
                include_excluded
                or not _is_excluded(PurePosixPath(relative_dir.as_posix(), name))
            )
        ]


def _copy_ignore(source: Path):
    def ignore(directory: str, names: list[str]) -> set[str]:
        current = Path(directory)
        ignored: set[str] = set()
        for name in names:
            path = current / name
            relative = PurePosixPath(path.relative_to(source).as_posix())
            if path.is_symlink() or _is_excluded(relative):
                ignored.add(name)
                continue
            mode = path.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise WorkspaceSafetyError(
                    f"special filesystem node is not allowed: {relative.as_posix()}"
                )
        return ignored

    return ignore


def stage_tree(
    workspace_root: str | Path,
    source: str | Path,
    stage: str | Path,
) -> Path:
    """Reset ``stage`` and copy a clean source tree into it."""
    root = _workspace_root(workspace_root)
    source_root = _tree_root(root, source)
    stage_root = _inside_workspace(root, stage)
    _independent(source_root, stage_root)

    if stage_root.is_symlink():
        raise WorkspaceSafetyError(f"stage cannot be a symlink: {stage_root}")
    if stage_root.exists():
        if not stage_root.is_dir():
            raise WorkspaceSafetyError(f"stage is not a directory: {stage_root}")
        shutil.rmtree(stage_root)
    for _path, _relative in _iter_files(source_root):
        pass
    stage_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_root,
        stage_root,
        copy_function=shutil.copy2,
        ignore=_copy_ignore(source_root),
    )
    return stage_root


def copy_clean_tree(source: str | Path, destination: str | Path) -> Path:
    """Copy regular, non-sensitive files into a new disposable directory."""
    source_path = Path(source)
    if source_path.is_symlink():
        raise WorkspaceSafetyError("clean-copy source cannot be a symlink")
    source_root = source_path.resolve()
    output = Path(destination)
    if not source_root.is_dir() or output.exists() or not output.parent.is_dir():
        raise WorkspaceSafetyError("clean-copy source/destination is invalid")
    output.mkdir()
    for path, relative in _iter_files(source_root):
        target = output.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    harden_tree(output)
    return output


def harden_tree(tree_root: str | Path) -> Path:
    """Strip group/other write bits without changing execute permissions."""
    raw = Path(tree_root)
    if raw.is_symlink():
        raise WorkspaceSafetyError("tree root cannot be a symlink")
    root = raw.resolve()
    if not root.is_dir():
        raise WorkspaceSafetyError(f"tree root is not a directory: {root}")

    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for path in (current, *(current / name for name in [*dirnames, *filenames])):
            if path.is_symlink():
                continue
            metadata = path.stat()
            if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                continue
            mode = stat.S_IMODE(metadata.st_mode)
            hardened = mode & ~0o022
            if hardened != mode:
                path.chmod(hardened)
    return root


def snapshot_files(
    tree_root: str | Path,
    paths: Iterable[str | Path] | None = None,
    *,
    include_excluded: bool = False,
) -> dict[str, str]:
    """Return deterministic ``relative path -> SHA256`` file hashes."""
    root = Path(tree_root).resolve()
    if not root.is_dir():
        raise WorkspaceSafetyError(f"tree root is not a directory: {root}")

    files: list[tuple[Path, str]] = []
    links: list[tuple[Path, str]] = []
    if paths is None:
        files.extend(
            _iter_files(
                root,
                reject_special=False,
                include_excluded=include_excluded,
            )
        )
        links.extend(_iter_symlinks(root, include_excluded=include_excluded))
    else:
        for relative in _normalize_paths(paths):
            path = root.joinpath(*PurePosixPath(relative).parts)
            if path.is_symlink():
                links.append((path, relative))
                continue
            parent = path.parent
            escaped_through_link = False
            while parent != root:
                if parent.is_symlink():
                    escaped_through_link = True
                    break
                parent = parent.parent
            if escaped_through_link:
                continue
            resolved = path.resolve()
            if root != resolved and root not in resolved.parents:
                continue
            if path.exists():
                files.append((path, relative))

    snapshot: dict[str, str] = {}
    for path, relative in sorted(files, key=lambda item: item[1]):
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            snapshot[relative] = (
                f"special:{stat.S_IFMT(metadata.st_mode):o}:"
                f"{stat.S_IMODE(metadata.st_mode):o}"
            )
            continue
        digest = hashlib.sha256()
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise WorkspaceSafetyError(f"cannot snapshot file: {relative}") from exc
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                snapshot[relative] = (
                    f"special:{stat.S_IFMT(metadata.st_mode):o}:"
                    f"{stat.S_IMODE(metadata.st_mode):o}"
                )
                continue
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        mode = stat.S_IMODE(metadata.st_mode)
        snapshot[relative] = f"file:{mode:o}:{digest.hexdigest()}"
    for path, relative in sorted(links, key=lambda item: item[1]):
        snapshot[relative] = f"symlink:{os.readlink(path)}"
    return snapshot


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Return sorted added, modified, or deleted paths."""
    return sorted(
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    )


def validate_changed_paths(
    changed: Iterable[str | Path], allowlist: Iterable[str | Path]
) -> list[str]:
    """Validate that every changed path is an exact allowlist member."""
    normalized_changed = _normalize_paths(changed)
    allowed = set(_normalize_paths(allowlist))
    unexpected = [path for path in normalized_changed if path not in allowed]
    if unexpected:
        raise WorkspaceSafetyError(
            "changed paths outside exact allowlist: " + ", ".join(unexpected)
        )
    return normalized_changed


def restore_files(
    workspace_root: str | Path,
    target_root: str | Path,
    trusted_root: str | Path,
    paths: Iterable[str | Path],
) -> list[str]:
    """Restore exact frozen files from a trusted tree, including deletions."""
    root = _workspace_root(workspace_root)
    target = _tree_root(root, target_root)
    trusted = _tree_root(root, trusted_root)
    _independent(target, trusted)
    normalized = _normalize_paths(paths)

    for relative in sorted(
        normalized, key=lambda value: value.count("/"), reverse=True
    ):
        source = _safe_member(trusted, relative)
        destination = _restore_member(target, relative)
        if source.is_dir():
            if destination.is_symlink() or (
                destination.exists() and not destination.is_dir()
            ):
                destination.unlink()
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if source.exists() and not source.is_file():
            raise WorkspaceSafetyError(f"trusted path is not restorable: {relative}")
        if destination.is_symlink():
            destination.unlink()
        elif destination.exists() and destination.is_dir():
            shutil.rmtree(destination)
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        elif destination.exists():
            destination.unlink()
    return normalized


def promote_files(
    workspace_root: str | Path,
    stage_root: str | Path,
    canonical_root: str | Path,
    allowlist: Iterable[str | Path],
) -> list[str]:
    """Promote a green stage only when all differences are allowlisted."""
    root = _workspace_root(workspace_root)
    stage = _tree_root(root, stage_root)
    canonical = _tree_root(root, canonical_root)
    _independent(stage, canonical)
    allowed = _normalize_paths(allowlist)
    changes = validate_changed_paths(
        changed_paths(snapshot_files(canonical), snapshot_files(stage)), allowed
    )

    for relative in changes:  # validate every source before mutating canonical
        source = _safe_member(stage, relative)
        destination = _safe_member(canonical, relative)
        if source.exists() and not source.is_file():
            raise WorkspaceSafetyError(f"stage path is not a file: {relative}")
        if destination.exists() and destination.is_dir():
            raise WorkspaceSafetyError(f"canonical path is a directory: {relative}")

    for relative in changes:
        source = _safe_member(stage, relative)
        destination = _safe_member(canonical, relative)
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            mode = stat.S_IMODE(destination.stat().st_mode)
            destination.chmod(mode & ~0o022)
        elif destination.exists():
            destination.unlink()
    return changes


def create_quarantine_zip(
    workspace_root: str | Path,
    canonical_root: str | Path,
    failed_stage_root: str | Path,
    approved_paths: Iterable[str | Path],
    spec_json: str,
    qa_markdown: str,
    destination: str | Path,
) -> Path:
    """Package last-green source and sanitized failed-stage evidence."""
    root = _workspace_root(workspace_root)
    canonical = _tree_root(root, canonical_root)
    failed = _tree_root(root, failed_stage_root)
    _independent(canonical, failed)
    output = _inside_workspace(root, destination)
    if output in canonical.parents or canonical in output.parents:
        raise WorkspaceSafetyError("quarantine archive cannot be inside canonical tree")
    if output in failed.parents or failed in output.parents:
        raise WorkspaceSafetyError("quarantine archive cannot be inside failed stage")
    if output.suffix.lower() != ".zip":
        raise WorkspaceSafetyError("quarantine destination must end in .zip")
    if output.exists() and not output.is_file():
        raise WorkspaceSafetyError(f"quarantine destination is not a file: {output}")

    approved = _normalize_paths(approved_paths)
    trusted_spec = json.dumps(json.loads(spec_json), indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    if temporary.exists():
        if temporary.is_dir():
            raise WorkspaceSafetyError(f"temporary archive is a directory: {temporary}")
        temporary.unlink()

    manifest: list[dict[str, str]] = []
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path, relative in _iter_files(canonical):
                archive.write(path, f"last-green/{relative}")
            for relative in approved:
                path = _safe_member(failed, relative)
                if path.exists() and not path.is_file():
                    raise WorkspaceSafetyError(
                        f"failed-stage path is not a file: {relative}"
                    )
                status = "included" if path.is_file() else "missing"
                manifest.append({"path": relative, "status": status})
                if path.is_file():
                    archive.write(path, f"failed-stage/{relative}")
            archive.writestr("evidence/approved-spec.json", trusted_spec)
            archive.writestr("evidence/QA.md", qa_markdown)
            archive.writestr(
                "evidence/failed-stage-files.json",
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output
