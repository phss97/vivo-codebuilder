"""Upload a finished workspace to S3 and return presigned GET URLs.

No-op when `CODEBUILDER_ARTIFACT_BUCKET` is unset — keeps local dev working
without AWS creds. Intentionally tolerant: any failure is logged and yields
an empty list, so `finalize()` never breaks on artifact upload.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# 7 days — matches README/deploy documentation.
PRESIGN_TTL_SECONDS = 7 * 24 * 3600

# Files we never upload (noise, potentially sensitive).
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
SKIP_FILES = {".DS_Store"}
_SKIP_DIRS = SKIP_DIRS
_SKIP_FILES = SKIP_FILES


def upload_file(local_path: str | Path, key: str) -> dict | None:
    """Upload a single file and return its presigned GET ref, or None if disabled/failed."""
    bucket = os.environ.get("CODEBUILDER_ARTIFACT_BUCKET")
    if not bucket:
        return None

    base = (os.environ.get("CODEBUILDER_ARTIFACT_PREFIX") or "").strip("/")
    if base:
        key = f"{base}/{key.lstrip('/')}"

    try:
        import boto3
    except ImportError:
        log.warning("boto3 not installed; skipping S3 upload")
        return None

    path = Path(local_path)
    if not path.is_file():
        return None

    region = os.environ.get("AWS_REGION", "us-east-1")
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.upload_file(str(path), bucket, key)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=PRESIGN_TTL_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 — observability, not correctness
        log.warning("s3 upload failed for %s: %s", key, exc)
        return None

    return {"file_path": path.name, "size": path.stat().st_size, "url": url}


def upload_workspace(workspace_dir: str | Path, prefix: str) -> list[dict]:
    bucket = os.environ.get("CODEBUILDER_ARTIFACT_BUCKET")
    if not bucket:
        return []

    # Optional base prefix (e.g. "pedro/vivo-codebuilder") scopes all writes
    # inside a folder of a shared bucket.
    base = (os.environ.get("CODEBUILDER_ARTIFACT_PREFIX") or "").strip("/")
    if base:
        prefix = f"{base}/{prefix.strip('/')}"

    try:
        import boto3
    except ImportError:
        log.warning("boto3 not installed; skipping S3 artifact upload")
        return []

    workspace = Path(workspace_dir)
    if not workspace.is_dir():
        return []

    region = os.environ.get("AWS_REGION", "us-east-1")
    s3 = boto3.client("s3", region_name=region)

    refs: list[dict] = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if path.name in _SKIP_FILES:
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(workspace).parts):
            continue

        rel = path.relative_to(workspace).as_posix()
        key = f"{prefix.strip('/')}/{rel}"
        try:
            s3.upload_file(str(path), bucket, key)
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=PRESIGN_TTL_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 — observability, not correctness
            log.warning("s3 upload failed for %s: %s", rel, exc)
            continue

        refs.append({"file_path": rel, "size": path.stat().st_size, "url": url})

    return refs
