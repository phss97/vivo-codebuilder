import base64
import zipfile
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from codebuilder.history import canonicalize_git_url

from . import git_tool
from .workspace_tool import _is_skipped_path, resolve_within


def _safe_name(name: str) -> str:
    return Path(name).name or "attachment"


def _sanitized_git_origin(url: str) -> str:
    """Reduce a clone URL to ``host/org/repo`` with NO credentials.

    Strips the fragment and query string (signed/`?token=` tokens) first, then
    reuses ``canonicalize_git_url`` to drop the scheme, userinfo (``user:pat@``),
    and ``.git`` suffix. This is what's safe to put in a record summary that
    later reaches the planner LLM prompt.
    """
    base = url.split("#", 1)[0].split("?", 1)[0]
    return canonicalize_git_url(base)


def _extract_zip_safely(data: bytes, extract_to: Path) -> None:
    with zipfile.ZipFile(BytesIO(data)) as zf:
        for info in zf.infolist():
            try:
                target = resolve_within(str(extract_to), info.filename)
            except ValueError as exc:
                raise ValueError(f"Unsafe zip member {info.filename!r}: {exc}") from exc
            if _is_skipped_path(Path(info.filename)):
                continue
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as dest:
                dest.write(source.read())


def materialize(attachments: list[dict], workspace_dir: str) -> list[dict]:
    """Decode/ingest every Attachment payload into the job workspace.

    Returns a list of {name, kind, path, summary} records for downstream agents.
    Zip archives are extracted. Git URLs are cloned. PDFs are preserved and also
    summarized to text. Images are saved raw so a multimodal agent can read them.
    """
    inputs_dir = Path(workspace_dir) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for att in attachments:
        kind = att.get("kind")
        name = _safe_name(att.get("name") or f"attachment_{len(records)}")
        if kind == "git":
            url = att.get("uri") or ""
            if not url:
                continue
            dest = inputs_dir / "repo"
            if dest.exists():
                dest_name = f"repo_{len(records)}"
                dest = inputs_dir / dest_name
            git_tool.clone(url, str(dest))
            origin = _sanitized_git_origin(url) or "(private source)"
            records.append({"kind": "git", "name": name, "path": str(dest.relative_to(workspace_dir)), "summary": f"git repo cloned from {origin}"})
        elif kind == "zip":
            data = base64.b64decode(att.get("content_b64", ""))
            extract_to = inputs_dir / (Path(name).stem or "archive")
            extract_to.mkdir(parents=True, exist_ok=True)
            _extract_zip_safely(data, extract_to)
            records.append({"kind": "zip", "name": name, "path": str(extract_to.relative_to(workspace_dir)), "summary": f"extracted {name}"})
        elif kind == "pdf":
            data = base64.b64decode(att.get("content_b64", ""))
            pdf_path = inputs_dir / name
            pdf_path.write_bytes(data)
            text = ""
            try:
                reader = PdfReader(BytesIO(data))
                text = "\n".join((p.extract_text() or "") for p in reader.pages)
            except Exception as exc:
                text = f"(failed to extract PDF text: {exc})"
            txt_path = pdf_path.with_suffix(".txt")
            txt_path.write_text(text, encoding="utf-8")
            records.append({"kind": "pdf", "name": name, "path": str(pdf_path.relative_to(workspace_dir)), "summary": f"PDF with {len(text)} chars of text"})
        elif kind == "image":
            data = base64.b64decode(att.get("content_b64", ""))
            img_path = inputs_dir / name
            img_path.write_bytes(data)
            records.append({"kind": "image", "name": name, "path": str(img_path.relative_to(workspace_dir)), "summary": f"image ({len(data)} bytes)"})
        else:
            continue

    return records
