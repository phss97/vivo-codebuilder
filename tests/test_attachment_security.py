"""Git clone URLs must never carry credentials into state or the planner prompt.

The materializer's git record summary previously embedded the raw clone URL, so
an HTTPS URL with a PAT/userinfo or a signed ?token= query leaked to the LLM.
"""

from pathlib import Path

import codebuilder.tools.attachment_tool as atool
from codebuilder.tools.attachment_tool import _sanitized_git_origin


def _no_clone(monkeypatch):
    monkeypatch.setattr(atool.git_tool, "clone", lambda url, dest: Path(dest).mkdir(parents=True, exist_ok=True))


def test_sanitized_origin_strips_userinfo_query_and_scp():
    assert _sanitized_git_origin("https://u:p@github.com/o/r.git") == "github.com/o/r"
    assert _sanitized_git_origin("git@github.com:o/r.git") == "github.com/o/r"
    assert "secret" not in _sanitized_git_origin("https://github.com/o/r?token=SECRET").lower()
    assert "secret" not in _sanitized_git_origin("https://github.com/o/r.git#SECRET").lower()


def test_materialize_git_summary_has_no_userinfo_token(tmp_path, monkeypatch):
    _no_clone(monkeypatch)
    url = "https://x-access-token:ghp_SUPERSECRET@github.com/org/repo.git"
    records = atool.materialize([{"kind": "git", "name": "repo", "uri": url}], str(tmp_path))
    blob = " ".join(str(v) for r in records for v in r.values()).lower()
    assert "supersecret" not in blob
    assert "x-access-token" not in blob
    assert "github.com/org/repo" in records[0]["summary"]


def test_materialize_git_summary_has_no_query_token(tmp_path, monkeypatch):
    _no_clone(monkeypatch)
    url = "https://github.com/org/repo.git?token=QUERYSECRET"
    records = atool.materialize([{"kind": "git", "name": "repo", "uri": url}], str(tmp_path))
    blob = " ".join(str(v) for r in records for v in r.values()).lower()
    assert "querysecret" not in blob
