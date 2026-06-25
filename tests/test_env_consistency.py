"""Tests for the .env.example <-> Settings consistency gate.

The Vivo-demoed symptom "config won't load even when filled in" comes from a
.env.example whose keys don't match the env vars pydantic-settings actually
reads (wrong names / missing the env_prefix). mypy doesn't read .env files and
tests use explicit fixture values, so nothing else catches this.
"""

import textwrap

from codebuilder.runtime_qa import check_env_example_consistency

_SETTINGS = textwrap.dedent(
    '''\
    from pydantic_settings import BaseSettings, SettingsConfigDict


    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="TERRA_")

        database_url: str
        sap_endpoint: str
        log_level: str = "INFO"
    '''
)


def _project(tmp_path, env_text: str) -> None:
    pkg = tmp_path / "src" / "app" / "config"
    pkg.mkdir(parents=True)
    (pkg / "settings.py").write_text(_SETTINGS)
    (tmp_path / ".env.example").write_text(env_text)


def test_flags_missing_prefixed_keys(tmp_path) -> None:
    _project(tmp_path, "DB_URL=\nSAP_ENVIRONMENT=\n")
    issues = check_env_example_consistency(str(tmp_path))
    joined = "\n".join(issues)
    assert "TERRA_DATABASE_URL" in joined
    assert "TERRA_SAP_ENDPOINT" in joined


def test_clean_when_keys_match(tmp_path) -> None:
    _project(tmp_path, "TERRA_DATABASE_URL=postgres://x\nTERRA_SAP_ENDPOINT=https://sap\n")
    assert check_env_example_consistency(str(tmp_path)) == []


def test_optional_field_not_required(tmp_path) -> None:
    # log_level has a default → optional → its absence must NOT be flagged.
    _project(tmp_path, "TERRA_DATABASE_URL=x\nTERRA_SAP_ENDPOINT=y\n")
    issues = check_env_example_consistency(str(tmp_path))
    assert not any("LOG_LEVEL" in i for i in issues)


def test_no_env_example_returns_empty(tmp_path) -> None:
    pkg = tmp_path / "src" / "app"
    pkg.mkdir(parents=True)
    (pkg / "settings.py").write_text(_SETTINGS)
    assert check_env_example_consistency(str(tmp_path)) == []


def test_no_settings_class_returns_empty(tmp_path) -> None:
    (tmp_path / ".env.example").write_text("FOO=bar\n")
    assert check_env_example_consistency(str(tmp_path)) == []
