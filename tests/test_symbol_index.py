"""Tests for the real-symbol index that grounds the writer in actual code.

The planner-declared ``public_api`` carries top-level names but not class
*fields*, so writers invent attributes (the `settings.sap_host` drift). The
index is extracted from the code already on disk, so a downstream writer sees
the real field names, constructor signature, and function signatures.
"""

import textwrap

from codebuilder.runtime_qa import build_symbol_index, extract_module_api

_SETTINGS_SRC = textwrap.dedent(
    '''\
    from pydantic_settings import BaseSettings


    class Settings(BaseSettings):
        database_url: str
        sap_endpoint: str
        sap_user: str
        sap_password_secret: str
        log_level: str = "INFO"


    def get_settings() -> Settings:
        return Settings()


    class SapClientImpl:
        def __init__(self, settings: Settings, secret_provider: object) -> None:
            self._settings = settings
    '''
)


def test_extract_includes_class_fields() -> None:
    api = extract_module_api(_SETTINGS_SRC)
    assert "class Settings" in api
    for field in ("database_url", "sap_endpoint", "sap_user", "sap_password_secret", "log_level"):
        assert field in api, f"missing field {field} in:\n{api}"


def test_extract_includes_functions_and_init_params() -> None:
    api = extract_module_api(_SETTINGS_SRC)
    assert "get_settings" in api
    assert "class SapClientImpl" in api
    # constructor params are the real wiring contract for the DI container
    assert "settings" in api and "secret_provider" in api


def test_extract_omits_invented_names() -> None:
    api = extract_module_api(_SETTINGS_SRC)
    # the exact drift bug: these were invented by the container, never defined
    assert "sap_host" not in api
    assert "max_retry_attempts" not in api


def test_extract_ignores_private_and_dunder() -> None:
    src = "def _helper():\n    pass\n\n\ndef public_fn():\n    pass\n"
    api = extract_module_api(src)
    assert "public_fn" in api
    assert "_helper" not in api


def test_build_symbol_index_maps_modules(tmp_path) -> None:
    pkg = tmp_path / "src" / "terra" / "config"
    pkg.mkdir(parents=True)
    (tmp_path / "src" / "terra" / "__init__.py").write_text("")
    (tmp_path / "src" / "terra" / "config" / "__init__.py").write_text("")
    (pkg / "settings.py").write_text(_SETTINGS_SRC)

    index = build_symbol_index(str(tmp_path))
    assert "terra.config.settings" in index
    assert "database_url" in index["terra.config.settings"]


def test_build_symbol_index_skips_unparseable(tmp_path) -> None:
    (tmp_path / "broken.py").write_text("def (:\n")  # syntax error
    # must not raise — drift prevention is best-effort
    assert build_symbol_index(str(tmp_path)) == {} or "broken" not in str(
        build_symbol_index(str(tmp_path))
    )
