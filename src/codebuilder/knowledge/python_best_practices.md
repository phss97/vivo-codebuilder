# Python Best Practices for Codebuilder

## Layout
- One responsibility per module. Avoid god-modules.
- Use `src/<package>/` layout for anything that might be installed.
- Tests live under `tests/`, mirroring package structure.

## Style
- Target Python 3.10+. Use modern typing: `list[int]`, `dict[str, Any]`, `X | None`.
- Prefer f-strings. No `%` formatting.
- Functions: small, pure where possible. Explicit args beat hidden globals.
- Type-hint public signatures.
- Avoid wildcard imports.

## Error handling
- Catch specific exceptions. Never `except:` or bare `except Exception` unless you re-raise.
- Validate only at boundaries (CLI args, HTTP input, file parsing). Trust internal calls.
- Fail loudly in setup; fail soft in user-facing loops.

## Dependencies
- Pin only what you need in `pyproject.toml`. Avoid utility libraries for one-liners.
- Prefer stdlib when equivalent (`pathlib`, `dataclasses`, `subprocess`, `argparse`).

## Testing
- Each new module gets a minimal pytest test file.
- Test the public API, not internals. Use fixtures for shared setup.
- Assert on outcomes, not on log lines.

## Packaging
- Include a `__main__.py` or a `[project.scripts]` entry for CLI tools.
- Expose a clean public import surface via `__init__.py`.
- Include a README with install + usage.
