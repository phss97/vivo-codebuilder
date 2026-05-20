---
name: code-review-gate
description: Universal acceptance checklist applied before any generated codebase is handed off — structure, tests, configuration, and code quality, independent of domain.
---

# Code Review Gate

Apply this gate to any generated project, regardless of language or domain.
Domain- or stack-specific patterns belong in the matching skill (e.g.
`rpa` for RPA projects, future skills for other internal standards).

## Pass Criteria

- Project layout matches the chosen tech stack's standard packaging
  (e.g. `pyproject.toml` + `src/` for Python packages, `package.json` + `src/`
  for JS, etc.). A loose script is acceptable only when the brief explicitly
  asks for one.
- Source and test code are separated. Tests live in the conventional
  location for the stack (`tests/`, `__tests__/`, etc.).
- Tests cover the core behavior from the brief, at least one failure path,
  and any retry / error semantics declared in the plan.
- Configuration is centralized; `.env.example` (or the stack's equivalent)
  documents every required value.
- Secrets, credentials, tokens, and machine-specific absolute paths are
  loaded from configuration — never hardcoded.
- Logging is structured enough to make failures debuggable.
- Public functions / methods carry type hints (or the stack's equivalent)
  and use meaningful names.
- Lint, format, and static-analysis tools declared by the plan all pass.
- The plan's documented entry point actually exists and runs.
- Files contain complete implementations — no TODOs, no `pass` / `...` /
  `NotImplementedError` stubs, no half-finished functions.

## Fail Conditions

- Tests missing for non-trivial business behavior, or covering only the
  happy path.
- Hardcoded credentials, tokens, or absolute paths.
- Cross-cutting concerns (logging, config, error handling) implemented
  inconsistently across modules.
- Generated output would need major rewrites before first handoff.
- Lint or test commands declared by the plan do not pass.

## Layering with Domain Skills

When a domain skill is active (e.g. `rpa`, or any future stack-specific
skill), apply its acceptance criteria *in addition to* the universal items
above. Domain checks supersede generic checks only where they explicitly
conflict.
