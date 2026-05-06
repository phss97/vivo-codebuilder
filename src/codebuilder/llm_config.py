"""Shared LLM / embedder env-var configuration helpers."""

from __future__ import annotations

import os


def embedder_config() -> dict:
    provider = os.environ.get("CODEBUILDER_EMBEDDER_PROVIDER", "openai")
    model = os.environ.get("CODEBUILDER_EMBEDDER_MODEL", "text-embedding-3-small")
    return {"provider": provider, "config": {"model_name": model}}
