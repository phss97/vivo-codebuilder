"""Focused contracts for the Claude role wrappers."""

from __future__ import annotations

import asyncio

import pytest

from codebuilder import cc_agent
from codebuilder.schemas import IntakeAssessment, ProductionReview


class _Text:
    def __init__(self, text: str) -> None:
        self.text = text


class _Assistant:
    def __init__(self, text: str) -> None:
        self.content = [_Text(text)]
        self.usage = None


class _Result:
    def __init__(self, structured_output=None) -> None:
        self.structured_output = structured_output
        self.subtype = "success"
        self.is_error = False
        self.usage = None
        self.model_usage = None
        self.total_cost_usd = None
        self.num_turns = None
        self.duration_ms = None


def _capturing_query(messages):
    async def _query(*, prompt, options):
        _query.prompt = prompt
        _query.options = options
        for message in messages:
            yield message

    return _query


VALID_INTAKE = {
    "ready": False,
    "understood_scope": "Add one validated API endpoint.",
    "evidence_inspected": ["pyproject.toml", "src/demo/api.py"],
    "authoritative_assets": [
        {"path": "schemas/customer.sql", "immutable": True, "sha256": "abc"}
    ],
    "blocking_questions": [
        {
            "id": "Q1",
            "question": "Which authentication contract is authoritative?",
            "rationale": "Two incompatible contracts are attached.",
        }
    ],
    "detected_stack": ["python"],
    "missing_verification_commands": ["integration test command"],
}


def test_intake_is_read_only_structured_and_uses_planner_settings():
    query = _capturing_query([_Result(VALID_INTAKE)])

    assessment = asyncio.run(
        cc_agent.run_intake(cwd=".", prompt="assess", query_fn=query)
    )

    assert assessment == IntakeAssessment.model_validate(VALID_INTAKE)
    assert query.options.model == cc_agent.PLANNER_MODEL
    assert query.options.effort == cc_agent.PLANNER_EFFORT
    assert set(query.options.allowed_tools) == {"Read", "Grep", "Glob", "Skill"}
    assert set(query.options.tools) == {"Read", "Grep", "Glob", "Skill"}
    assert set(query.options.disallowed_tools) == {"Write", "Edit", "Bash"}
    assert "Do not plan or implement" in query.options.system_prompt
    assert "Preserve every code" in query.options.system_prompt


def test_test_author_has_mutating_tools_and_exact_file_contract():
    query = _capturing_query([_Assistant("tests written"), _Result()])

    transcript = asyncio.run(
        cc_agent.run_test_author(
            cwd=".",
            prompt="write criterion C1",
            declared_test_files=["tests/test_api.py", "tests/test_contract.py"],
            query_fn=query,
        )
    )

    assert transcript == "tests written"
    assert query.options.model == cc_agent.EXECUTOR_MODEL
    assert query.options.effort == cc_agent.EXECUTOR_EFFORT
    assert query.options.permission_mode == "bypassPermissions"
    assert query.options.sandbox["enabled"] is True
    assert query.options.sandbox["allowUnsandboxedCommands"] is False
    assert {"Write", "Edit", "Bash"} <= set(query.options.allowed_tools)
    assert set(query.options.tools) == {
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Glob",
        "Grep",
        "Skill",
    }
    assert "- tests/test_api.py" in query.options.system_prompt
    assert "- tests/test_contract.py" in query.options.system_prompt
    assert "Do not modify production source" in query.options.system_prompt
    assert "must never be hidden by weakening a test" in query.options.system_prompt

    guard = query.options.hooks["PreToolUse"][0].hooks[0]
    allowed = asyncio.run(
        guard({"tool_input": {"file_path": "tests/new.py"}}, None, {"signal": None})
    )
    denied = asyncio.run(
        guard({"tool_input": {"file_path": "/tmp/outside.py"}}, None, {"signal": None})
    )
    assert allowed == {}
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize(
    "paths",
    [[], ["../src/app.py"], ["/tmp/test_app.py"], [r"C:\\tmp\\test_app.py"]],
)
def test_test_author_rejects_missing_or_unsafe_paths(paths):
    query = _capturing_query([_Result()])

    with pytest.raises(ValueError):
        asyncio.run(
            cc_agent.run_test_author(
                cwd=".", prompt="write tests", declared_test_files=paths, query_fn=query
            )
        )


def test_reviewer_contract_is_generic_and_read_only():
    query = _capturing_query([_Result({"passed": True, "issues": []})])

    review = asyncio.run(
        cc_agent.run_reviewer(
            cwd=".",
            prompt="Review approved criterion C1.",
            system_prompt="Use the supplied criterion IDs in every issue.",
            query_fn=query,
        )
    )

    assert review == ProductionReview(passed=True)
    assert set(query.options.disallowed_tools) == {"Write", "Edit", "Bash"}
    assert "approved specification and success criteria" in query.options.system_prompt
    assert "Use the supplied criterion IDs" in query.options.system_prompt
    assert "RPA" not in query.options.system_prompt
