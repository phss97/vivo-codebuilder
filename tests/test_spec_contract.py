from __future__ import annotations

import sys

import pytest

from codebuilder.runtime_qa import (
    check_spec_contract,
    plan_spec_hash,
    run_verification_command,
    validate_plan,
)
from codebuilder.schemas import (
    AuthoritativeAsset,
    TerminologyEntry,
    Plan,
    VerificationCommand,
)


def _plan(
    *,
    mode: str = "new_project",
    package_name: str = "my_project",
    source_path: str = "src/my_project/service.py",
    public_api: str = "create_registro(data: dict) -> dict",
) -> Plan:
    return Plan.model_validate(
        {
            "project_name": "Demo",
            "mode": mode,
            "tech_stack": ["python"],
            "language": "Portuguese",
            "revision": 2,
            "package_name": package_name,
            "identifier_contract": {
                "packages": [package_name],
                "symbols": [public_api.split("(", 1)[0]],
            },
            "work_packages": [
                {
                    "id": "wp-1",
                    "title": "Registro",
                    "what_to_build": "Build the approved record behavior.",
                    "expected_behavior": "Creates one record from valid data.",
                    "success_criteria": [
                        {"id": "criterion-1", "description": "A record is created."}
                    ],
                    "tests": [
                        {
                            "id": "test-1",
                            "criterion_ids": ["criterion-1"],
                            "path": "tests/test_service.py",
                            "test_name": "test_create_registro",
                            "expected_behavior": "The exact public function creates it.",
                        }
                    ],
                    "files": [
                        {
                            "path": source_path,
                            "purpose": "Record service.",
                            "public_api": [public_api],
                        },
                        {
                            "path": "tests/test_service.py",
                            "purpose": "Approved acceptance test.",
                            "kind": "test",
                            "public_api": [],
                        },
                    ],
                }
            ],
            "verification_commands": [
                {
                    "id": "pytest",
                    "category": "test",
                    "argv": ["pytest", "-q"],
                }
            ],
        }
    )


def test_structured_plan_renders_markdown_and_hashes_only_canonical_fields():
    plan = validate_plan(_plan())

    assert "Specification revision: 2" in plan.plan_markdown
    assert "create_registro" in plan.plan_markdown
    original_hash = plan_spec_hash(plan)
    plan.plan_markdown = "an arbitrary stale rendering"
    assert plan_spec_hash(plan) == original_hash


def test_plan_markdown_exposes_the_complete_approval_contract():
    plan = _plan()
    first = plan.work_packages[0]
    second = first.model_copy(deep=True)
    second.id = "wp-2"
    second.title = "Relatorio"
    second.depends_on = ["wp-1"]
    second.success_criteria[0].id = "criterion-2"
    second.tests[0].id = "test-2"
    second.tests[0].criterion_ids = ["criterion-2"]
    second.tests[0].path = "tests/test_report.py"
    second.tests[0].test_name = "test_build_report"
    second.files[0].path = "src/my_project/report.py"
    second.files[0].public_api = ["build_report() -> str"]
    second.files[1].path = "tests/test_report.py"
    plan.work_packages.append(second)
    plan.identifier_contract.modules = ["my_project.service"]
    plan.identifier_contract.fields = ["customer_id"]
    plan.identifier_contract.environment_variables = ["CUSTOMER_API_URL"]
    plan.identifier_contract.entry_points = ["demo=my_project.cli:main"]
    plan.terminology = [
        TerminologyEntry(
            canonical="create record",
            translations={"pt-BR": "criar registro"},
        )
    ]
    digest = "ab" * 32
    plan.authoritative_assets = [
        AuthoritativeAsset(path="schemas/customer.json", sha256=digest)
    ]
    plan.verification_commands[0] = VerificationCommand(
        id="pytest",
        category="test",
        argv=["pytest", "-q", "tests/test_service.py"],
        cwd="src",
        timeout_seconds=47,
        required=True,
    )

    markdown = validate_plan(plan).plan_markdown

    assert "- packages: `my_project`" in markdown
    assert "- modules: `my_project.service`" in markdown
    assert "- fields: `customer_id`" in markdown
    assert "- environment_variables: `CUSTOMER_API_URL`" in markdown
    assert "- entry_points: `demo=my_project.cli:main`" in markdown
    assert "`create record` — pt-BR: criar registro" in markdown
    assert f"`schemas/customer.json` — immutable=true, sha256=`{digest}`" in markdown
    assert "Depends on: `wp-1`" in markdown
    assert (
        "`pytest` [test, cwd=`src`, required=true, timeout=47s, network=false]: "
        "`pytest -q tests/test_service.py`"
    ) in markdown


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"open_questions": ["Which database?"]}, "open_questions"),
        ({"verification_commands": []}, "verification_commands"),
        ({"package_name": "projeto_ação"}, "package_name"),
    ],
)
def test_validate_plan_rejects_unapproved_or_non_ascii_contract(change, message):
    plan = _plan().model_copy(update=change)

    with pytest.raises(ValueError, match=message):
        validate_plan(plan)


def test_validate_plan_rejects_path_ownership_and_criterion_drift():
    plan = _plan()
    duplicate = plan.work_packages[0].model_copy(
        update={"id": "wp-2", "depends_on": ["wp-1"]}
    )
    plan.work_packages.append(duplicate)
    plan.work_packages[0].tests[0].criterion_ids = ["renamed-criterion"]

    with pytest.raises(ValueError) as error:
        validate_plan(plan)

    message = str(error.value)
    assert "owned by both" in message
    assert "unknown criteria" in message
    assert "criteria without tests" in message


def test_validate_plan_rejects_dependency_cycle_and_placeholder_package():
    plan = _plan()
    first = plan.work_packages[0]
    first.depends_on = ["wp-2"]
    second = first.model_copy(
        update={
            "id": "wp-2",
            "title": "TODO placeholder",
            "depends_on": ["wp-1"],
            "files": [
                first.files[0].model_copy(update={"path": "src/my_project/other.py"}),
                first.files[1].model_copy(update={"path": "tests/test_other.py"}),
            ],
            "tests": [
                first.tests[0].model_copy(
                    update={"id": "test-2", "path": "tests/test_other.py"}
                )
            ],
        }
    )
    plan.work_packages.append(second)

    with pytest.raises(ValueError) as error:
        validate_plan(plan)

    assert "placeholder text 'TODO'" in str(error.value)
    assert "cyclic" in str(error.value)


def test_validate_plan_accepts_portuguese_prose_containing_todo():
    """Portuguese "todo" means "all/whole". The planner is told to write prose in
    the job's language, so the placeholder scan must not fire on ordinary words."""
    plan = _plan()
    plan.work_packages[0].what_to_build = (
        "Orquestra todo o ciclo de vida do motor de faturamento"
    )

    assert validate_plan(plan) is plan


def test_declared_schema_requires_a_field_parity_test():
    plan = _plan()
    plan.authoritative_assets = [AuthoritativeAsset(path="db/schema.sql")]

    with pytest.raises(ValueError, match="verifies_schema"):
        validate_plan(plan)

    plan.work_packages[0].tests[0].verifies_schema = "db/schema.sql"
    assert validate_plan(plan)


def test_schema_parity_test_must_name_a_declared_schema():
    plan = _plan()
    plan.authoritative_assets = [AuthoritativeAsset(path="db/schema.sql")]
    plan.work_packages[0].tests[0].verifies_schema = "db/other.sql"

    with pytest.raises(ValueError, match="not a declared schema file"):
        validate_plan(plan)


def test_parity_test_may_name_a_schema_the_plan_never_declares():
    # A patch job's DDL already lives in the attached repo, so it is neither an
    # owned file nor necessarily an authoritative asset. Rejecting that plan would
    # tell the planner its correct answer is wrong.
    plan = _plan(mode="patch_existing")
    plan.work_packages[0].tests[0].verifies_schema = "db/schema.sql"

    assert validate_plan(plan)


def test_fields_alone_do_not_demand_a_parity_test():
    # identifier_contract.fields is non-empty on most plans; gating on it would
    # reject every job that has no schema to check against.
    plan = _plan()
    plan.identifier_contract.fields = ["customer_id"]

    assert validate_plan(plan)


def test_identifier_contract_repeats_are_deduped_not_rejected():
    # A repeat enforces the same presence check twice, so rejecting the whole
    # planner run over it is a validator defect. FAILED and failed are distinct
    # identifiers in code and must both survive.
    plan = _plan()
    plan.identifier_contract.fields = ["job_name", "job_name", "FAILED", "failed"]

    assert validate_plan(plan).identifier_contract.fields == [
        "job_name",
        "FAILED",
        "failed",
    ]


def test_public_api_contract_catches_create_to_build_translation_drift(tmp_path):
    plan = validate_plan(_plan())
    (tmp_path / "src/my_project").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/my_project/service.py").write_text(
        "def build_registro(data: dict) -> dict:\n    return data\n"
    )
    (tmp_path / "tests/test_service.py").write_text("def test_placeholder(): pass\n")

    output = check_spec_contract(str(tmp_path), plan)

    assert "exact public API missing: create_registro" in output
    assert "build_registro" not in plan.plan_markdown


def test_contract_requires_the_exact_approved_test_name(tmp_path):
    plan = validate_plan(_plan())
    (tmp_path / "src/my_project").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/my_project/service.py").write_text(
        "def create_registro(data: dict) -> dict:\n    return data\n"
    )
    (tmp_path / "tests/test_service.py").write_text("def test_build_registro(): pass\n")

    output = check_spec_contract(str(tmp_path), plan)

    assert "exact test missing: test_create_registro" in output


def test_contract_rejects_public_signature_environment_and_entry_point_drift(
    tmp_path,
):
    plan = validate_plan(_plan())
    plan.identifier_contract.fields = ["customer_id"]
    plan.identifier_contract.environment_variables = ["CUSTOMER_API_URL"]
    plan.identifier_contract.entry_points = ["demo=my_project.cli:main"]
    (tmp_path / "src/my_project").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/my_project/service.py").write_text(
        "def create_registro() -> dict:\n    return {}\n"
    )
    (tmp_path / "tests/test_service.py").write_text(
        "def test_create_registro(): pass\n"
    )
    (tmp_path / ".env.example").write_text("OTHER_URL=\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="my-project"\nversion="0.1.0"\n'
        '[project.scripts]\ndemo="my_project.wrong:main"\n'
    )

    output = check_spec_contract(str(tmp_path), plan)

    assert "signature drift for create_registro" in output
    assert "Field drift: expected exact field 'customer_id'" in output
    assert "CUSTOMER_API_URL" in output
    assert "demo=my_project.cli:main" in output


def test_package_contract_catches_meu_projeto_to_my_project_drift(tmp_path):
    plan = validate_plan(_plan())
    (tmp_path / "src/meu_projeto").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/meu_projeto/service.py").write_text(
        "def create_registro(data: dict) -> dict:\n    return data\n"
    )
    (tmp_path / "tests/test_service.py").write_text("def test_placeholder(): pass\n")

    output = check_spec_contract(str(tmp_path), plan)

    assert "Package drift: expected exact package 'my_project'" in output


def test_patch_contract_preserves_existing_portuguese_package_name(tmp_path):
    plan = validate_plan(
        _plan(
            mode="patch_existing",
            package_name="meu_projeto",
            source_path="src/meu_projeto/service.py",
        )
    )
    (tmp_path / "src/meu_projeto").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/meu_projeto/service.py").write_text(
        "def create_registro(data: dict) -> dict:\n    return data\n"
    )
    (tmp_path / "tests/test_service.py").write_text(
        "def test_create_registro(): pass\n"
    )

    assert check_spec_contract(str(tmp_path), plan) == "PASS"


def test_command_runner_uses_argv_and_does_not_inherit_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEBUILDER_TEST_SECRET", "must-not-leak")
    marker = tmp_path / "shell-expanded"
    command = VerificationCommand(
        id="safe",
        category="test",
        argv=[
            sys.executable,
            "-c",
            "import os,sys; assert 'CODEBUILDER_TEST_SECRET' not in os.environ; "
            "assert sys.argv[1] == '; touch shell-expanded'",
            "; touch shell-expanded",
        ],
    )

    result = run_verification_command(str(tmp_path), command)

    if result.returncode == 127:
        assert "sandbox unavailable" in result.stderr.lower()
        assert not marker.exists()
        return
    assert result.passed, result.stderr
    assert not marker.exists()
