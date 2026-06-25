"""Targeted tests for the three Vivo-feedback fixes:

1. Symbol contract — ``FileSkeleton.public_api`` round-trips and
   ``CodebuilderFlow._symbol_contract`` renders the project-wide name map.
2. Output language — the planner emits ``Plan.language`` and the ``plan`` body
   resolves it onto ``state.language`` (caller override wins, else the planner's
   detection, else English).
3. Windows ``.exe`` build kit — ``run_rpa_deterministic_gate`` requires
   ``build.spec`` + a Windows build script + ``pyinstaller`` in ``pyproject``.
"""

from pathlib import Path
from types import SimpleNamespace

from codebuilder.crews.planner_crew import PlannerCrew
from codebuilder.main import CodebuilderFlow
from codebuilder.runtime_qa import run_rpa_deterministic_gate
from codebuilder.schemas import FileSkeleton, Plan, PlanSkeleton, SubTask


# ``plan`` is wrapped by @listen + @human_feedback; reach the raw body so the
# unit test doesn't trigger the feedback request (which would raise
# HumanFeedbackPending). Same unwrap pattern as test_revise_plan_recovery.
_PLAN_BODY = CodebuilderFlow.plan.unwrap().__wrapped__


class _FakeResult:
    def __init__(self, pydantic) -> None:
        self.pydantic = pydantic


class _FakeCrew:
    def __init__(self, result: _FakeResult) -> None:
        self._result = result

    def kickoff(self, inputs=None):
        return self._result


def _plan(language: str = "", *, public_api: list[str] | None = None) -> Plan:
    return Plan(
        project_name="demo",
        mode="new_project",
        tech_stack=["python"],
        domain="rpa",
        language=language,
        subtasks=[
            SubTask(
                id="s01",
                title="domain",
                description="d",
                files=[
                    FileSkeleton(
                        path="src/demo/domain/registro.py",
                        purpose="Registro entity and factory.",
                        public_api=public_api or [],
                    )
                ],
                test_criteria="t",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Issue 1 — symbol contract                                                   #
# --------------------------------------------------------------------------- #
def test_file_skeleton_public_api_round_trips() -> None:
    api = ["create_registro(data: dict) -> Registro", "class RegistroRepository(Protocol)"]
    plan = _plan(public_api=api)

    restored = Plan.model_validate(plan.model_dump())

    assert restored.subtasks[0].files[0].public_api == api
    # Default is an empty list, not None, for files with no importable API.
    assert FileSkeleton(path="README.md", purpose="Docs.").public_api == []


def test_symbol_contract_renders_only_files_with_public_api() -> None:
    flow = CodebuilderFlow()
    flow.state.plan = Plan(
        project_name="demo",
        mode="new_project",
        tech_stack=["python"],
        subtasks=[
            SubTask(
                id="s01",
                title="domain",
                description="d",
                files=[
                    FileSkeleton(
                        path="src/demo/domain/registro.py",
                        purpose="Registro factory.",
                        public_api=["create_registro(data: dict) -> Registro"],
                    ),
                    FileSkeleton(path="README.md", purpose="Docs."),  # no public_api
                ],
                test_criteria="t",
            )
        ],
    )

    contract = flow._symbol_contract()

    assert "src/demo/domain/registro.py → [create_registro(data: dict) -> Registro]" in contract
    assert "README.md" not in contract


def test_symbol_contract_sentinels() -> None:
    # No plan, no written artifacts → nothing to contract.
    no_plan = CodebuilderFlow()
    assert no_plan._symbol_contract() == "(no symbols available)"

    # Plan present but no declared public_api and no artifacts on disk yet.
    no_symbols = CodebuilderFlow()
    no_symbols.state.plan = _plan(public_api=[])
    assert no_symbols._symbol_contract() == "(no symbols available)"


def test_symbol_contract_prefers_real_written_api(tmp_path) -> None:
    """Once a file is written, the contract carries its REAL fields (extracted
    from disk), so a later writer can't drift onto an invented attribute."""
    from codebuilder.schemas import CodeArtifact

    settings_src = (
        "class Settings:\n"
        "    database_url: str\n"
        "    sap_endpoint: str\n"
    )
    pkg = tmp_path / "src" / "demo" / "config"
    pkg.mkdir(parents=True)
    (pkg / "settings.py").write_text(settings_src, encoding="utf-8")

    flow = CodebuilderFlow()
    flow.state.artifacts = [
        CodeArtifact(subtask_id="s1", file_path="src/demo/config/settings.py", language="python")
    ]
    contract = flow._symbol_contract(str(tmp_path))
    assert "demo.config.settings" in contract
    assert "database_url" in contract and "sap_endpoint" in contract
    assert "sap_host" not in contract  # the drift attribute is never present


# --------------------------------------------------------------------------- #
# Issue 2 — output language resolution                                        #
# --------------------------------------------------------------------------- #
def test_plan_language_round_trips_on_skeleton_and_plan() -> None:
    skeleton = PlanSkeleton(
        project_name="demo",
        mode="new_project",
        tech_stack=["python"],
        language="Portuguese",
        files=[FileSkeleton(path="src/demo/__init__.py", purpose="Package init.")],
    )
    assert PlanSkeleton.model_validate(skeleton.model_dump()).language == "Portuguese"

    plan = _plan(language="Portuguese")
    assert Plan.model_validate(plan.model_dump()).language == "Portuguese"


def test_plan_body_caller_override_wins(monkeypatch) -> None:
    flow = CodebuilderFlow()
    flow.state.language = "English"  # explicit kickoff override
    monkeypatch.setattr(
        PlannerCrew, "crew", lambda self: _FakeCrew(_FakeResult(_plan(language="Portuguese")))
    )

    _PLAN_BODY(flow)

    assert flow.state.language == "English"
    assert flow.state.status == "awaiting_approval"


def test_plan_body_uses_planner_detection_when_no_override(monkeypatch) -> None:
    flow = CodebuilderFlow()  # no override
    monkeypatch.setattr(
        PlannerCrew, "crew", lambda self: _FakeCrew(_FakeResult(_plan(language="Portuguese")))
    )

    _PLAN_BODY(flow)

    assert flow.state.language == "Portuguese"


def test_plan_body_defaults_to_english_when_both_empty(monkeypatch) -> None:
    flow = CodebuilderFlow()  # no override
    monkeypatch.setattr(
        PlannerCrew, "crew", lambda self: _FakeCrew(_FakeResult(_plan(language="")))
    )

    _PLAN_BODY(flow)

    assert flow.state.language == "English"


# --------------------------------------------------------------------------- #
# Issue 3 — Windows .exe build kit gate                                       #
# --------------------------------------------------------------------------- #
def _pyproject(*, with_pyinstaller: bool) -> str:
    dev = ["ruff", "pytest", "pytest-cov", "mypy"]
    if with_pyinstaller:
        dev.append("pyinstaller")
    deps = ", ".join(f'"{d}"' for d in dev)
    return (
        "[project]\n"
        'name = "demo"\n'
        'requires-python = ">=3.13"\n\n'
        "[project.optional-dependencies]\n"
        f"dev = [{deps}]\n\n"
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n'
    )


def _write_rpa_workspace(root: Path, *, build_kit: bool = True) -> None:
    (root / "pyproject.toml").write_text(_pyproject(with_pyinstaller=build_kit), encoding="utf-8")
    (root / ".env.example").write_text("CCM_URL=\n", encoding="utf-8")

    pkg = root / "src" / "demo"
    for layer in ("domain", "application", "infrastructure"):
        (pkg / layer).mkdir(parents=True, exist_ok=True)
        (pkg / layer / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    # RPA component + config/logging path fragments the gate scans for.
    (pkg / "application" / "orchestrator.py").write_text("def main() -> None:\n    ...\n", encoding="utf-8")
    (pkg / "infrastructure" / "producer.py").write_text("class Producer:\n    ...\n", encoding="utf-8")
    (pkg / "infrastructure" / "consumer.py").write_text("class Consumer:\n    ...\n", encoding="utf-8")
    (pkg / "config.py").write_text("SETTINGS = {}\n", encoding="utf-8")
    (pkg / "logging.py").write_text("def get_logger():\n    ...\n", encoding="utf-8")

    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_demo.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    if build_kit:
        (root / "build.spec").write_text("# PyInstaller spec\n", encoding="utf-8")
        (root / "build.ps1").write_text("pyinstaller build.spec\n", encoding="utf-8")


def test_rpa_gate_passes_with_full_build_kit(tmp_path: Path) -> None:
    _write_rpa_workspace(tmp_path, build_kit=True)

    result = run_rpa_deterministic_gate(str(tmp_path))

    assert result.passed is True, result.issues


def test_rpa_gate_fails_without_build_kit(tmp_path: Path) -> None:
    _write_rpa_workspace(tmp_path, build_kit=False)

    result = run_rpa_deterministic_gate(str(tmp_path))

    assert result.passed is False
    joined = "\n".join(result.issues)
    assert "build.spec" in joined
    assert "build script" in joined
    assert "pyinstaller" in joined


def test_full_architecture_gate_threads_language(monkeypatch, tmp_path: Path) -> None:
    # _rpa_full_gate only reaches the reviewer kickoff when the deterministic
    # gate passes; build a full workspace and capture the language it forwards.
    import codebuilder.runtime_qa as runtime_qa

    _write_rpa_workspace(tmp_path, build_kit=True)
    captured: dict = {}

    class _GateCrew:
        def kickoff(self, inputs: dict):
            captured.update(inputs)
            from codebuilder.schemas import ReviewResult

            return SimpleNamespace(
                pydantic=ReviewResult(subtask_id="architecture_gate", passed=True)
            )

    class _Reviewer:
        def __init__(self, workspace_dir: str):
            self.workspace_dir = workspace_dir

        def architecture_gate_crew(self):
            return _GateCrew()

    monkeypatch.setattr("codebuilder.crews.reviewer_crew.ReviewerCrew", _Reviewer)

    result = runtime_qa.run_full_architecture_gate(str(tmp_path), _plan(), language="Portuguese")

    assert result.passed is True
    assert captured["language"] == "Portuguese"
