"""Final-QA repair must fix EVERY file implicated by the failures, not one.

Systemic cross-file symbol drift (the terra failure mode) spans several files
at once — a single-file repair can never converge. The repair writer returns a
CodeBundleArtifact; the flow persists and records every artifact in it.
"""

from pathlib import Path
from types import SimpleNamespace

import codebuilder.main as main
from codebuilder.main import CodebuilderFlow
from codebuilder.schemas import CodeArtifact, CodeBundleArtifact, QAReport


def _bundle_repair_crew(workspace_dir: str, files: dict[str, str]):
    class FakeWriterCrew:
        def __init__(self, workspace_dir: str):
            self.workspace_dir = workspace_dir

        def repair_crew(self):
            ws = self.workspace_dir

            class FakeCrew:
                def kickoff(self, inputs: dict):
                    artifacts = []
                    for path, content in files.items():
                        (Path(ws) / path).write_text(content, encoding="utf-8")
                        artifacts.append(
                            CodeArtifact(
                                subtask_id="final_qa_repair",
                                file_path=path,
                                language="python",
                            )
                        )
                    return SimpleNamespace(
                        pydantic=CodeBundleArtifact(
                            subtask_id="final_qa_repair", artifacts=artifacts
                        )
                    )

            return FakeCrew()

    return FakeWriterCrew


def test_repair_applies_every_implicated_file(monkeypatch, tmp_path: Path) -> None:
    flow = CodebuilderFlow()
    flow.state.workspace_dir = str(tmp_path)
    flow.state.qa_report = QAReport(
        passed=False,
        type_output="container.py:9: error: ... [call-arg]\nconftest.py:3: error: ... [attr-defined]",
        integration_notes="Type check (mypy) found cross-file symbol drift.",
    )
    monkeypatch.setenv("CODEBUILDER_MAX_FINAL_QA_REPAIRS", "1")

    monkeypatch.setattr(
        main, "run_final_qa", lambda build_dir, **_kw: QAReport(passed=True, lint_output="PASS")
    )
    monkeypatch.setattr(
        main,
        "WriterCrew",
        _bundle_repair_crew(str(tmp_path), {"container.py": "x = 1\n", "conftest.py": "y = 2\n"}),
    )

    flow._repair_final_qa_if_needed(str(tmp_path))

    assert flow.state.qa_report.passed
    written = {a.file_path for a in flow.state.artifacts}
    assert {"container.py", "conftest.py"} <= written
    assert (tmp_path / "container.py").exists() and (tmp_path / "conftest.py").exists()
