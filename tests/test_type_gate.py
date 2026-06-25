"""Tests for the static type-check (mypy) QA gate.

The gate catches the cross-file symbol drift that ruff and the
file-existence import gate miss (attribute drift, wrong kwargs, undefined
names, bad arg types) while ignoring annotation-completeness noise so it
never bricks a job on a real repo's pre-existing untyped dependencies.
"""

from codebuilder.runtime_qa import filter_type_errors, run_final_qa


class FakeTool:
    def __init__(self, output: str):
        self.output = output
        self.calls: list[str] = []

    def _run(self, path: str = ".") -> str:
        self.calls.append(path)
        return self.output


_ATTR_ERR = "src/pkg/container.py:12: error: \"Settings\" has no attribute \"sap_host\"  [attr-defined]"
_CALLARG_ERR = "src/pkg/container.py:9: error: Unexpected keyword argument \"host\"  [call-arg]"
_NOISE_ERR = "src/pkg/x.py:1: error: Function is missing a return type annotation  [no-untyped-def]"


# --- filter_type_errors ------------------------------------------------------


def test_filter_keeps_gated_codes() -> None:
    out = "\n".join([_ATTR_ERR, _CALLARG_ERR])
    kept = filter_type_errors(out)
    assert "attr-defined" in kept
    assert "call-arg" in kept


def test_filter_drops_ungated_noise() -> None:
    assert filter_type_errors(_NOISE_ERR) == ""


def test_filter_empty_is_empty() -> None:
    assert filter_type_errors("") == ""
    assert filter_type_errors("Success: no issues found in 3 source files") == ""


# --- run_final_qa type gate --------------------------------------------------


def _passing(extra: dict) -> dict:
    base = dict(
        lint_runner=FakeTool("PASS"),
        test_runner=FakeTool("PASS\n1 passed"),
        lint_paths=["x.py"],
    )
    base.update(extra)
    return base


def test_final_qa_fails_on_gated_type_error(tmp_path) -> None:
    report = run_final_qa(str(tmp_path), **_passing({"type_runner": FakeTool(_ATTR_ERR)}))
    assert not report.passed
    assert "attr-defined" in report.type_output


def test_final_qa_passes_when_type_clean(tmp_path) -> None:
    report = run_final_qa(str(tmp_path), **_passing({"type_runner": FakeTool("PASS")}))
    assert report.passed, f"type={report.type_output!r} notes={report.integration_notes!r}"


def test_final_qa_ignores_ungated_type_noise(tmp_path) -> None:
    report = run_final_qa(str(tmp_path), **_passing({"type_runner": FakeTool(_NOISE_ERR)}))
    assert report.passed, f"type={report.type_output!r}"


def test_final_qa_type_skip_does_not_block(tmp_path) -> None:
    skip = "SKIP: mypy not installed in the runtime; review logic manually."
    report = run_final_qa(str(tmp_path), **_passing({"type_runner": FakeTool(skip)}))
    assert report.passed
