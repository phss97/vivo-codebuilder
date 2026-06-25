from pathlib import Path

from codebuilder.crews.planner_crew import PlannerCrew
from codebuilder.crews.reviewer_crew import ReviewerCrew
from codebuilder.crews.writer_crew import WriterCrew


SKILLS_DIR = Path(__file__).resolve().parents[1] / "src" / "codebuilder" / "skills"


def _skill_names(agent) -> set[str]:
    names = set()
    for skill in agent.skills or []:
        names.add(skill.name if hasattr(skill, "name") else Path(skill).name)
    return names


def test_skill_packages_exist_with_matching_frontmatter() -> None:
    for name in ("rpa", "code-review-gate"):
        skill_file = SKILLS_DIR / name / "SKILL.md"
        assert skill_file.is_file()
        assert f"name: {name}" in skill_file.read_text(encoding="utf-8")


def test_planner_and_writer_have_rpa_skill(tmp_path: Path) -> None:
    assert "rpa" in _skill_names(PlannerCrew().planner())
    assert "rpa" in _skill_names(WriterCrew(workspace_dir=str(tmp_path)).writer())


def test_planner_uses_bounded_workspace_tools(tmp_path: Path) -> None:
    tool_names = {tool.name for tool in PlannerCrew(workspace_dir=str(tmp_path)).planner().tools}

    assert {"workspace_read", "workspace_list"} <= tool_names
    assert "Read a file's content" not in tool_names
    assert "List files and directories" not in tool_names


def test_reviewer_and_qa_have_rpa_and_gate_skills(tmp_path: Path) -> None:
    reviewer_crew = ReviewerCrew(workspace_dir=str(tmp_path))

    assert {"rpa", "code-review-gate"} <= _skill_names(reviewer_crew.reviewer())
    assert {"rpa", "code-review-gate"} <= _skill_names(reviewer_crew.qa_agent())
