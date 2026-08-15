"""Manual Phase 6.1 Planner smoke checks; all context is supplied in memory."""

from __future__ import annotations

from pathlib import Path

from backend_ai.agent import Planner, PlannerRequest, PlannerTaskType
from backend_ai.tools import ProjectContext


def main() -> None:
    planner = Planner()
    first = planner.plan(PlannerRequest("Add a users endpoint", project_context=None))
    second = planner.plan(PlannerRequest("Add a users endpoint", project_context=None))
    assert first.plan is not None
    assert first.to_dict() == second.to_dict()
    assert first.plan.task_type is PlannerTaskType.FEATURE
    assert first.plan.completeness.value == "PARTIAL"
    assert first.plan.steps[1].dependencies == ("step-1",)

    supplied = ProjectContext(
        root=Path("/supplied/context-only"),
        project_type="python",
        stack_summary="Python + FastAPI + pytest",
        languages=(),
        frameworks=(),
        package_managers=(),
        databases=(),
        test_frameworks=(),
        infrastructure=(),
        source_directories=("src",),
        test_directories=("tests",),
        documentation_directories=(),
        config_files=("pyproject.toml",),
        dependency_files=("pyproject.toml",),
        important_files=(),
        entry_points=(),
        project_files=(),
        confidence="high",
        evidence=("caller supplied context",),
        warnings=(),
        truncated=False,
        truncation_reason=None,
        completeness="complete",
    )
    contextual = planner.create_plan("Update the README documentation", project_context=supplied)
    assert contextual.task_type is PlannerTaskType.DOCUMENTATION_CHANGE
    assert contextual.completeness.value == "COMPLETE"
    assert all("read_file(" not in str(step.to_dict()) for step in contextual.steps)
    print("Phase 6.1 planner smoke passed")


if __name__ == "__main__":
    main()
