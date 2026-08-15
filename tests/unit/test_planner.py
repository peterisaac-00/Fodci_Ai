from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from backend_ai.agent import (
    ExecutionPlan,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlanStepStatus,
    PlanValidationError,
    PlanValidator,
    Planner,
    PlannerConfig,
    PlannerConfidence,
    PlannerRequest,
    PlannerResultStatus,
    PlannerTaskType,
    create_plan,
)
from backend_ai.agent.loop import AgentLoop
from backend_ai.tools import Detection, ProjectContext


def _context(*, partial: bool = False) -> ProjectContext:
    return ProjectContext(
        root=Path("/tmp/fodci-planner-project"),
        project_type="python",
        stack_summary="Python + FastAPI + PostgreSQL + pytest",
        languages=(),
        frameworks=(Detection("FastAPI", "high", ("src/api.py: FastAPI evidence",)),),
        package_managers=(),
        databases=(Detection("PostgreSQL", "medium", ("pyproject.toml: dependency evidence",)),),
        test_frameworks=(Detection("pytest", "high", ("pyproject.toml: pytest evidence",)),),
        infrastructure=(),
        source_directories=("src",),
        test_directories=("tests",),
        documentation_directories=("docs",),
        config_files=("pyproject.toml",),
        dependency_files=("pyproject.toml",),
        important_files=("pyproject.toml", "README.md"),
        entry_points=(),
        project_files=("pyproject.toml", "src/api.py", "tests/test_api.py"),
        confidence="medium" if partial else "high",
        evidence=("pyproject.toml: project evidence", "src/api.py: source evidence"),
        warnings=("bounded discovery was partial",) if partial else (),
        truncated=partial,
        truncation_reason="max_files" if partial else None,
        completeness="partial" if partial else "complete",
    )


def test_feature_plan_is_structured_context_aware_and_deterministic() -> None:
    request = PlannerRequest("Add a users endpoint that returns all users.", _context())
    first = Planner().plan(request)
    second = Planner().plan(request)

    assert first.status is PlannerResultStatus.CREATED
    assert first.plan is not None
    assert first.to_dict() == second.to_dict()
    assert first.plan.task_type is PlannerTaskType.FEATURE
    assert first.plan.confidence.value == "HIGH"
    assert first.plan.completeness is PlanCompleteness.COMPLETE
    assert "Python + FastAPI + PostgreSQL + pytest" in " ".join(first.plan.constraints)
    assert any("pytest" in item for item in first.plan.constraints)
    assert first.plan.steps[0].step_id == "step-1"
    assert first.plan.steps[1].dependencies == ("step-1",)
    assert all("(" not in step.objective for step in first.plan.steps)
    assert all("read_file(" not in str(step.to_dict()) for step in first.plan.steps)


def test_task_categories_are_conservative() -> None:
    cases = {
        "Fix the broken authentication flow": PlannerTaskType.BUG_FIX,
        "Refactor the repository layer": PlannerTaskType.REFACTOR,
        "Add tests for the users service": PlannerTaskType.TEST_ADDITION,
        "Update configuration settings": PlannerTaskType.CONFIGURATION_CHANGE,
        "Update the README documentation": PlannerTaskType.DOCUMENTATION_CHANGE,
        "Change the project dependency version": PlannerTaskType.DEPENDENCY_CHANGE,
        "Investigate why the API returns 500": PlannerTaskType.INVESTIGATION,
        "Do something mysterious": PlannerTaskType.UNKNOWN,
    }
    for task, expected in cases.items():
        plan = Planner().create_plan(task, project_context=_context())
        assert plan.task_type is expected


def test_ambiguous_unknown_and_missing_context_reduce_confidence_without_inventing_details() -> None:
    ambiguous = Planner().plan(PlannerRequest("Fix authentication", _context()))
    assert ambiguous.plan is not None
    assert ambiguous.plan.task_type is PlannerTaskType.BUG_FIX
    assert ambiguous.plan.confidence.value == "LOW"
    assert ambiguous.plan.completeness is PlanCompleteness.REQUIRES_CLARIFICATION
    assert any("ambiguous" in warning for warning in ambiguous.plan.warnings)
    assert any("clarification" in item for item in ambiguous.plan.assumptions)

    unknown = Planner().create_plan("Improve the API", project_context=_context())
    assert unknown.task_type is PlannerTaskType.UNKNOWN
    assert unknown.completeness is PlanCompleteness.REQUIRES_CLARIFICATION
    assert "JWT" not in str(unknown.to_dict())

    no_context = Planner().plan(PlannerRequest("Add a login endpoint"))
    assert no_context.plan is not None
    assert no_context.plan.confidence.value == "LOW"
    assert no_context.plan.completeness is PlanCompleteness.PARTIAL
    assert any("not supplied" in warning for warning in no_context.plan.warnings)
    assert any("not confirmed" in item for item in no_context.plan.assumptions)

    partial = Planner().create_plan("Add a login endpoint", project_context=_context(partial=True))
    assert partial.completeness is PlanCompleteness.PARTIAL
    assert partial.confidence.value == "MEDIUM"


def test_expected_changes_verification_risks_and_inspection_first_steps() -> None:
    dependency = Planner().create_plan("Change the dependency version", project_context=_context())
    assert dependency.risks[0].level is PlanRiskLevel.HIGH
    assert any("dependency" in item.casefold() for item in dependency.expected_changes)
    assert dependency.verification_strategy
    assert "install" not in dependency.steps[1].objective.casefold()

    docs = Planner().create_plan("Update the README documentation", project_context=_context())
    assert docs.risks[0].level is PlanRiskLevel.LOW
    assert "documentation" in docs.steps[1].objective.casefold()
    assert "unnecessary code execution" in docs.verification_strategy[0]


def test_budget_truncation_is_explicit_and_utf8_safe() -> None:
    config = PlannerConfig(max_steps=3, max_assumptions=1, max_constraints=2, max_risks=1, max_warnings=2, max_task_length=32, max_plan_text_length=40)
    result = Planner(config=config).plan(PlannerRequest("أضف ميزة جديدة " * 50, _context(), config))
    assert result.plan is not None
    assert len(result.plan.steps) <= 3
    assert len(result.plan.task.encode("utf-8")) <= 32
    assert result.plan.completeness is PlanCompleteness.PARTIAL or result.plan.completeness is PlanCompleteness.REQUIRES_CLARIFICATION
    assert result.validation.valid is True
    assert any("truncated" in warning for warning in result.plan.warnings)


def test_validator_rejects_duplicate_unknown_cycle_and_execution_payloads() -> None:
    duplicate = ExecutionPlan(
        "task", "task", "goal", PlannerTaskType.FEATURE,
        (PlanStep("s1", "one", "objective", "why", "result"), PlanStep("s1", "two", "objective", "why", "result")),
        (), (), (), (), (),
        PlannerConfidence.HIGH,
        (), PlanCompleteness.COMPLETE,
    )
    assert not PlanValidator().validate(duplicate).valid

    unknown_dependency = PlanStep("s1", "one", "objective", "why", "result", ("missing",))
    plan = ExecutionPlan("task", "task", "goal", PlannerTaskType.FEATURE, (unknown_dependency,), (), (), (), (), (), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE)
    validation = PlanValidator().validate(plan)
    assert not validation.valid
    assert any("unknown step" in error for error in validation.errors)

    cycle = ExecutionPlan(
        "task", "task", "goal", PlannerTaskType.FEATURE,
        (PlanStep("s1", "one", "objective", "why", "result", ("s2",)), PlanStep("s2", "two", "objective", "why", "result", ("s1",))),
        (), (), (),         (), (), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE,

    )
    assert any("DAG" in error for error in PlanValidator().validate(cycle).errors)

    injected = PlanStep("s1", "Call read_file('x')", "objective", "why", "result")
    injected_plan = ExecutionPlan("task", "task", "goal", PlannerTaskType.FEATURE, (injected,), (), (), (), (), (), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE)
    assert any("payload" in error for error in PlanValidator().validate(injected_plan).errors)
    with pytest.raises(PlanValidationError):
        PlanValidator().validate_or_raise(injected_plan)


def test_invalid_enum_and_excessive_steps_are_rejected() -> None:
    with pytest.raises(ValueError):
        PlanStep("s1", "title", "objective", "why", "result", risk_level="HIGH")  # type: ignore[arg-type]
    steps = tuple(PlanStep(f"s{index}", "title", "objective", "why", "result") for index in range(4))
    plan = ExecutionPlan("task", "task", "goal", PlannerTaskType.FEATURE, steps, (), (), (), (), (), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE)
    assert any("max_steps" in error for error in PlanValidator().validate(plan, config=PlannerConfig(max_steps=3)).errors)


def test_plan_convenience_api_and_agent_loop_remain_separate() -> None:
    plan = create_plan("Add a users endpoint", project_context=_context())
    assert isinstance(plan, ExecutionPlan)
    assert "planner" not in AgentLoop.__init__.__code__.co_varnames


def test_planner_has_no_filesystem_subprocess_or_network_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Planner must not execute or inspect the project")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    plan = Planner().create_plan("إضافة endpoint للمستخدمين", project_context=None)
    assert plan.normalized_task.startswith("إضافة")
    assert plan.completeness is PlanCompleteness.PARTIAL
