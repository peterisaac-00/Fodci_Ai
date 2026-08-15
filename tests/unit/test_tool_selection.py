from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from backend_ai.agent import (
    AgentLoop,
    ExecutionPlan,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    ToolCategory,
    ToolSelectionConfig,
    ToolSelectionConfidence,
    ToolSelectionDecision,
    ToolSelectionRequest,
    ToolSelectionResult,
    ToolSelectionRisk,
    ToolSelectionStatus,
    ToolSelectionValidationError,
    ToolSelectionValidator,
    ToolSelector,
)
from backend_ai.agent.registry import ToolRegistry
from backend_ai.tools import (
    DeleteFileTool,
    EditFileTool,
    ListFilesTool,
    ProjectContextTool,
    ProjectStructureTool,
    ReadFileTool,
    SearchCodeTool,
    ToolMetadata,
    WriteFileTool,
)


def _plan(*steps: PlanStep) -> ExecutionPlan:
    return ExecutionPlan(
        task="Add a health-check endpoint and test it.",
        normalized_task="Add a health-check endpoint and test it.",
        goal="Deliver the requested feature while preserving the existing project architecture and conventions.",
        task_type=PlannerTaskType.FEATURE,
        steps=tuple(steps),
        assumptions=(),
        constraints=(),
        risks=(),
        expected_changes=("minimal source and test files",),
        verification_strategy=("verify behavior and scope",),
        confidence=PlannerConfidence.HIGH,
        warnings=(),
        completeness=PlanCompleteness.COMPLETE,
    )


def _step(step_id: str, title: str, objective: str | None = None, dependencies: tuple[str, ...] = ()) -> PlanStep:
    return PlanStep(step_id, title, objective or title, "plan rationale", "expected result", dependencies, PlanRiskLevel.MEDIUM)


def test_capability_discovery_is_registry_driven_and_default_is_read_only() -> None:
    selector = ToolSelector()
    registry = ToolRegistry.default()
    capabilities = selector.capabilities_for(registry)
    assert tuple(item.tool_name for item in capabilities) == registry.names()
    assert all(item.category is ToolCategory.READ_ONLY for item in capabilities)
    assert registry.names() == ("list_files", "project_context", "project_structure", "read_file", "search_code")


def test_manual_health_check_plan_maps_to_declarative_tools() -> None:
    plan = _plan(
        _step("s1", "Inspect project structure"),
        _step("s2", "Locate application entry point", dependencies=("s1",)),
        _step("s3", "Inspect discovered source file", dependencies=("s2",)),
        _step("s4", "Implement endpoint", dependencies=("s3",)),
        _step("s5", "Add endpoint tests", dependencies=("s4",)),
        _step("s6", "Run tests", dependencies=("s5",)),
        _step("s7", "Interpret test results", dependencies=("s6",)),
    )
    registry = ToolRegistry.with_test_result_parsing()
    selector = ToolSelector()
    result = selector.select(ToolSelectionRequest(plan, registry=registry))
    assert result.status is ToolSelectionStatus.TOOL_UNAVAILABLE
    # The first five are available; execution is correctly not pretended to exist.
    assert [decision.selected_tool for decision in result.decisions[:5]] == [
        "project_structure", "search_code", "read_file", None, None,
    ]
    assert result.decisions[5].status is ToolSelectionStatus.TOOL_UNAVAILABLE
    assert result.decisions[6].status is ToolSelectionStatus.SELECTED
    assert result.decisions[6].selected_tool == "parse_test_result"


def test_inspection_mapping_and_alternatives() -> None:
    plan = _plan(_step("s1", "Find where authentication is implemented"))
    result = ToolSelector().select(ToolSelectionRequest(plan))
    decision = result.decisions[0]
    assert decision.status is ToolSelectionStatus.SELECTED
    assert decision.selected_tool == "search_code"
    assert decision.alternatives == ("list_files",)
    assert decision.risk_level is ToolSelectionRisk.READ_ONLY
    assert "read_file" not in decision.forbidden_tools
    assert "edit_file" in decision.forbidden_tools
    assert all("(" not in candidate.to_dict()["reason"] for candidate in decision.candidates)


def test_exact_file_project_discovery_and_context_mapping() -> None:
    cases = (
        ("Inspect exact file contents", "read_file"),
        ("Discover project structure", "project_structure"),
        ("Build canonical project context", "project_context"),
        ("List the project file tree", "list_files"),
    )
    for title, expected in cases:
        decision = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", title)))).decisions[0]
        assert decision.selected_tool == expected
        assert decision.risk_level is ToolSelectionRisk.READ_ONLY


def test_mutation_tools_require_explicit_registry_and_intent() -> None:
    missing = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Create a new file"))))
    assert missing.status is ToolSelectionStatus.TOOL_UNAVAILABLE
    assert missing.decisions[0].selected_tool is None

    registry = ToolRegistry.with_file_modification()
    create = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Create a new file")), registry=registry)).decisions[0]
    assert create.status is ToolSelectionStatus.SELECTED
    assert create.selected_tool == "write_file"
    assert create.risk_level is ToolSelectionRisk.MUTATING
    assert "delete_file" in create.forbidden_tools
    assert "safe-edit" in " ".join(create.warnings).casefold()

    edit = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Modify existing authentication logic")), registry=registry)).decisions[0]
    assert edit.selected_tool == "edit_file"
    assert edit.risk_level is ToolSelectionRisk.MUTATING

    delete = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Delete the obsolete file")), registry=registry)).decisions[0]
    assert delete.selected_tool == "delete_file"
    assert delete.risk_level is ToolSelectionRisk.DESTRUCTIVE
    assert "write_file" in delete.forbidden_tools


def test_git_and_execution_tools_are_selected_only_when_supplied() -> None:
    git_registry = ToolRegistry.with_git_inspection()
    status = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Inspect repository status")), registry=git_registry)).decisions[0]
    diff = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Inspect actual changes")), registry=git_registry)).decisions[0]
    assert status.selected_tool == "git_status"
    assert diff.selected_tool == "git_diff"
    assert status.risk_level is ToolSelectionRisk.READ_ONLY

    command_registry = ToolRegistry.with_command_policy()
    command = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Run an approved command")), registry=command_registry)).decisions[0]
    assert command.selected_tool == "run_command_with_policy"
    assert command.risk_level is ToolSelectionRisk.EXECUTION
    assert "CommandPolicy" in " ".join(command.warnings)

    app_registry = ToolRegistry.with_application_execution()
    app = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Launch application")), registry=app_registry)).decisions[0]
    assert app.selected_tool == "run_application"

    tests_registry = ToolRegistry.with_test_execution()
    tests = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Run tests")), registry=tests_registry)).decisions[0]
    assert tests.selected_tool == "run_tests"
    assert "edit_file" in tests.forbidden_tools

    parser_registry = ToolRegistry.with_test_result_parsing()
    parser = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Interpret test results")), registry=parser_registry)).decisions[0]
    assert parser.selected_tool == "parse_test_result"
    assert parser.risk_level is ToolSelectionRisk.READ_ONLY
    assert "run_tests" in parser.forbidden_tools


def test_unavailable_ambiguous_and_missing_prerequisites_are_structured() -> None:
    unavailable = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Run tests"))))
    assert unavailable.status is ToolSelectionStatus.TOOL_UNAVAILABLE
    assert "run_tests" in unavailable.decisions[0].selection_reason

    ambiguous = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Inspect the relevant area"))))
    assert ambiguous.status is ToolSelectionStatus.AMBIGUOUS_SELECTION
    assert ambiguous.decisions[0].selected_tool is None
    assert set(ambiguous.decisions[0].alternatives) == {"list_files", "read_file", "search_code"}

    strict = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Inspect exact file contents")), strict_prerequisites=True))
    assert strict.status is ToolSelectionStatus.MISSING_PREREQUISITES
    assert strict.decisions[0].missing_prerequisites == ("target file path is known",)

    satisfied = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Inspect exact file contents")), strict_prerequisites=True, available_inputs=("target file path is known",)))
    assert satisfied.status is ToolSelectionStatus.SELECTED


def test_selection_is_deterministic_bounded_and_serializable() -> None:
    plan = _plan(_step("s1", "Find where authentication is implemented"))
    request = ToolSelectionRequest(plan, config=ToolSelectionConfig(max_alternatives=1))
    selector = ToolSelector()
    first = selector.select(request)
    second = selector.select(request)
    assert first.to_dict() == second.to_dict()
    assert len(first.decisions[0].alternatives) <= 1
    assert first.to_dict()["available_tools"] == sorted(first.to_dict()["available_tools"])


def test_invalid_plan_step_and_validator_reject_malformed_selection() -> None:
    plan = _plan(_step("s1", "Inspect project structure"))
    invalid_request = ToolSelectionRequest(plan, selected_step_ids=("missing",))
    result = ToolSelector().select(invalid_request)
    assert result.status is ToolSelectionStatus.INVALID_REQUEST
    assert result.errors

    decision = ToolSelectionDecision(
        plan_step_id="missing",
        status=ToolSelectionStatus.SELECTED,
        selected_tool="write_file",
        tool_category=ToolCategory.MUTATING,
        selection_reason="bad",
        confidence=ToolSelectionConfidence.HIGH,
        required_inputs=(), optional_inputs=(), prerequisites=(), missing_prerequisites=(), expected_output="bad",
        alternatives=("write_file",), forbidden_tools=(), risk_level=ToolSelectionRisk.MUTATING, warnings=(), candidates=(),
    )
    malformed = ToolSelectionResult(ToolSelectionStatus.SELECTED, plan.task, (decision,), ("read_file",), (), ())
    errors = ToolSelectionValidator().validate(malformed, plan=plan)
    assert any("unknown plan_step_id" in error for error in errors)
    assert any("unavailable selected tool" in error for error in errors)
    assert any("also an alternative" in error for error in errors)
    with pytest.raises(ToolSelectionValidationError):
        ToolSelectionValidator().validate_or_raise(malformed, plan=plan)


def test_no_tool_execution_filesystem_or_subprocess_side_effects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class ExplodingTool:
        name = "search_code"
        description = "exploding test double"
        metadata = ToolMetadata(name, description, {})

        def run(self, arguments):
            raise AssertionError("ToolSelector must never call Tool.run")

    def forbidden(*args, **kwargs):
        raise AssertionError("ToolSelector must not create subprocesses")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    registry = ToolRegistry((ExplodingTool(),))
    result = ToolSelector().select(ToolSelectionRequest(_plan(_step("s1", "Find authentication implementation")), registry=registry))
    assert result.decisions[0].selected_tool == "search_code"
    assert not list(tmp_path.iterdir())


def test_planner_regression_and_selector_does_not_activate_agent_loop() -> None:
    from backend_ai.agent import Planner

    plan = Planner().create_plan("Add a users endpoint")
    result = ToolSelector().select(plan)
    assert result.decisions
    assert "ToolSelector" not in AgentLoop.__init__.__code__.co_varnames
    assert "Planner" not in ToolRegistry.default().names()
