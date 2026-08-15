"""Manual Phase 6.2 smoke checks; selection only, never tool execution."""

from __future__ import annotations

from backend_ai.agent import (
    ExecutionPlan,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    ToolSelectionRequest,
    ToolSelectionStatus,
    ToolSelector,
)
from backend_ai.agent.registry import ToolRegistry
from backend_ai.tools import (
    EditFileTool,
    ProjectStructureTool,
    ReadFileTool,
    RunTestsTool,
    SearchCodeTool,
    TestResultParserTool,
)


def main() -> None:
    steps = (
        PlanStep("s1", "Inspect project structure", "Inspect project structure", "smoke", "structure facts", (), PlanRiskLevel.LOW),
        PlanStep("s2", "Locate application entry point", "Locate application entry point", "smoke", "location", ("s1",), PlanRiskLevel.MEDIUM),
        PlanStep("s3", "Inspect discovered source file", "Inspect discovered source file", "smoke", "source contents", ("s2",), PlanRiskLevel.MEDIUM),
        PlanStep("s4", "Modify existing endpoint", "Modify existing endpoint", "smoke", "modified source", ("s3",), PlanRiskLevel.MEDIUM),
        PlanStep("s5", "Run tests", "Run tests", "smoke", "raw test result", ("s4",), PlanRiskLevel.MEDIUM),
        PlanStep("s6", "Interpret test results", "Interpret test results", "smoke", "semantic result", ("s5",), PlanRiskLevel.LOW),
    )
    plan = ExecutionPlan(
        "Add a health-check endpoint and test it.",
        "Add a health-check endpoint and test it.",
        "Deliver the requested feature.",
        PlannerTaskType.FEATURE,
        steps,
        (),
        (),
        (),
        ("source and test files",),
        ("inspect diff and run relevant checks later",),
        PlannerConfidence.HIGH,
        (),
        PlanCompleteness.COMPLETE,
    )
    registry = ToolRegistry((ProjectStructureTool(), SearchCodeTool(), ReadFileTool(), EditFileTool(), RunTestsTool(), TestResultParserTool()))
    result = ToolSelector().select(ToolSelectionRequest(plan, registry=registry))
    assert result.status is ToolSelectionStatus.SELECTED
    assert [decision.selected_tool for decision in result.decisions] == [
        "project_structure", "search_code", "read_file", "edit_file", "run_tests", "parse_test_result",
    ]
    assert all(decision.status is ToolSelectionStatus.SELECTED for decision in result.decisions)
    assert result.to_dict() == ToolSelector().select(ToolSelectionRequest(plan, registry=registry)).to_dict()
    print("Phase 6.2 tool-selection smoke passed")


if __name__ == "__main__":
    main()
