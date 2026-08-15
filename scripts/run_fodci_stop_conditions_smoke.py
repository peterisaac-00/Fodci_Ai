"""Bounded Phase 6.4 stop-condition smoke; no tool, filesystem, or subprocess execution."""
from __future__ import annotations

from backend_ai.agent import (
    ExecutionPlan,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    StopConditionRequest,
    StopDecision,
    VerificationEvidence,
    evaluate_stop_condition,
)
from backend_ai.agent.models import ToolResult


def plan(*ids: str) -> ExecutionPlan:
    return ExecutionPlan(
        "smoke task",
        "smoke task",
        "bounded smoke",
        PlannerTaskType.INVESTIGATION,
        tuple(PlanStep(item, item, item, "evidence", "result", risk_level=PlanRiskLevel.LOW) for item in ids),
        (), (), (), (), (), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE,
    )


def main() -> None:
    read_only = evaluate_stop_condition(StopConditionRequest(plan=plan("inspect", "observe"), completed_step_ids=("inspect",), completion_evidence=("first observation",)))
    assert read_only.decision is StopDecision.CONTINUE

    verified = evaluate_stop_condition(StopConditionRequest(
        plan=plan("edit", "verify"),
        completed_step_ids=("edit", "verify"),
        last_tool_result=ToolResult("call-2", "verify_modification", True, data={"success": True, "complete": True}),
        completion_evidence=("explicit targets verified",),
        verification=VerificationEvidence.passed("verify_modification", "verification passed"),
    ))
    assert verified.decision is StopDecision.DONE

    unavailable = evaluate_stop_condition(StopConditionRequest(plan=plan("edit"), missing_capabilities=("write_file",)))
    assert unavailable.decision is StopDecision.BLOCKED

    bounded = evaluate_stop_condition(StopConditionRequest(emergency_bound_reached=True))
    assert bounded.decision is StopDecision.BLOCKED
    assert not bounded.decision is StopDecision.DONE
    print("Phase 6.4 stop-condition smoke passed")


if __name__ == "__main__":
    main()
