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
    StopReason,
    VerificationEvidence,
    VerificationState,
    evaluate_stop_condition,
)
from backend_ai.agent.models import ToolResult


def _plan(*step_ids: str) -> ExecutionPlan:
    steps = tuple(
        PlanStep(step_id, f"Step {step_id}", f"Objective {step_id}", "evidence", "result", risk_level=PlanRiskLevel.LOW)
        for step_id in step_ids
    )
    return ExecutionPlan(
        task="bounded task",
        normalized_task="bounded task",
        goal="complete bounded task",
        task_type=PlannerTaskType.INVESTIGATION,
        steps=steps,
        assumptions=(),
        constraints=(),
        risks=(),
        expected_changes=(),
        verification_strategy=(),
        confidence=PlannerConfidence.HIGH,
        warnings=(),
        completeness=PlanCompleteness.COMPLETE,
    )


def _success(name: str = "inspect") -> ToolResult:
    return ToolResult("call-1", name, True, data={"ok": True})


def _failure(code: str, message: str = "failed") -> ToolResult:
    return ToolResult("call-1", "tool", False, error_code=code, message=message)


def test_valid_final_without_remaining_work_is_done() -> None:
    result = evaluate_stop_condition(StopConditionRequest(plan=_plan("s1"), completed_step_ids=("s1",), final_action_valid=True))
    assert result.decision is StopDecision.DONE
    assert result.reason is StopReason.FINAL_RESPONSE


def test_valid_final_with_required_step_remaining_is_continue() -> None:
    result = evaluate_stop_condition(StopConditionRequest(plan=_plan("s1", "s2"), completed_step_ids=("s1",), final_action_valid=True))
    assert result.decision is StopDecision.CONTINUE
    assert result.reason is StopReason.PLAN_STEP_REMAINS
    assert result.remaining_required_steps == ("s2",)


def test_successful_mutation_without_verification_is_continue() -> None:
    result = evaluate_stop_condition(StopConditionRequest(
        plan=_plan("edit"),
        completed_step_ids=(),
        completion_evidence=("edit_file returned success",),
        verification=VerificationEvidence.pending("edit_file", "verification required"),
        last_tool_result=_success("edit_file"),
    ))
    assert result.decision is StopDecision.CONTINUE
    assert result.reason is StopReason.VERIFICATION_REQUIRED


def test_successful_modification_and_verification_is_done() -> None:
    result = evaluate_stop_condition(StopConditionRequest(
        plan=_plan("edit", "verify"),
        completed_step_ids=("edit", "verify"),
        completion_evidence=("all explicit targets verified",),
        verification=VerificationEvidence.passed("verify_modification", "verified"),
        last_tool_result=_success("verify_modification"),
    ))
    assert result.decision is StopDecision.DONE
    assert result.reason is StopReason.VERIFICATION_PASSED


def test_failed_verification_is_not_done() -> None:
    result = evaluate_stop_condition(StopConditionRequest(
        plan=_plan("verify"),
        completed_step_ids=("edit",),
        verification=VerificationEvidence.failed("verify_modification", "content mismatch"),
        last_tool_result=_success("verify_modification"),
    ))
    assert result.decision is StopDecision.CONTINUE
    assert result.reason is StopReason.VERIFICATION_FAILED


def test_unavailable_tool_is_blocked_not_failed() -> None:
    result = evaluate_stop_condition(StopConditionRequest(
        plan=_plan("s1"),
        missing_capabilities=("write_file",),
    ))
    assert result.decision is StopDecision.BLOCKED
    assert result.reason is StopReason.MISSING_CAPABILITY
    assert result.blocking_conditions == ("write_file",)


def test_policy_denial_is_blocked() -> None:
    result = evaluate_stop_condition(StopConditionRequest(last_tool_result=_failure("COMMAND_DENIED", "policy denied command")))
    assert result.decision is StopDecision.BLOCKED
    assert result.reason is StopReason.POLICY_DENIED


def test_fatal_infrastructure_failure_is_failed() -> None:
    result = evaluate_stop_condition(StopConditionRequest(fatal_error="infrastructure corrupted"))
    assert result.decision is StopDecision.FAILED
    assert result.reason is StopReason.INTERNAL_ERROR


def test_invalid_action_is_failed() -> None:
    result = evaluate_stop_condition(StopConditionRequest(invalid_action=True))
    assert result.decision is StopDecision.FAILED
    assert result.reason is StopReason.INVALID_ACTION


def test_emergency_bound_is_structured_non_done_block() -> None:
    result = evaluate_stop_condition(StopConditionRequest(emergency_bound_reached=True))
    assert result.decision is StopDecision.BLOCKED
    assert result.reason is StopReason.EMERGENCY_BOUND_REACHED
    assert result.emergency_bound_reached


def test_recoverable_tool_failure_can_continue_without_recovery_logic() -> None:
    result = evaluate_stop_condition(StopConditionRequest(last_tool_result=_failure("INVALID_ARGUMENT"), tool_result_recoverable=True))
    assert result.decision is StopDecision.CONTINUE
    assert result.reason is StopReason.TOOL_RESULT_REQUIRES_FOLLOW_UP


def test_incomplete_context_blocks_completion() -> None:
    result = evaluate_stop_condition(StopConditionRequest(final_action_valid=True, context_complete=False))
    assert result.decision is StopDecision.BLOCKED
    assert result.reason is StopReason.INCOMPLETE_CONTEXT


def test_stop_evaluation_is_deterministic_and_serializable() -> None:
    request = StopConditionRequest(
        plan=_plan("s1"),
        completed_step_ids=("s1",),
        completion_evidence=("verified",),
        verification=VerificationEvidence.passed("tests", "PASS"),
        final_action_valid=True,
    )
    first = evaluate_stop_condition(request)
    second = evaluate_stop_condition(request)
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["decision"] == "DONE"


def test_budget_exhaustion_precedes_continue() -> None:
    from backend_ai.agent import BudgetDimension, ExecutionBudget, ExecutionBudgetLedger

    ledger = ExecutionBudgetLedger(ExecutionBudget(max_tool_calls=0))
    decision = ledger.check("next tool", dimension=BudgetDimension.TOOL_CALLS)
    result = evaluate_stop_condition(StopConditionRequest(budget_decision=decision, final_action_valid=True))
    assert result.decision is StopDecision.BUDGET_EXHAUSTED
    assert result.reason is StopReason.BUDGET_EXHAUSTED
