from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import (
    CompletionDecision,
    CompletionStatus,
    EvidenceStrength,
    ExecutionBudget,
    ExecutionBudgetLedger,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    RecoveryContext,
    TaskCompletionCriterion,
    TaskCompletionEvidence,
    TaskCompletionRequest,
    TaskCompletionVerifier,
    ToolResult,
    VerificationEvidence,
    VerificationState,
    decide_recovery,
    verify_task_completion,
)
from backend_ai.agent.planner import ExecutionPlan
from backend_ai.tools.base import ToolErrorCode


def _plan(*steps: PlanStep, verification_strategy: tuple[str, ...] = ()) -> ExecutionPlan:
    return ExecutionPlan("task", "task", "goal", PlannerTaskType.FEATURE, steps, (), (), (), (), verification_strategy, PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE)


def _step(step_id: str, *, verification_required: bool = False) -> PlanStep:
    return PlanStep(step_id, step_id, step_id, "reason", "result", (), PlanRiskLevel.LOW, verification_required)


def test_complete_plan_with_direct_evidence_is_complete() -> None:
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1")), completed_step_ids=("s1",), tool_results=(ToolResult("c", "read_file", True, data={"ok": True}),), evidence=(TaskCompletionEvidence("tool", "direct observation", EvidenceStrength.DIRECT),)))
    assert result.decision is CompletionDecision.COMPLETE
    assert result.status is CompletionStatus.COMPLETE


def test_final_claim_without_required_plan_step_is_incomplete() -> None:
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1")), final_response="done"))
    assert result.decision is CompletionDecision.CONTINUE
    assert "plan:s1" in result.remaining_criteria


def test_required_verification_pending_is_not_complete() -> None:
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1"), verification_strategy=("run tests",)), completed_step_ids=("s1",), verification=VerificationEvidence.pending("tests", "not run")))
    assert result.decision is CompletionDecision.VERIFICATION_UNAVAILABLE


def test_verification_passed_can_satisfy_required_criterion() -> None:
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1"), verification_strategy=("verify",)), completed_step_ids=("s1",), verification=VerificationEvidence.passed("verify_modification", "verified")))
    assert result.decision is CompletionDecision.COMPLETE


def test_failed_tool_blocks_completion() -> None:
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1")), completed_step_ids=("s1",), tool_results=(ToolResult("c", "read_file", False, error_code=ToolErrorCode.FILE_NOT_FOUND.value, message="missing"),)))
    assert result.decision is CompletionDecision.BLOCKED
    assert result.status is CompletionStatus.BLOCKED


def test_critical_unexpected_modification_blocks_completion() -> None:
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1")), completed_step_ids=("s1",), critical_unexpected_modifications=("secrets.txt",)))
    assert result.decision is CompletionDecision.BLOCKED


def test_noncritical_unexpected_modification_is_explicitly_unverified() -> None:
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1")), completed_step_ids=("s1",), unexpected_modifications=("notes.txt",)))
    assert result.decision is CompletionDecision.VERIFICATION_UNAVAILABLE


def test_budget_exhaustion_blocks_completion() -> None:
    ledger = ExecutionBudgetLedger(ExecutionBudget(max_tool_calls=0))
    ledger.check_tool_operation("read_file")
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1")), completed_step_ids=("s1",), budget=ledger.snapshot()))
    assert result.decision is CompletionDecision.BLOCKED


def test_recovery_user_intervention_blocks_completion() -> None:
    failure = ToolResult("c", "edit_file", False, error_code=ToolErrorCode.CONCURRENT_MODIFICATION.value, message="user changed file")
    recovery = decide_recovery(RecoveryContext(failure, "edit_file"))
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1")), recovery=recovery))
    assert result.decision is CompletionDecision.BLOCKED


def test_investigation_can_complete_without_mutation_or_tests() -> None:
    plan = ExecutionPlan("investigate", "investigate", "find cause", PlannerTaskType.INVESTIGATION, (_step("s1"),), (), (), (), (), (), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE)
    result = verify_task_completion(TaskCompletionRequest("investigate", plan, completed_step_ids=("s1",), evidence=(TaskCompletionEvidence("search_code", "finding", EvidenceStrength.DIRECT),)))
    assert result.complete


def test_deterministic_repeated_evaluation() -> None:
    request = TaskCompletionRequest("task", _plan(_step("s1")), completed_step_ids=("s1",), evidence=(TaskCompletionEvidence("tool", "same", EvidenceStrength.DIRECT),))
    assert verify_task_completion(request).to_dict() == verify_task_completion(request).to_dict()


def test_tool_pass_without_task_criterion_is_not_automatically_enough() -> None:
    criterion = TaskCompletionCriterion("expected-file", "expected file is verified", kind="custom")
    result = verify_task_completion(TaskCompletionRequest("task", criteria=(criterion,), tool_results=(ToolResult("c", "run_tests", True, data={"status": "PASS"}),)))
    assert result.decision is not CompletionDecision.COMPLETE


def test_evidence_is_bounded() -> None:
    evidence = TaskCompletionEvidence("tool", "x" * 5000)
    assert len(evidence.detail) <= 1024


def test_required_regression_verification_missing_is_not_complete() -> None:
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1")), completed_step_ids=("s1",), evidence=(TaskCompletionEvidence("targeted", "targeted test passed", EvidenceStrength.DIRECT),), regression_required=True))
    assert result.decision is CompletionDecision.VERIFICATION_UNAVAILABLE
    assert "regression" in result.remaining_criteria


def test_regression_detected_blocks_completion_even_after_targeted_pass() -> None:
    regression = SimpleNamespace(status=SimpleNamespace(value="REGRESSION_DETECTED"), comparison=SimpleNamespace(message="new failure detected after modification"))
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1")), completed_step_ids=("s1",), evidence=(TaskCompletionEvidence("targeted", "targeted test passed", EvidenceStrength.DIRECT),), regression_required=True, regression_protection=regression))
    assert result.decision is CompletionDecision.BLOCKED
    assert result.status is CompletionStatus.BLOCKED


def test_regression_free_evidence_allows_completion() -> None:
    regression = SimpleNamespace(status=SimpleNamespace(value="REGRESSION_FREE"), comparison=SimpleNamespace(message="no new failure detected"))
    result = verify_task_completion(TaskCompletionRequest("task", _plan(_step("s1")), completed_step_ids=("s1",), evidence=(TaskCompletionEvidence("targeted", "targeted test passed", EvidenceStrength.DIRECT),), regression_required=True, regression_protection=regression))
    assert result.decision is CompletionDecision.COMPLETE
