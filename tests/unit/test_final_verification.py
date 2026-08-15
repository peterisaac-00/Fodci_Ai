from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import (
    CompletionDecision,
    ExecutionBudget,
    ExecutionBudgetLedger,
    FinalCriterionStatus,
    FinalVerificationConfig,
    FinalVerificationRequest,
    FinalVerificationStatus,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    RecoveryResult,
    SelfCorrectionStatus,
    TaskCompletionEvidence,
    TaskCompletionRequest,
    VerificationEvidence,
    verify_final_state,
    verify_task_completion,
)
from backend_ai.agent.planner import ExecutionPlan
from backend_ai.agent.stop_conditions import StopConditionRequest, StopDecision, evaluate_stop_condition
from backend_ai.agent.recovery import RecoveryAction, RecoveryConfidence, RecoveryContext, RecoveryDecision, RecoverySeverity, ErrorCategory, RecoveryStatus
from backend_ai.agent.models import ToolResult


def _plan(kind=PlannerTaskType.FEATURE, *, complete=PlanCompleteness.COMPLETE, verification=("run tests",)):
    step = PlanStep("s1", "Implement bounded change", "Apply the requested change", "fixture", "verified result", (), PlanRiskLevel.MEDIUM)
    return ExecutionPlan("task", "task", "goal", kind, (step,), (), (), (), ("source change",), tuple(verification), PlannerConfidence.HIGH, (), complete)


def _budget(*, exhausted=()):
    return replace(ExecutionBudgetLedger(ExecutionBudget()).snapshot(), exhausted_dimensions=tuple(exhausted))


def _mutation(success=True, complete=True, unexpected=(), truncated=False):
    return SimpleNamespace(success=success, complete=complete, unexpected_changes=tuple(unexpected), truncated=truncated)


def _regression(status):
    return SimpleNamespace(status=SimpleNamespace(value=status), comparison=SimpleNamespace(message=status))


def _full_request(**overrides):
    values = dict(
        task="implement bounded change",
        plan=_plan(),
        completed_step_ids=("s1",),
        tool_results=(ToolResult("c1", "edit_file", True, data={"changed": True}),),
        verification=VerificationEvidence.passed("parse_test_result", "targeted tests passed"),
        mutation_verification=_mutation(),
        budget=_budget(),
    )
    values.update(overrides)
    return FinalVerificationRequest(**values)


def test_fully_verified_implementation() -> None:
    result = verify_final_state(_full_request())
    assert result.status is FinalVerificationStatus.VERIFIED
    assert result.verified is True
    assert result.final_message == "Task completed and verified."


def test_fully_verified_bug_fix_with_regression() -> None:
    result = verify_final_state(_full_request(plan=_plan(PlannerTaskType.BUG_FIX), regression_required=True, regression_protection=_regression("REGRESSION_FREE")))
    assert result.status is FinalVerificationStatus.VERIFIED


def test_targeted_pass_but_regression_failure_is_not_verified() -> None:
    result = verify_final_state(_full_request(regression_required=True, regression_protection=_regression("REGRESSION_DETECTED")))
    assert result.status is FinalVerificationStatus.NOT_VERIFIED
    assert result.verified is False


def test_targeted_pass_but_regression_missing_is_incomplete() -> None:
    result = verify_final_state(_full_request(regression_required=True))
    assert result.status is FinalVerificationStatus.INCOMPLETE
    assert "regression" in result.missing_evidence


def test_targeted_pass_but_baseline_missing_is_insufficient() -> None:
    result = verify_final_state(_full_request(regression_required=True, regression_protection=_regression("INSUFFICIENT_EVIDENCE")))
    assert result.status is FinalVerificationStatus.INCOMPLETE


def test_mutation_verified_but_tests_missing_is_insufficient() -> None:
    result = verify_final_state(_full_request(verification=VerificationEvidence.not_required()))
    assert result.status in {FinalVerificationStatus.INSUFFICIENT_EVIDENCE, FinalVerificationStatus.INCOMPLETE}


def test_tests_pass_but_plan_incomplete_is_incomplete() -> None:
    result = verify_final_state(_full_request(plan=_plan(complete=PlanCompleteness.PARTIAL)))
    assert result.status is FinalVerificationStatus.INCOMPLETE


def test_unresolved_recovery_is_incomplete() -> None:
    decision = RecoveryDecision(RecoveryStatus.CONTINUE, SimpleNamespace(category=ErrorCategory.TEST_FAILURE, recoverable=True, severity=RecoverySeverity.MEDIUM, another_action_allowed=True, user_intervention_required=False, terminate_task=False, safety_or_policy_boundary=False, code="TEST_FAILURE", message="retry"), RecoveryAction.VERIFY, "retry", RecoveryConfidence.MEDIUM, "run_tests", "test", True, False, True, True, "within budget")
    result = verify_final_state(_full_request(recovery=RecoveryResult(decision), tool_results=()))
    assert result.status is FinalVerificationStatus.INCOMPLETE


def test_policy_denial_is_blocked() -> None:
    result = verify_final_state(_full_request(policy_denied=True))
    assert result.status is FinalVerificationStatus.BLOCKED


def test_safety_block_is_blocked() -> None:
    result = verify_final_state(_full_request(safety_blocked=True))
    assert result.status is FinalVerificationStatus.BLOCKED


def test_budget_exhaustion_prevents_verified() -> None:
    result = verify_final_state(_full_request(budget=_budget(exhausted=("tool_calls",))))
    assert result.status is FinalVerificationStatus.BUDGET_EXHAUSTED


def test_timeout_or_truncated_evidence_is_not_verified() -> None:
    result = verify_final_state(_full_request(verification=VerificationEvidence.pending("parse_test_result", "timeout")))
    assert result.status is FinalVerificationStatus.INCOMPLETE
    assert "verification" in result.truncated_evidence


def test_output_or_incomplete_evidence_is_not_verified() -> None:
    result = verify_final_state(_full_request(evidence_complete=False))
    assert result.status is FinalVerificationStatus.INSUFFICIENT_EVIDENCE


def test_unexpected_modifications_are_negative_evidence() -> None:
    result = verify_final_state(_full_request(unexpected_modifications=("unrelated.py",)))
    assert result.status is FinalVerificationStatus.NOT_VERIFIED


def test_conflicting_model_claim_is_not_authoritative() -> None:
    result = verify_final_state(_full_request(final_action_claim="Everything is fixed", regression_required=True, regression_protection=_regression("REGRESSION_DETECTED")))
    assert result.status is FinalVerificationStatus.NOT_VERIFIED
    assert result.conflicting_evidence


def test_action_final_without_evidence_is_insufficient() -> None:
    result = verify_final_state(FinalVerificationRequest("task", final_action_claim="done"))
    assert result.status is FinalVerificationStatus.INSUFFICIENT_EVIDENCE


def test_documentation_task_does_not_force_tests() -> None:
    result = verify_final_state(FinalVerificationRequest("docs", _plan(PlannerTaskType.DOCUMENTATION_CHANGE, verification=()), completed_step_ids=("s1",), tool_results=(ToolResult("c1", "edit_file", True, data={"changed": True}),), mutation_verification=_mutation(), verification=VerificationEvidence.not_required(), budget=_budget())
)
    assert result.status is FinalVerificationStatus.VERIFIED
    assert next(item for item in result.criteria if item.criterion_id == "tests").status is FinalCriterionStatus.NOT_REQUIRED


def test_investigation_task_can_verify_structured_observation_without_tests() -> None:
    result = verify_final_state(FinalVerificationRequest("investigate", _plan(PlannerTaskType.INVESTIGATION, verification=()), completed_step_ids=("s1",), tool_results=(ToolResult("c1", "read_file", True, data={"content": "bounded"}),), verification=VerificationEvidence.not_required(), budget=_budget()))
    assert result.status is FinalVerificationStatus.VERIFIED


def test_pre_existing_failures_require_explicit_acceptance() -> None:
    denied = verify_final_state(_full_request(regression_required=True, regression_protection=_regression("PRE_EXISTING_FAILURES_ONLY")))
    accepted = verify_final_state(_full_request(regression_required=True, regression_protection=_regression("PRE_EXISTING_FAILURES_ONLY"), config=FinalVerificationConfig(accept_pre_existing_failures=True)))
    assert denied.status is FinalVerificationStatus.INCOMPLETE
    assert accepted.status is FinalVerificationStatus.VERIFIED


def test_self_correction_exhaustion_blocks_fix_chain_when_required() -> None:
    exhausted = SimpleNamespace(status=SelfCorrectionStatus.EXHAUSTED)
    result = verify_final_state(_full_request(self_correction=exhausted, config=FinalVerificationConfig(require_fix_chain=True)))
    assert result.status is FinalVerificationStatus.NOT_VERIFIED


def test_self_correction_chain_requires_structured_fix_evidence() -> None:
    passed = SimpleNamespace(status=SelfCorrectionStatus.PASSED, final_failure_analysis=None, final_root_cause=None, final_fix_result=None, final_parsed_result=None)
    result = verify_final_state(_full_request(self_correction=passed, config=FinalVerificationConfig(require_fix_chain=True)))
    assert result.status is FinalVerificationStatus.INCOMPLETE


def test_completion_verifier_requires_final_verification() -> None:
    incomplete = verify_final_state(_full_request(regression_required=True))
    request = TaskCompletionRequest("task", criteria=(), evidence=(TaskCompletionEvidence("tool", "targeted passed", strength=__import__("backend_ai.agent").agent.EvidenceStrength.DIRECT),), final_verification_required=True, final_verification=incomplete)
    result = verify_task_completion(request)
    assert result.decision is not CompletionDecision.COMPLETE


def test_completion_verifier_accepts_verified_final_result() -> None:
    verified = verify_final_state(_full_request())
    request = TaskCompletionRequest("task", evidence=(TaskCompletionEvidence("tool", "structured observation", strength=__import__("backend_ai.agent").agent.EvidenceStrength.DIRECT),), final_verification_required=True, final_verification=verified)
    result = verify_task_completion(request)
    assert result.decision is CompletionDecision.COMPLETE


def test_stop_condition_remains_authoritative_after_completion() -> None:
    blocked = evaluate_stop_condition(StopConditionRequest(final_action_valid=True, completion_decision="INCOMPLETE", completion_evidence=("final verification incomplete",)))
    done = evaluate_stop_condition(StopConditionRequest(final_action_valid=True, completion_decision="COMPLETE", completion_evidence=("verified",)))
    assert blocked.decision is StopDecision.CONTINUE
    assert done.decision is StopDecision.DONE


def test_repeated_verification_is_deterministic() -> None:
    request = _full_request(final_action_claim="done")
    first = verify_final_state(request).to_dict()
    second = verify_final_state(request).to_dict()
    assert first == second


def test_no_direct_filesystem_or_execution_side_effects() -> None:
    before = Path.cwd()
    verify_final_state(_full_request())
    assert Path.cwd() == before


def test_mutation_failure_is_not_verified() -> None:
    result = verify_final_state(_full_request(mutation_verification=_mutation(success=False)))
    assert result.status is FinalVerificationStatus.NOT_VERIFIED


def test_mutation_truncation_is_incomplete() -> None:
    result = verify_final_state(_full_request(mutation_verification=_mutation(complete=False, truncated=True)))
    assert result.status is FinalVerificationStatus.INCOMPLETE
    assert "mutation_verification" in result.truncated_evidence


def test_explicit_required_tests_without_plan_need_test_evidence() -> None:
    result = verify_final_state(FinalVerificationRequest("run checks", verification=VerificationEvidence.not_required(), config=FinalVerificationConfig(require_tests=True)))
    assert result.status is FinalVerificationStatus.INSUFFICIENT_EVIDENCE


def test_explicit_required_mutation_without_plan_needs_mutation_evidence() -> None:
    result = verify_final_state(FinalVerificationRequest("change file", config=FinalVerificationConfig(require_mutation_verification=True)))
    assert result.status is FinalVerificationStatus.INSUFFICIENT_EVIDENCE


def test_regression_blocked_is_blocked() -> None:
    result = verify_final_state(_full_request(regression_required=True, regression_protection=_regression("VERIFICATION_BLOCKED")))
    assert result.status is FinalVerificationStatus.BLOCKED


def test_regression_incomplete_is_incomplete() -> None:
    result = verify_final_state(_full_request(regression_required=True, regression_protection=_regression("VERIFICATION_INCOMPLETE")))
    assert result.status is FinalVerificationStatus.INCOMPLETE


def test_skipped_required_plan_step_is_incomplete() -> None:
    result = verify_final_state(_full_request(completed_step_ids=(), skipped_step_ids=("s1",)))
    assert result.status is FinalVerificationStatus.INCOMPLETE


def test_blocked_plan_step_is_blocked() -> None:
    result = verify_final_state(_full_request(blocked_step_ids=("s1",)))
    assert result.status is FinalVerificationStatus.BLOCKED


def test_missing_capability_is_blocked() -> None:
    result = verify_final_state(_full_request(missing_capabilities=("run_tests",)))
    assert result.status is FinalVerificationStatus.BLOCKED


def test_user_intervention_recovery_is_blocked() -> None:
    decision = RecoveryDecision(RecoveryStatus.USER_INTERVENTION_REQUIRED, SimpleNamespace(category=ErrorCategory.CONCURRENT_MODIFICATION, recoverable=False, severity=RecoverySeverity.HIGH, another_action_allowed=False, user_intervention_required=True, terminate_task=True, safety_or_policy_boundary=False, code="CONCURRENT_MODIFICATION", message="user changed file"), RecoveryAction.USER_INTERVENTION_REQUIRED, "user change", RecoveryConfidence.HIGH, "edit_file", "edit", False, False, False, True, "within budget")
    result = verify_final_state(_full_request(recovery=RecoveryResult(decision)))
    assert result.status is FinalVerificationStatus.BLOCKED


def test_output_truncated_tool_evidence_is_insufficient() -> None:
    result = verify_final_state(_full_request(tool_results=(ToolResult("c1", "run_tests", True, data={"status": "PASS"}, truncated=True),)))
    assert result.status is FinalVerificationStatus.INSUFFICIENT_EVIDENCE


def test_required_authority_results_are_checked_without_replacing_them() -> None:
    completion = SimpleNamespace(complete=False, decision=SimpleNamespace(value="CONTINUE"))
    stop = SimpleNamespace(decision=SimpleNamespace(value="CONTINUE"))
    result = verify_final_state(_full_request(task_completion=completion, stop_evaluation=stop, config=FinalVerificationConfig(require_completion_authority=True, require_stop_authority=True)))
    assert result.status is FinalVerificationStatus.NOT_VERIFIED


def test_missing_required_authority_is_insufficient() -> None:
    result = verify_final_state(_full_request(config=FinalVerificationConfig(require_completion_authority=True, require_stop_authority=True)))
    assert result.status is FinalVerificationStatus.INSUFFICIENT_EVIDENCE


def test_self_correction_successful_chain_can_verify() -> None:
    correction = SimpleNamespace(status=SimpleNamespace(value="PASSED"), final_failure_analysis=object(), final_root_cause=object(), final_fix_result=SimpleNamespace(verified=True), final_parsed_result=SimpleNamespace(overall_status=SimpleNamespace(value="PASS")))
    result = verify_final_state(_full_request(self_correction=correction, config=FinalVerificationConfig(require_fix_chain=True)))
    assert result.status is FinalVerificationStatus.VERIFIED


def test_internal_execution_error_is_failed() -> None:
    result = verify_final_state(_full_request(tool_results=(ToolResult("c1", "run_tests", False, error_code="INTERNAL_ERROR", message="runner failed"),)))
    assert result.status is FinalVerificationStatus.FAILED
