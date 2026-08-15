from __future__ import annotations

from pathlib import Path

from backend_ai.agent import (
    AutomaticFixConfig,
    AutomaticFixRequest,
    AutomaticFixOrchestrator,
    FixChangeType,
    FixConfidence,
    FixEvidence,
    FixFailureReason,
    FixLocation,
    FixPlan,
    FixRiskLevel,
    FixStatus,
    RootCauseAnalysisStatus,
    RootCauseAnalysis,
    RootCauseConfidence,
    RootCauseHypothesis,
    RootCauseLocation,
    RootCauseLocationKind,
    FailureClassification,
    CausalStatus,
)
from backend_ai.agent.execution_budget import ExecutionBudget, ExecutionBudgetLedger
from backend_ai.tools.safe_editing import SafeEditPolicy


def _rca(path: str = "src/service.py", *, confidence: RootCauseConfidence = RootCauseConfidence.HIGH, status: CausalStatus = CausalStatus.PRIMARY_CANDIDATE) -> RootCauseAnalysis:
    evidence = __import__("backend_ai.agent").agent.RootCauseEvidence("r1", __import__("backend_ai.agent").agent.RootCauseEvidenceType.LOCATION, "parser", "exact implementation location", RootCauseConfidence.HIGH, "structured RCA", ("f1",))
    hypothesis = RootCauseHypothesis("h1", "The implementation returns the wrong value.", FailureClassification.ASSERTION_FAILURE, RootCauseLocation(RootCauseLocationKind.IMPLEMENTATION, path, 1, "value", confidence, True), ("implementation executes", "wrong value is returned", "assertion fails"), (evidence,), (), ("f1",), confidence, confidence, status, False, False)
    return RootCauseAnalysis(RootCauseAnalysisStatus.ANALYZED, "expected 2 but received 1", (hypothesis,), (), (), (evidence,), (), (), (), False, True, True, False)


def _plan(path: str = "src/service.py", *, confidence: FixConfidence = FixConfidence.HIGH, risk: FixRiskLevel = FixRiskLevel.LOW, old: str = "return 1\n", new: str = "return 2\n") -> FixPlan:
    evidence = (FixEvidence("f1", "RCA", "exact location and structured evidence", FixConfidence.HIGH, "RootCauseAnalysis", ("f1",)),)
    return FixPlan(path, FixLocation(path, 1, "value", "implementation", confidence, True), FixChangeType.SMALL_IMPLEMENTATION, "Replace the returned constant with the evidence-backed expected value.", "file contains the requested new UTF-8 content", risk, confidence, ("f1",), "h1", evidence, old, new)


def _request(root: Path, plan: FixPlan, *, rca: RootCauseAnalysis | None = None, ledger: ExecutionBudgetLedger | None = None, config: AutomaticFixConfig | None = None) -> AutomaticFixRequest:
    policy = SafeEditPolicy.for_modification(backup_enabled=True)
    return AutomaticFixRequest(str(root), rca or _rca(plan.target_file), plan, policy, ledger, config or AutomaticFixConfig())


def test_valid_fix_uses_transaction_and_verifies_post_state(tmp_path: Path) -> None:
    target = tmp_path / "src" / "service.py"
    target.parent.mkdir()
    target.write_text("return 1\n", encoding="utf-8")
    result = AutomaticFixOrchestrator().apply(_request(tmp_path, _plan()))
    assert result.status is FixStatus.FIX_VERIFIED
    assert result.attempted is True
    assert result.verified is True
    assert result.tests_rerun is False
    assert result.retries == 0
    assert target.read_text(encoding="utf-8") == "return 2\n"
    assert result.transaction is not None and result.transaction.status == "committed"


def test_low_confidence_is_rejected_without_mutation(tmp_path: Path) -> None:
    target = tmp_path / "src" / "service.py"
    target.parent.mkdir()
    target.write_text("return 1\n", encoding="utf-8")
    result = AutomaticFixOrchestrator().apply(_request(tmp_path, _plan(confidence=FixConfidence.LOW), rca=_rca(confidence=RootCauseConfidence.LOW)))
    assert result.status is FixStatus.NO_SAFE_FIX
    assert result.decision.reason is FixFailureReason.LOW_CONFIDENCE
    assert result.attempted is False
    assert target.read_text(encoding="utf-8") == "return 1\n"


def test_missing_or_non_actionable_rca_is_rejected(tmp_path: Path) -> None:
    result = AutomaticFixOrchestrator().apply(_request(tmp_path, _plan(), rca=None))
    # _request supplies a valid default RCA when rca is None, so use explicit missing analysis.
    result = AutomaticFixOrchestrator().apply(AutomaticFixRequest(str(tmp_path), None, _plan(), SafeEditPolicy.for_modification()))
    assert result.status is FixStatus.NO_SAFE_FIX
    assert result.decision.reason is FixFailureReason.INSUFFICIENT_EVIDENCE


def test_unsafe_sensitive_and_ambiguous_paths_are_rejected(tmp_path: Path) -> None:
    for path, reason in (("../secret.py", FixFailureReason.UNSAFE_PATH), (".env", FixFailureReason.SENSITIVE_PATH)):
        plan = _plan(path)
        result = AutomaticFixOrchestrator().apply(_request(tmp_path, plan))
        assert result.decision.reason is reason
        assert result.attempted is False
    ambiguous = _plan()
    ambiguous = FixPlan(ambiguous.target_file, FixLocation("other.py", 1, "value", "implementation", FixConfidence.HIGH, True), ambiguous.change_type, ambiguous.intended_change, ambiguous.expected_post_state, ambiguous.risk, ambiguous.confidence, ambiguous.affected_failure_ids, ambiguous.hypothesis_id, ambiguous.evidence, ambiguous.old_content, ambiguous.new_content)
    assert AutomaticFixOrchestrator().apply(_request(tmp_path, ambiguous)).decision.reason is FixFailureReason.AMBIGUOUS_LOCATION


def test_malformed_plan_and_missing_target_are_safe(tmp_path: Path) -> None:
    result = AutomaticFixOrchestrator().apply(AutomaticFixRequest(str(tmp_path), _rca(), None, SafeEditPolicy.for_modification()))
    assert result.status is FixStatus.REJECTED and result.attempted is False
    result = AutomaticFixOrchestrator().apply(_request(tmp_path, _plan()))
    assert result.status is FixStatus.FAILED
    assert result.attempted is True
    assert result.verified is False


def test_policy_denial_blocks_before_mutation(tmp_path: Path) -> None:
    target = tmp_path / "src" / "service.py"
    target.parent.mkdir()
    target.write_text("return 1\n", encoding="utf-8")
    result = AutomaticFixOrchestrator().apply(AutomaticFixRequest(str(tmp_path), _rca(), _plan(), SafeEditPolicy()))
    assert result.status is FixStatus.REJECTED
    assert result.decision.reason is FixFailureReason.POLICY_DENIAL
    assert result.attempted is False
    assert target.read_text(encoding="utf-8") == "return 1\n"


def test_budget_denial_occurs_before_mutation(tmp_path: Path) -> None:
    target = tmp_path / "src" / "service.py"
    target.parent.mkdir()
    target.write_text("return 1\n", encoding="utf-8")
    ledger = ExecutionBudgetLedger(ExecutionBudget(max_action_steps=0, max_mutations=1))
    result = AutomaticFixOrchestrator().apply(_request(tmp_path, _plan(), ledger=ledger))
    assert result.status is FixStatus.BLOCKED
    assert result.decision.reason is FixFailureReason.BUDGET_EXHAUSTION
    assert result.attempted is False
    assert result.budget_decision is not None and result.budget_decision.operation_started is False
    assert target.read_text(encoding="utf-8") == "return 1\n"


def test_one_attempt_has_no_retry_or_test_rerun(tmp_path: Path) -> None:
    target = tmp_path / "src" / "service.py"
    target.parent.mkdir()
    target.write_text("return 1\n", encoding="utf-8")
    result = AutomaticFixOrchestrator().apply(_request(tmp_path, _plan()))
    assert result.retries == 0
    assert result.tests_rerun is False
    assert result.attempted is True


def test_serialization_does_not_include_fix_contents(tmp_path: Path) -> None:
    result = AutomaticFixOrchestrator().apply(_request(tmp_path, _plan()))
    rendered = str(result.to_dict())
    assert "has_old_content" in rendered
    assert "return 1\\n" not in rendered
    assert "return 2\\n" not in rendered
