from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import (
    AutomaticTestDecision,
    AutomaticTestExecution,
    AutomaticTestExecutionState,
    AutomaticTestResult,
    AutomaticTestStatus,
    BoundedSelfCorrectionLoop,
    CausalStatus,
    FailureClassification,
    FailureConfidence,
    FailureFinding,
    FailureLocation,
    FailureLocationKind,
    FailureSeverity,
    FixChangeType,
    FixConfidence,
    FixEvidence,
    FixLocation,
    FixPlan,
    FixRiskLevel,
    FixStatus,
    RootCauseAnalysis,
    RootCauseAnalysisStatus,
    RootCauseConfidence,
    RootCauseHypothesis,
    RootCauseLocation,
    RootCauseLocationKind,
    SelfCorrectionConfig,
    SelfCorrectionRequest,
    SelfCorrectionStatus,
)
from backend_ai.agent.automatic_testing import AutomaticTestRequest
from backend_ai.agent.execution_budget import BudgetDecision, BudgetDimension, BudgetExhaustion
from backend_ai.tools.safe_editing import SafeEditPolicy
from backend_ai.tools.test_result_parser import TestParseStatus


@dataclass
class FakeFix:
    status: FixStatus
    verified: bool
    tests_rerun: bool = False
    retries: int = 0


class FakeTests:
    def __init__(self, statuses: list[TestParseStatus]) -> None:
        self.statuses = statuses
        self.index = 0

    def run(self, request):
        status = self.statuses[min(self.index, len(self.statuses) - 1)]
        self.index += 1
        return AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.BUDGET_EXHAUSTED if status is TestParseStatus.UNKNOWN else AutomaticTestStatus.RUN, "fake structured result"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status="completed"))


class FakeParser:
    def __init__(self, statuses: list[TestParseStatus]) -> None:
        self.statuses = statuses
        self.index = 0

    def parse(self, result):
        status = self.statuses[min(self.index, len(self.statuses) - 1)]
        self.index += 1
        return SimpleNamespace(overall_status=status)


def _failure() -> object:
    finding = FailureFinding("finding-1", FailureClassification.ASSERTION_FAILURE, FailureSeverity.HIGH, FailureConfidence.HIGH, "assertion differs", FailureLocation(FailureLocationKind.TEST_LOCATION, "tests/test_value.py", 4, "test_value", None, FailureConfidence.HIGH), test_name="test_value", exception_type="AssertionError")
    return __import__("backend_ai.agent").agent.TestFailureAnalysis(__import__("backend_ai.agent").agent.FailureAnalysisStatus.ANALYZED, FailureClassification.ASSERTION_FAILURE, (finding,), (), finding.finding_id, (), False, True, "HIGH", "complete")


def _rca() -> RootCauseAnalysis:
    evidence = __import__("backend_ai.agent").agent.RootCauseEvidence("e1", __import__("backend_ai.agent").agent.RootCauseEvidenceType.LOCATION, "test", "exact evidence", RootCauseConfidence.HIGH, "fixture", ("finding-1",))
    hypothesis = RootCauseHypothesis("h1", "implementation differs from expectation", FailureClassification.ASSERTION_FAILURE, RootCauseLocation(RootCauseLocationKind.IMPLEMENTATION, "src/value.py", 2, "value", RootCauseConfidence.HIGH, True), ("code executes", "value differs", "assertion fails"), (evidence,), (), ("finding-1",), RootCauseConfidence.HIGH, RootCauseConfidence.HIGH, CausalStatus.PRIMARY_CANDIDATE, False, False)
    return RootCauseAnalysis(RootCauseAnalysisStatus.ANALYZED, "assertion differs", (hypothesis,), (), (), (evidence,), (), (), (), False, True, True, False)


def _plan() -> FixPlan:
    return FixPlan("src/value.py", FixLocation("src/value.py", 2, "value", "implementation", FixConfidence.HIGH, True), FixChangeType.SMALL_IMPLEMENTATION, "change value", "expected new state", FixRiskLevel.LOW, FixConfidence.HIGH, ("finding-1",), "h1", (FixEvidence("f1", "RCA", "supported", FixConfidence.HIGH, "RCA", ("finding-1",)),), "old", "new")


def _loop(statuses: list[TestParseStatus], *, max_attempts: int = 3, provider=True, fix_status=FixStatus.FIX_VERIFIED):
    failure = _failure()
    rca = _rca()
    fake_fix = lambda request: FakeFix(fix_status, fix_status is FixStatus.FIX_VERIFIED)
    return BoundedSelfCorrectionLoop(test_orchestrator=FakeTests(statuses), parser=FakeParser(statuses), failure_analyzer=SimpleNamespace(analyze=lambda request: failure), root_cause_analyzer=SimpleNamespace(analyze=lambda request: rca), fix_applier=fake_fix).run(SelfCorrectionRequest(test_request=AutomaticTestRequest(task='test self correction', project_root=Path('.')), fix_plan_provider=(lambda root, analysis, attempt: _plan()) if provider else None, fix_policy=SafeEditPolicy.for_modification(), config=SelfCorrectionConfig(max_attempts=max_attempts)))


def test_pass_stops_immediately_without_fix() -> None:
    result = _loop([TestParseStatus.PASS])
    assert result.status is SelfCorrectionStatus.PASSED
    assert len(result.attempts) == 1
    assert result.attempts[0].next_action == "COMPLETE"
    assert result.attempts[0].fix_status is None


def test_no_actionable_fix_stops_after_analysis() -> None:
    result = _loop([TestParseStatus.FAIL], provider=False)
    assert result.status is SelfCorrectionStatus.NO_ACTIONABLE_FIX
    assert result.attempts[0].next_action == "STOP_NO_ACTIONABLE_FIX"


def test_repeated_failure_stops_without_second_fix() -> None:
    result = _loop([TestParseStatus.FAIL, TestParseStatus.FAIL], max_attempts=3)
    assert result.status is SelfCorrectionStatus.REPEATED_FAILURE
    assert len(result.attempts) == 2
    assert result.attempts[0].fix_status == FixStatus.FIX_VERIFIED.value
    assert result.attempts[1].fix_status is None


def test_max_attempts_exhausts_before_final_fix() -> None:
    result = _loop([TestParseStatus.FAIL, TestParseStatus.FAIL, TestParseStatus.FAIL], max_attempts=2)
    assert result.status is SelfCorrectionStatus.REPEATED_FAILURE or result.status is SelfCorrectionStatus.EXHAUSTED
    assert result.attempts[-1].next_action in {"STOP_REPEATED_FAILURE", "STOP_MAX_ATTEMPTS"}


def test_policy_block_stops_without_retest() -> None:
    result = _loop([TestParseStatus.FAIL], fix_status=FixStatus.BLOCKED)
    assert result.status is SelfCorrectionStatus.BLOCKED
    assert len(result.attempts) == 1
    assert result.attempts[0].next_action == "STOP_BLOCKED"


def test_no_progress_stops_without_retry() -> None:
    result = _loop([TestParseStatus.FAIL], fix_status=FixStatus.FAILED)
    assert result.status is SelfCorrectionStatus.NO_PROGRESS
    assert result.attempts[0].next_action == "STOP_NO_PROGRESS"


def test_max_attempts_is_host_configured_and_bounded() -> None:
    assert SelfCorrectionConfig(max_attempts=1).max_attempts == 1
    try:
        SelfCorrectionConfig(max_attempts=65)
    except ValueError:
        pass
    else:
        raise AssertionError("max_attempts safety ceiling was not enforced")


def test_same_inputs_are_deterministic() -> None:
    first = _loop([TestParseStatus.FAIL, TestParseStatus.FAIL])
    second = _loop([TestParseStatus.FAIL, TestParseStatus.FAIL])
    assert [(item.next_action, item.failure_signature, item.action_signature) for item in first.attempts] == [(item.next_action, item.failure_signature, item.action_signature) for item in second.attempts]
