from pathlib import Path

from backend_ai.agent import (
    AutomaticFixConfig,
    AutomaticTestOrchestrator,
    AutomaticTestRequest,
    BoundedSelfCorrectionLoop,
    ExecutionPlan,
    FinalVerificationConfig,
    FinalVerificationRequest,
    FinalVerificationStatus,
    FixChangeType,
    FixConfidence,
    FixEvidence,
    FixLocation,
    FixPlan,
    FixRiskLevel,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    RegressionBaseline,
    RegressionStatus,
    RegressionTestScope,
    RootCauseLocationKind,
    SelfCorrectionConfig,
    SelfCorrectionRequest,
    SelfCorrectionStatus,
    ToolRegistry,
    VerificationEvidence,
    verify_final_state,
)
from backend_ai.tools.safe_editing import SafeEditPolicy
from backend_ai.tools.test_result_parser import TestResultParser, TestParseStatus


def _bug_plan() -> ExecutionPlan:
    step = PlanStep("fix", "Fix and verify bug", "Apply and verify the bug fix", "real checkpoint", "target and regression evidence", (), PlanRiskLevel.MEDIUM)
    return ExecutionPlan("fix bug", "fix bug", "fix bug", PlannerTaskType.BUG_FIX, (step,), (), (), (), ("source mutation",), ("targeted tests", "regression verification"), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE)


def _write_project(root: Path, *, negative: bool = False) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n", encoding="utf-8")
    (root / "src" / "value.py").write_text("def add(a, b):\n    return a - b\n\ndef subtract(a, b):\n    return a - b\n", encoding="utf-8")
    (root / "tests" / "test_add.py").write_text("from src.value import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")
    (root / "tests" / "test_subtract.py").write_text("from src.value import subtract\n\ndef test_subtract():\n    assert subtract(5, 3) == 2\n", encoding="utf-8")


def _requests(root: Path):
    registry = ToolRegistry.with_test_execution()
    target = AutomaticTestRequest("fix bug", root, registry=registry, user_requested=True, test_target="tests/test_add.py")
    suite = AutomaticTestRequest("regression suite", root, registry=registry, user_requested=True)
    return target, suite


def _baseline(root: Path):
    target, _ = _requests(root)
    result = AutomaticTestOrchestrator().run(target)
    assert result.test_run_result is not None
    parsed = TestResultParser().parse(result.test_run_result)
    assert parsed.overall_status is TestParseStatus.FAIL
    return RegressionBaseline.capture(result, parsed), target


def _provider(*, negative: bool):
    def provide(rca, analysis, attempt):
        old = "def add(a, b):\n    return a - b\n\ndef subtract(a, b):\n    return a - b\n"
        new = "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a + b\n" if negative else "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n"
        hypothesis = rca.hypotheses[0]
        return FixPlan("src/value.py", FixLocation("src/value.py", 2, "add", "implementation", FixConfidence.HIGH, True), FixChangeType.SMALL_IMPLEMENTATION, "fix add implementation", "make add satisfy the failing assertion", FixRiskLevel.LOW, FixConfidence.HIGH, tuple(finding.finding_id for finding in analysis.findings[:8]), hypothesis.hypothesis_id, (FixEvidence("real-r1", "RCA evidence", "the implementation returned the wrong arithmetic operation", FixConfidence.HIGH, "RCA", tuple(finding.finding_id for finding in analysis.findings[:8])),), old, new)
    return provide


def _run_real_checkpoint(root: Path, *, negative: bool):
    baseline, target = _baseline(root)
    _, suite = _requests(root)
    request = SelfCorrectionRequest(
        test_request=target,
        fix_plan_provider=_provider(negative=negative),
        fix_policy=SafeEditPolicy.for_modification(),
        fix_config=AutomaticFixConfig(min_confidence=FixConfidence.MEDIUM),
        config=SelfCorrectionConfig(max_attempts=3, require_regression_protection=True),
        regression_baseline=baseline,
        regression_test_request=suite,
        regression_scope=RegressionTestScope.PROJECT_SUITE,
    )
    result = BoundedSelfCorrectionLoop().run(request)
    return result, baseline


def test_real_positive_checkpoint_reaches_verified(tmp_path: Path) -> None:
    _write_project(tmp_path)
    result, _ = _run_real_checkpoint(tmp_path, negative=False)
    assert result.status is SelfCorrectionStatus.REGRESSION_FREE
    assert result.final_fix_result is not None and result.final_fix_result.verified
    assert result.regression_protection is not None
    assert result.regression_protection.status is RegressionStatus.REGRESSION_FREE
    assert result.final_parsed_result is not None and result.final_parsed_result.overall_status is TestParseStatus.PASS
    final = verify_final_state(FinalVerificationRequest(
        "fix bug",
        _bug_plan(),
        completed_step_ids=("fix",),
        tool_results=(),
        verification=VerificationEvidence.passed("parse_test_result", "targeted test passed"),
        mutation_verification=result.final_fix_result.transaction.verification,
        regression_protection=result.regression_protection,
        regression_required=True,
        self_correction=result,
        budget=result.execution_budget,
        config=FinalVerificationConfig(require_fix_chain=True),
    ))
    assert final.status is FinalVerificationStatus.VERIFIED


def test_real_negative_checkpoint_never_verifies(tmp_path: Path) -> None:
    _write_project(tmp_path, negative=True)
    result, _ = _run_real_checkpoint(tmp_path, negative=True)
    assert result.status is SelfCorrectionStatus.REGRESSION_DETECTED
    assert result.regression_protection is not None
    assert result.regression_protection.status is RegressionStatus.REGRESSION_DETECTED
    final = verify_final_state(FinalVerificationRequest(
        "fix bug",
        _bug_plan(),
        completed_step_ids=("fix",),
        verification=VerificationEvidence.passed("parse_test_result", "targeted test passed"),
        mutation_verification=result.final_fix_result.transaction.verification,
        regression_protection=result.regression_protection,
        regression_required=True,
        self_correction=result,
        budget=result.execution_budget,
        config=FinalVerificationConfig(require_fix_chain=True),
    ))
    assert final.status is FinalVerificationStatus.NOT_VERIFIED
    assert final.verified is False
