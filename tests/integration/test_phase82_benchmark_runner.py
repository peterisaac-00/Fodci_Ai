from __future__ import annotations

from pathlib import Path

from backend_ai.agent import (
    AutomaticFixConfig,
    AutomaticTestOrchestrator,
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
    SelfCorrectionConfig,
    SelfCorrectionStatus,
    ToolRegistry,
    VerificationEvidence,
    verify_final_state,
)
from backend_ai.evaluation import (
    AllowedScope,
    BenchmarkConfig,
    BenchmarkExecutionResult,
    BenchmarkRequest,
    BenchmarkRunner,
    BenchmarkStatus,
    BenchmarkTaskStatus,
    EvaluationConstraint,
    EvaluationDifficulty,
    EvaluationTask,
    EvaluationTaskCategory,
    EvaluationTestType,
    ExpectedArea,
    ExpectedAreaType,
    ExpectedBehavior,
    GroundTruth,
    ProjectDefinition,
    Requirement,
    SuccessCriterion,
    SuccessCriterionType,
    TestDefinition,
)
from backend_ai.tools.safe_editing import SafeEditPolicy
from backend_ai.tools.test_result_parser import TestParseStatus, TestResultParser


def _task() -> EvaluationTask:
    return EvaluationTask(
        task_id="EVAL-REAL-ADD",
        title="Fix broken addition",
        description="Fix the broken add implementation without changing subtraction.",
        version="1.0",
        category=EvaluationTaskCategory.BUG_FIX,
        difficulty=EvaluationDifficulty.EASY,
        project_definition=ProjectDefinition(project_type="backend", language="Python", runtime="Python 3.12", test_framework="pytest"),
        user_intent="Make add(2, 3) return 5.",
        requirements=(Requirement("REQ-ADD", "add returns the sum"),),
        expected_behaviors=(ExpectedBehavior("BEH-ADD", "add(2, 3)", "call add", "5", "5"),),
        allowed_scope=AllowedScope(allowed_files=("src/value.py", "tests/test_*.py"), allowed_directories=("src/", "tests/"), allowed_change_types=("EDIT",)),
        expected_areas=(ExpectedArea("addition implementation", ("src/value.py",), ExpectedAreaType.REQUIRED_CHANGE),),
        tests=(TestDefinition("TEST-ADD", "add test", EvaluationTestType.UNIT, "tests/test_add.py", True, "PASS", ("REQ-ADD",), ("BEH-ADD",)),),
        success_criteria=(SuccessCriterion("CRIT-ADD", "target test and regression pass", SuccessCriterionType.REGRESSION_FREE, True, "parsed target and regression evidence", test_ids=("TEST-ADD",), behavior_ids=("BEH-ADD",)),),
        ground_truth=GroundTruth(expected_behavior_ids=("BEH-ADD",), required_outcomes=("add(2, 3) equals 5",), allowed_implementation_alternatives=("any safe arithmetic implementation",)),
        constraints=EvaluationConstraint(max_files_expected=2, required_language="Python", required_test_framework="pytest"),
    )


def _fixture(_task: EvaluationTask, root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n", encoding="utf-8")
    (root / "src" / "value.py").write_text("def add(a, b):\n    return a - b\n\ndef subtract(a, b):\n    return a - b\n", encoding="utf-8")
    (root / "tests" / "test_add.py").write_text("from src.value import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")
    (root / "tests" / "test_subtract.py").write_text("from src.value import subtract\n\ndef test_subtract():\n    assert subtract(5, 3) == 2\n", encoding="utf-8")


def _plan() -> ExecutionPlan:
    step = PlanStep("fix", "Fix and verify", "Fix the broken addition", "real benchmark", "target and regression evidence", (), PlanRiskLevel.MEDIUM)
    return ExecutionPlan("fix addition", "fix addition", "fix addition", PlannerTaskType.BUG_FIX, (step,), (), (), (), ("src/value.py",), ("target test", "regression"), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE)


# Keep the adapter in the integration test: BenchmarkRunner itself remains a generic
# orchestration layer and does not contain a task-specific fix or execution engine.
class ExistingFodciRuntime:
    def __init__(self, *, introduce_regression: bool) -> None:
        self.introduce_regression = introduce_regression

    def execute(self, task: EvaluationTask, workspace_root: Path, *, max_wall_time: float) -> BenchmarkExecutionResult:
        registry = ToolRegistry.with_test_execution()
        target_request = __import__("backend_ai.agent", fromlist=["AutomaticTestRequest"]).AutomaticTestRequest("fix addition", workspace_root, registry=registry, user_requested=True, test_target="tests/test_add.py")
        suite_request = __import__("backend_ai.agent", fromlist=["AutomaticTestRequest"]).AutomaticTestRequest("regression suite", workspace_root, registry=registry, user_requested=True)
        initial = AutomaticTestOrchestrator().run(target_request)
        assert initial.test_run_result is not None
        parsed = TestResultParser().parse(initial.test_run_result)
        baseline = RegressionBaseline.capture(initial, parsed)

        def provide(rca, analysis, attempt):
            old = "def add(a, b):\n    return a - b\n\ndef subtract(a, b):\n    return a - b\n"
            new = "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a + b\n" if self.introduce_regression else "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n"
            hypothesis = rca.hypotheses[0]
            return FixPlan("src/value.py", FixLocation("src/value.py", 2, "add", "implementation", FixConfidence.HIGH, True), FixChangeType.SMALL_IMPLEMENTATION, "fix addition", "make add satisfy the failing assertion", FixRiskLevel.LOW, FixConfidence.HIGH, tuple(f.finding_id for f in analysis.findings[:8]), hypothesis.hypothesis_id, (FixEvidence("bench-r1", "RCA evidence", "wrong arithmetic operation", FixConfidence.HIGH, "RCA", tuple(f.finding_id for f in analysis.findings[:8])),), old, new)

        correction = BoundedSelfCorrectionLoop().run(__import__("backend_ai.agent", fromlist=["SelfCorrectionRequest"]).SelfCorrectionRequest(test_request=target_request, fix_plan_provider=provide, fix_policy=SafeEditPolicy.for_modification(), fix_config=AutomaticFixConfig(min_confidence=FixConfidence.MEDIUM), config=SelfCorrectionConfig(max_attempts=3, require_regression_protection=True), regression_baseline=baseline, regression_test_request=suite_request, regression_scope=RegressionTestScope.PROJECT_SUITE))
        regression = correction.regression_protection
        final_status = RegressionStatus.REGRESSION_FREE if regression and regression.status is RegressionStatus.REGRESSION_FREE else RegressionStatus.REGRESSION_DETECTED
        verification = verify_final_state(FinalVerificationRequest("fix addition", _plan(), completed_step_ids=("fix",), verification=VerificationEvidence.passed("parse_test_result", "targeted test passed") if correction.final_parsed_result and correction.final_parsed_result.overall_status is TestParseStatus.PASS else VerificationEvidence.missing("targeted test"), mutation_verification=correction.final_fix_result.transaction.verification if correction.final_fix_result else None, regression_protection=regression, regression_required=True, self_correction=correction, budget=correction.execution_budget, config=FinalVerificationConfig(require_fix_chain=True)))
        passed = correction.status is SelfCorrectionStatus.REGRESSION_FREE and verification.status is FinalVerificationStatus.VERIFIED
        return BenchmarkExecutionResult(BenchmarkTaskStatus.PASSED if passed else BenchmarkTaskStatus.FAILED, execution_status=correction.status.value, termination_reason=correction.status.value, test_evidence=correction.final_parsed_result.to_dict() if correction.final_parsed_result else None, mutation_evidence=correction.final_fix_result.transaction.verification.to_dict() if correction.final_fix_result else None, completion_evidence={"self_correction_status": correction.status.value}, final_verification_evidence=verification.to_dict(), stop_condition_evidence={"final_verification": verification.status.value}, failure_information=() if passed else (correction.status.value,), recovery_state=None, budget_state=correction.execution_budget.to_dict(), tests_requested=True, tests_executed=True)


def test_real_successful_benchmark_task(tmp_path: Path) -> None:
    result = BenchmarkRunner().run(BenchmarkRequest((_task(),), config=BenchmarkConfig(), runtime=ExistingFodciRuntime(introduce_regression=False), fixture_provider=_fixture))
    assert result.status is BenchmarkStatus.COMPLETED
    assert result.task_runs[0].status is BenchmarkTaskStatus.PASSED
    assert result.task_runs[0].evidence.final_verification_evidence["status"] == "VERIFIED"
    assert result.task_runs[0].evidence.changed_paths == ("src/value.py",)


def test_real_negative_benchmark_task_records_regression_failure(tmp_path: Path) -> None:
    result = BenchmarkRunner().run(BenchmarkRequest((_task(),), config=BenchmarkConfig(), runtime=ExistingFodciRuntime(introduce_regression=True), fixture_provider=_fixture))
    assert result.status is BenchmarkStatus.FAILED
    assert result.task_runs[0].status is BenchmarkTaskStatus.FAILED
    assert result.task_runs[0].evidence.final_verification_evidence["status"] == "NOT_VERIFIED"
    assert result.task_runs[0].evidence.failure_information
