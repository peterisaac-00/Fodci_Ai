from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from backend_ai.evaluation.acceptance import (
    AcceptanceDecision,
    AcceptancePolicy,
    AcceptanceRequest,
    AcceptanceStore,
    ModelAcceptanceEvaluator,
    RegressionCategory,
    render_acceptance_report,
)
from backend_ai.evaluation.benchmark import (
    BenchmarkComparisonRunner,
    BenchmarkComparisonStore,
    BenchmarkDataset,
    BenchmarkExecutionResult,
    BenchmarkModelSpec,
    BenchmarkProtocolConfig,
    BenchmarkRunStore,
    BenchmarkTaskStatus,
)
from tests.unit.test_benchmark_phase115 import _models, _task


class _AcceptanceRuntime:
    def __init__(self, passed: bool) -> None:
        self.passed = passed

    def execute(self, task, workspace_root: Path, *, max_wall_time: float):
        if self.passed:
            (workspace_root / "result.txt").write_text("accepted fixture\n", encoding="utf-8")
            return BenchmarkExecutionResult(
                status=BenchmarkTaskStatus.PASSED,
                execution_status="COMPLETED",
                termination_reason="COMPLETED",
                test_evidence={"status": "PASS"},
                completion_evidence={"status": "COMPLETE"},
                final_verification_evidence={"status": "VERIFIED"},
                budget_state={"attempts": 1, "tool_calls": 2, "successful_tool_calls": 2, "failed_tool_calls": 0},
                recovery_state={"encountered_failure": True, "recovered": True},
                tests_requested=True,
                tests_executed=True,
            )
        return BenchmarkExecutionResult(
            status=BenchmarkTaskStatus.FAILED,
            execution_status="FAILED",
            termination_reason="FAILED",
            test_evidence={"status": "FAIL"},
            completion_evidence={"status": "FAILED"},
            failure_information=("critical task failure",),
            budget_state={"attempts": 1, "tool_calls": 2, "successful_tool_calls": 1, "failed_tool_calls": 1},
            recovery_state={"encountered_failure": True, "recovered": False},
            tests_requested=True,
            tests_executed=True,
        )


class _AcceptanceFactory:
    def __init__(self, *, base_passed: bool, candidate_passed: bool) -> None:
        self.base_passed = base_passed
        self.candidate_passed = candidate_passed

    def create(self, model, protocol):
        return _AcceptanceRuntime(self.base_passed if model.model_version == "base" else self.candidate_passed)


def _evidence(tmp_path: Path, *, base_passed: bool, candidate_passed: bool, comparison_id: str):
    dataset = BenchmarkDataset.from_tasks((_task("EVAL-ACCEPTANCE-001", "API_ENDPOINT", "EASY"), _task("EVAL-ACCEPTANCE-002", "BUG_FIX", "HARD")))
    base, candidate = _models(tmp_path)
    runner = BenchmarkComparisonRunner(
        runtime_factory=_AcceptanceFactory(base_passed=base_passed, candidate_passed=candidate_passed),
        protocol=BenchmarkProtocolConfig(store_path=None, timeout_seconds=1.0),
        run_store=BenchmarkRunStore(None),
        comparison_store=BenchmarkComparisonStore(None),
    )
    comparison = runner.run(dataset, base_model=base, candidate_model=candidate, comparison_id=comparison_id)
    return dataset, runner.run_store.get(f"{comparison_id}-base"), runner.run_store.get(f"{comparison_id}-candidate"), comparison


def _policy(**overrides) -> AcceptancePolicy:
    values = {
        "minimum_task_success_rate": 0.50,
        "minimum_test_pass_rate": 0.50,
        "minimum_tool_success_rate": 0.50,
        "minimum_error_recovery_rate": 0.50,
        "maximum_failure_rate": 0.50,
        "maximum_average_attempts": 4.0,
        "maximum_allowed_regression": 0,
        "minimum_required_improvement": 0.05,
        "minimum_improved_metrics": 2,
        "regression_tolerance": 0.01,
        "critical_regression_delta": 0.10,
        "maximum_overfitting_gap": 0.20,
        "require_held_out_test": True,
        "reject_on_overfitting_gap": True,
        "require_complete_evidence": True,
        "require_reproducibility": True,
    }
    values.update(overrides)
    return AcceptancePolicy(**values)


def _request(tmp_path: Path, *, base_passed: bool, candidate_passed: bool, comparison_id: str, policy: AcceptancePolicy | None = None, training_config=None, training_fingerprint="sha256:" + "a" * 64, validation_success_rate=None, held_out_test=True):
    dataset, base_run, candidate_run, comparison = _evidence(tmp_path, base_passed=base_passed, candidate_passed=candidate_passed, comparison_id=comparison_id)
    return AcceptanceRequest(
        evaluation_id=comparison_id,
        comparison=comparison,
        base_run=base_run,
        candidate_run=candidate_run,
        dataset=dataset,
        policy=policy or _policy(),
        candidate_training_config=training_config,
        training_dataset_fingerprint=training_fingerprint,
        validation_success_rate=validation_success_rate,
        held_out_test=held_out_test,
    )


def test_clear_improvement_is_accepted() -> None:
    request = _request(Path("/tmp"), base_passed=False, candidate_passed=True, comparison_id="accept-clear", training_config={"epochs": 1, "learning_rate": 0.001})
    report = ModelAcceptanceEvaluator().evaluate(request)
    assert report.decision is AcceptanceDecision.ACCEPT
    assert not report.regressions
    assert len(report.metrics) >= 6
    assert report.reproducibility.valid
    assert "minimum_improved_metrics" not in report.reason
    assert "FODCI MODEL ACCEPTANCE REPORT" in render_acceptance_report(report)


def test_no_meaningful_improvement_is_rejected() -> None:
    request = _request(Path("/tmp"), base_passed=True, candidate_passed=True, comparison_id="reject-no-improvement", training_config={"epochs": 1})
    report = ModelAcceptanceEvaluator().evaluate(request)
    assert report.decision is AcceptanceDecision.REJECT
    assert "minimum_improved_metrics" in report.reason


def test_improvement_with_critical_capability_regression_is_rejected() -> None:
    request = _request(Path("/tmp"), base_passed=True, candidate_passed=False, comparison_id="reject-capability", training_config={"epochs": 1})
    report = ModelAcceptanceEvaluator().evaluate(request)
    assert report.decision is AcceptanceDecision.REJECT
    assert any(item.category is RegressionCategory.CAPABILITY and item.critical for item in report.regressions)
    assert "critical_regression" in report.reason


def test_overfitting_gap_is_reported_and_rejected() -> None:
    request = _request(Path("/tmp"), base_passed=False, candidate_passed=False, comparison_id="reject-overfit", training_config={"epochs": 1}, validation_success_rate=1.0)
    report = ModelAcceptanceEvaluator().evaluate(request)
    assert report.decision is AcceptanceDecision.REJECT
    assert any(item.category is RegressionCategory.OVERFITTING for item in report.regressions)


def test_missing_training_metadata_fails_closed_as_invalid_evaluation() -> None:
    request = _request(Path("/tmp"), base_passed=False, candidate_passed=True, comparison_id="invalid-metadata", training_config=None, training_fingerprint=None)
    report = ModelAcceptanceEvaluator().evaluate(request)
    assert report.decision is AcceptanceDecision.INVALID_EVALUATION
    assert "training_configuration" in report.reason
    assert "training_dataset_fingerprint" in report.reason
    assert not report.reproducibility.valid


def test_mismatched_benchmark_identity_is_invalid_evaluation(tmp_path: Path) -> None:
    request = _request(tmp_path, base_passed=False, candidate_passed=True, comparison_id="invalid-benchmark", training_config={"epochs": 1})
    other_dataset = BenchmarkDataset.from_tasks((_task("EVAL-OTHER-001"),))
    invalid_request = replace(request, dataset=other_dataset)
    report = ModelAcceptanceEvaluator().evaluate(invalid_request)
    assert report.decision is AcceptanceDecision.INVALID_EVALUATION
    assert "comparison_dataset_fingerprint_mismatch" in report.reason


def test_held_out_test_is_required() -> None:
    request = _request(Path("/tmp"), base_passed=False, candidate_passed=True, comparison_id="invalid-heldout", training_config={"epochs": 1}, held_out_test=False)
    report = ModelAcceptanceEvaluator().evaluate(request)
    assert report.decision is AcceptanceDecision.INVALID_EVALUATION
    assert "held_out_test_not_confirmed" in report.reason or "held_out_test_identity" in report.reason


def test_acceptance_policy_can_change_thresholds_without_code_changes() -> None:
    request = _request(Path("/tmp"), base_passed=False, candidate_passed=True, comparison_id="custom-policy", policy=_policy(minimum_required_improvement=1.1), training_config={"epochs": 1})
    report = ModelAcceptanceEvaluator().evaluate(request)
    assert report.decision is AcceptanceDecision.REJECT
    assert report.policy.minimum_required_improvement == 1.1


def test_acceptance_report_store_is_immutable_and_reloadable(tmp_path: Path) -> None:
    request = _request(tmp_path, base_passed=False, candidate_passed=True, comparison_id="stored-accept", training_config={"epochs": 1})
    report = ModelAcceptanceEvaluator().evaluate(request)
    store = AcceptanceStore(tmp_path / "acceptance.json")
    assert store.save(report).to_json() == report.to_json()
    reloaded = AcceptanceStore(tmp_path / "acceptance.json").get("stored-accept")
    assert reloaded is not None
    assert reloaded.to_json() == report.to_json()
    assert len(AcceptanceStore(tmp_path / "acceptance.json").list_reports()) == 1
