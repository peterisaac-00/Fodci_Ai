from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from backend_ai.evaluation.benchmark import (
    BENCHMARK_DATASET_VERSION,
    BenchmarkComparisonRunner,
    BenchmarkComparisonStore,
    BenchmarkContaminationError,
    BenchmarkDataset,
    BenchmarkExecutionResult,
    BenchmarkModelSpec,
    BenchmarkProtocolConfig,
    BenchmarkRunStore,
    BenchmarkTaskResult,
    BenchmarkTaskStatus,
    FodciBenchmarkRuntimeFactory,
    ModelIdentity,
    build_comparison,
    load_benchmark_dataset,
    render_comparison_report,
)
from backend_ai.evaluation.task_model import (
    AllowedScope,
    EvaluationDifficulty,
    EvaluationTask,
    ExpectedArea,
    ExpectedAreaType,
    ExpectedBehavior,
    GroundTruth,
    ProjectDefinition,
    Requirement,
    SuccessCriterion,
    SuccessCriterionType,
)


class _MockRuntime:
    def __init__(self, should_pass: bool) -> None:
        self.should_pass = should_pass

    def execute(self, task, workspace_root: Path, *, max_wall_time: float):
        if self.should_pass:
            (workspace_root / "result.txt").write_text("fixed\n", encoding="utf-8")
            return BenchmarkExecutionResult(
                status=BenchmarkTaskStatus.PASSED,
                execution_status="COMPLETED",
                termination_reason="COMPLETED",
                test_evidence={"status": "PASS"},
                completion_evidence={"status": "COMPLETE"},
                final_verification_evidence={"status": "VERIFIED"},
                budget_state={"attempts": 1, "tool_calls": 2, "successful_tool_calls": 2, "failed_tool_calls": 0},
                recovery_state={"encountered_failure": False, "recovered": False},
                tests_requested=True,
                tests_executed=True,
            )
        return BenchmarkExecutionResult(
            status=BenchmarkTaskStatus.FAILED,
            execution_status="FAILED",
            termination_reason="FAILED",
            test_evidence={"status": "FAIL"},
            completion_evidence={"status": "FAILED"},
            failure_information=("mock failure",),
            budget_state={"attempts": 2, "tool_calls": 1, "successful_tool_calls": 0, "failed_tool_calls": 1},
            recovery_state={"encountered_failure": True, "recovered": False},
            tests_requested=True,
            tests_executed=True,
        )


class _MockFactory:
    def create(self, model, protocol):
        return _MockRuntime(model.model_version == "candidate")


def _task(task_id: str = "EVAL-MOCK-001", category: str = "API_ENDPOINT", difficulty: str = "EASY") -> EvaluationTask:
    return EvaluationTask(
        task_id=task_id,
        title="Mock backend task",
        description="Implement a bounded backend fixture change.",
        version="1.0",
        category=category,
        difficulty=difficulty,
        project_definition=ProjectDefinition(project_type="backend-service", language="Python", runtime="Python 3.11+", test_framework="pytest"),
        user_intent="Create the required backend fixture change.",
        expected_behaviors=(ExpectedBehavior("B1", "empty fixture", "write result", "result exists", "complete"),),
        requirements=(Requirement("R1", "Create the result file."),),
        allowed_scope=AllowedScope(allowed_files=("result.txt",), allowed_change_types=("CREATE",)),
        expected_areas=(ExpectedArea("result", ("result.txt",), ExpectedAreaType.REQUIRED_CHANGE),),
        success_criteria=(SuccessCriterion("C1", "The task completes.", SuccessCriterionType.COMPLETION, True, "completion evidence", ("R1",), (), ("B1",)),),
        ground_truth=GroundTruth(("B1",), ("result exists",), (), (), ("write result",)),
        metadata={"benchmark_only": "true"},
    )


def _dataset() -> BenchmarkDataset:
    return BenchmarkDataset.from_tasks((_task(), _task("EVAL-MOCK-002", "DATABASE", "MEDIUM")))


def _models(tmp_path: Path):
    checkpoint = tmp_path / "mock.pt"
    checkpoint.write_bytes(b"mock-checkpoint")
    fingerprint = "sha256:" + hashlib.sha256(b"mock-checkpoint").hexdigest()
    return (
        BenchmarkModelSpec("base", ModelIdentity("MockModel", "base", str(checkpoint), fingerprint, 1), checkpoint, checkpoint_model_version="mock-base"),
        BenchmarkModelSpec("candidate", ModelIdentity("MockModel", "candidate", str(checkpoint), fingerprint, 1), checkpoint, checkpoint_model_version="mock-candidate"),
    )


def test_benchmark_dataset_loads_and_is_distinct_from_training() -> None:
    dataset = load_benchmark_dataset()
    assert dataset.dataset_version == BENCHMARK_DATASET_VERSION
    assert dataset.benchmark_only is True
    assert len(dataset.tasks) == 6
    assert dataset.dataset_fingerprint.startswith("sha256:")
    assert not dataset.source_record_ids


def test_contamination_is_rejected_before_execution() -> None:
    dataset = BenchmarkDataset.from_tasks((_task(),), training_dataset_fingerprints=("sha256:" + "0" * 64,))
    with pytest.raises(BenchmarkContaminationError):
        dataset.validate_contamination(training_dataset_fingerprint=dataset.dataset_fingerprint)


def test_comparison_runs_both_models_on_same_protocol_and_persists_raw_results(tmp_path: Path) -> None:
    dataset = _dataset()
    base, candidate = _models(tmp_path)
    protocol = BenchmarkProtocolConfig(store_path=tmp_path / "runs.json", timeout_seconds=1.0)
    runner = BenchmarkComparisonRunner(
        runtime_factory=_MockFactory(),
        protocol=protocol,
        run_store=BenchmarkRunStore(tmp_path / "runs.json"),
        comparison_store=BenchmarkComparisonStore(tmp_path / "comparisons.json"),
    )
    comparison = runner.run(dataset, base_model=base, candidate_model=candidate, comparison_id="mock-comparison")

    assert comparison.status == "IMPROVED"
    assert comparison.dataset_fingerprint == dataset.dataset_fingerprint
    assert comparison.base_run_id == "mock-comparison-base"
    assert comparison.candidate_run_id == "mock-comparison-candidate"
    assert comparison.overall_metrics[0].delta is not None
    assert any(item.name == "task_success_rate" and item.classification == "IMPROVED" for item in comparison.overall_metrics)
    assert comparison.by_category
    assert comparison.by_difficulty
    assert (tmp_path / "runs.json").is_file()
    assert (tmp_path / "comparisons.json").is_file()
    assert len(BenchmarkRunStore(tmp_path / "runs.json").list_runs()) == 2
    report = render_comparison_report(comparison)
    assert "Base" in report and "Candidate" in report and "Delta" in report
    assert "By Category" in report and "By Difficulty" in report


def test_failed_task_preserves_raw_failure_and_metrics(tmp_path: Path) -> None:
    dataset = BenchmarkDataset.from_tasks((_task(),))
    base, candidate = _models(tmp_path)
    protocol = BenchmarkProtocolConfig(store_path=None, timeout_seconds=1.0)
    runner = BenchmarkComparisonRunner(runtime_factory=_MockFactory(), protocol=protocol, run_store=BenchmarkRunStore(None), comparison_store=BenchmarkComparisonStore(None))
    comparison = runner.run(dataset, base_model=base, candidate_model=candidate, comparison_id="failed-preserved")
    base_run = runner.run_store.get("failed-preserved-base")
    assert base_run is not None
    assert base_run.task_results[0].success is False
    assert base_run.task_results[0].failure_reason == "mock failure"
    assert base_run.task_results[0].errors == ("mock failure",)
    assert comparison.overall_metrics


def test_report_comparison_is_deterministic_for_fixed_inputs(tmp_path: Path) -> None:
    dataset = _dataset()
    base, candidate = _models(tmp_path)
    protocol = BenchmarkProtocolConfig(store_path=None, timeout_seconds=1.0)
    runner = BenchmarkComparisonRunner(runtime_factory=_MockFactory(), protocol=protocol, run_store=BenchmarkRunStore(None), comparison_store=BenchmarkComparisonStore(None))
    first = runner.run(dataset, base_model=base, candidate_model=candidate, comparison_id="first")
    second = runner.run(dataset, base_model=base, candidate_model=candidate, comparison_id="second")
    first_report = render_comparison_report(first)
    second_report = render_comparison_report(second)
    assert first_report.replace("first", "comparison") == second_report.replace("second", "comparison")
