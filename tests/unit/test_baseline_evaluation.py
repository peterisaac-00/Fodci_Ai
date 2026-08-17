from __future__ import annotations

from pathlib import Path

from backend_ai.evaluation.baseline import (
    BASELINE_EVALUATION_PROTOCOL_VERSION,
    BaselineEvaluationConfig,
    BaselineEvaluationConflictError,
    BaselineEvaluationDataset,
    BaselineEvaluationRunner,
    BaselineEvaluationStore,
    BaselineEvaluationStatus,
    BenchmarkExecutionResult,
    ModelIdentity,
    load_evaluation_dataset,
)
from backend_ai.evaluation.benchmark_runner import BenchmarkTaskStatus


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, task, workspace_root: Path, *, max_wall_time: float):
        self.calls.append(task.task_id)
        if task.task_id.endswith("001"):
            return BenchmarkExecutionResult(
                BenchmarkTaskStatus.PASSED,
                execution_status="COMPLETED",
                completion_evidence={"status": "COMPLETE"},
                budget_state={"attempts": 2, "tool_calls": 2, "successful_tool_calls": 2, "failed_tool_calls": 0},
            )
        if task.task_id.endswith("002"):
            return BenchmarkExecutionResult(
                BenchmarkTaskStatus.FAILED,
                execution_status="FAILED",
                failure_information=("verification failure",),
                recovery_state={"encountered_failure": True, "recovered": False},
                budget_state={"attempts": 3, "tool_calls": 2, "successful_tool_calls": 1, "failed_tool_calls": 1},
            )
        return BenchmarkExecutionResult(
            BenchmarkTaskStatus.PASSED,
            execution_status="COMPLETED",
            test_evidence={"status": "PASS"},
            tests_requested=True,
            tests_executed=True,
            recovery_state={"encountered_failure": True, "recovered": True},
            budget_state={"attempts": 4, "tool_calls": 3, "successful_tool_calls": 3, "failed_tool_calls": 0},
        )


def test_dataset_is_evaluation_only_and_deterministic() -> None:
    dataset = load_evaluation_dataset()
    assert dataset.evaluation_only is True
    assert dataset.dataset_version == "evaluation-v1"
    assert len(dataset.tasks) == 6
    assert dataset.dataset_fingerprint == BaselineEvaluationDataset.from_tasks(tuple(reversed(dataset.tasks))).dataset_fingerprint
    assert dict(dataset.category_counts)["API_ENDPOINT"] == 1


def test_runner_calculates_raw_metrics_and_preserves_failure() -> None:
    dataset = load_evaluation_dataset()
    runtime = FakeRuntime()
    store = BaselineEvaluationStore(None)
    runner = BaselineEvaluationRunner(
        runtime=runtime,
        model_identity=ModelIdentity("FakeFodciModel", "test-v1", None, None, 1),
        config=BaselineEvaluationConfig(store_path=None),
        store=store,
    )
    run = runner.run(dataset, evaluation_id="baseline-test-1", timestamp="2026-08-17T00:00:00Z")
    assert run.status is BaselineEvaluationStatus.COMPLETED
    assert run.aggregate.total_tasks == 6
    assert run.aggregate.successful_tasks == 5
    assert run.aggregate.failed_tasks == 1
    assert run.aggregate.task_success_rate == 5 / 6
    assert run.aggregate.test_pass_rate == 1.0
    assert run.aggregate.tool_success_rate == 15 / 16
    assert run.aggregate.recovery_attempted_tasks == 5
    assert run.aggregate.recovery_successful_tasks == 4
    assert run.aggregate.recovery_success_rate == 0.8
    assert run.aggregate.average_attempts == (2 + 3 + 4 + 4 + 4 + 4) / 6
    assert len(runtime.calls) == 6
    assert any(item.failure_reason == "verification failure" for item in run.task_results)


def test_failed_task_does_not_stop_benchmark_and_store_is_historical(tmp_path: Path) -> None:
    dataset = load_evaluation_dataset()
    path = tmp_path / "baseline_runs.json"
    runner = BaselineEvaluationRunner(
        runtime=FakeRuntime(),
        model_identity=ModelIdentity("FakeFodciModel", "test-v1", None, None, 1),
        config=BaselineEvaluationConfig(store_path=path),
    )
    first = runner.run(dataset, evaluation_id="baseline-test-1", timestamp="2026-08-17T00:00:00Z")
    second = runner.run(dataset, evaluation_id="baseline-test-2", timestamp="2026-08-17T00:00:00Z")
    assert path.is_file()
    reloaded = BaselineEvaluationStore(path)
    assert [item.evaluation_id for item in reloaded.list_runs()] == ["baseline-test-1", "baseline-test-2"]
    assert reloaded.get(first.evaluation_id).to_json() == first.to_json()
    assert reloaded.get(second.evaluation_id).aggregate.to_dict() == second.aggregate.to_dict()
    try:
        runner.run(dataset, evaluation_id="baseline-test-1", timestamp="different")
    except BaselineEvaluationConflictError:
        pass
    else:
        raise AssertionError("historical evaluation IDs must not be overwritten")


def test_model_identity_requires_real_fingerprint_format() -> None:
    identity = ModelIdentity("FodciModel", "fodci-tiny-v1", "artifacts/checkpoints/fodci-tiny-v1.pt", "sha256:" + "a" * 64, 1)
    assert identity.to_dict()["model_version"] == "fodci-tiny-v1"
    assert BASELINE_EVALUATION_PROTOCOL_VERSION == "11.1"
