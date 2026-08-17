from __future__ import annotations

from pathlib import Path

from backend_ai.evaluation.baseline import (
    BaselineEvaluationConfig,
    BaselineEvaluationRunner,
    BaselineEvaluationStore,
    ModelIdentity,
    load_evaluation_dataset,
)
from backend_ai.evaluation.benchmark_runner import BenchmarkExecutionResult, BenchmarkTaskStatus


class IntegrationRuntime:
    def execute(self, task, workspace_root: Path, *, max_wall_time: float):
        return BenchmarkExecutionResult(
            BenchmarkTaskStatus.PASSED,
            execution_status="COMPLETED",
            completion_evidence={"status": "COMPLETE", "source": "integration-runtime"},
            budget_state={"attempts": 1, "tool_calls": 1, "successful_tool_calls": 1, "failed_tool_calls": 0},
        )


def test_phase111_pipeline_and_historical_reload(tmp_path: Path) -> None:
    dataset = load_evaluation_dataset()
    store_path = tmp_path / "baseline_runs.json"
    runner = BaselineEvaluationRunner(
        runtime=IntegrationRuntime(),
        model_identity=ModelIdentity("IntegrationModel", "fixture-v1", None, None, 1),
        config=BaselineEvaluationConfig(store_path=store_path),
    )
    run = runner.run(dataset, evaluation_id="integration-baseline-1", timestamp="2026-08-17T00:00:00Z")
    assert run.aggregate.task_success_rate == 1.0
    assert run.aggregate.average_attempts == 1.0
    assert run.aggregate.failure_rate == 0.0
    assert store_path.is_file()
    reloaded = BaselineEvaluationStore(store_path)
    assert reloaded.get("integration-baseline-1").dataset_fingerprint == dataset.dataset_fingerprint
    assert reloaded.get("integration-baseline-1").model_identity.model_version == "fixture-v1"
