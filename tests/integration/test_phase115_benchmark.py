from __future__ import annotations

from pathlib import Path

from backend_ai.evaluation.benchmark import (
    BenchmarkComparisonRunner,
    BenchmarkComparisonStore,
    BenchmarkDataset,
    BenchmarkModelSpec,
    BenchmarkProtocolConfig,
    BenchmarkRunStore,
    BenchmarkTaskStatus,
    render_comparison_report,
)
from tests.unit.test_benchmark_phase115 import _MockFactory, _models, _task


def test_phase115_end_to_end_benchmark_comparison(tmp_path: Path) -> None:
    dataset = BenchmarkDataset.from_tasks((_task("EVAL-INTEGRATION-001", "API_ENDPOINT", "EASY"), _task("EVAL-INTEGRATION-002", "TESTING", "HARD")))
    base, candidate = _models(tmp_path)
    run_store = BenchmarkRunStore(tmp_path / "benchmark_runs.json")
    comparison_store = BenchmarkComparisonStore(tmp_path / "comparisons.json")
    runner = BenchmarkComparisonRunner(
        runtime_factory=_MockFactory(),
        protocol=BenchmarkProtocolConfig(store_path=tmp_path / "benchmark_runs.json", timeout_seconds=1.0),
        run_store=run_store,
        comparison_store=comparison_store,
    )

    comparison = runner.run(dataset, base_model=base, candidate_model=candidate, comparison_id="phase115-integration")

    assert comparison.status == "IMPROVED"
    assert comparison.dataset_version == "benchmark-v1"
    assert comparison.base_run_id == "phase115-integration-base"
    assert comparison.candidate_run_id == "phase115-integration-candidate"
    assert len(run_store.list_runs()) == 2
    assert all(run.task_results[0].status in {BenchmarkTaskStatus.PASSED.value, BenchmarkTaskStatus.FAILED.value} for run in run_store.list_runs())
    report = render_comparison_report(comparison)
    assert "FODCI MODEL BENCHMARK" in report
    assert "Delta" in report
    assert "By Category" in report
    assert "By Difficulty" in report
