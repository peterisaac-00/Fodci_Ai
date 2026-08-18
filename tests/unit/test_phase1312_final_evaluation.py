from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def load_report() -> dict[str, object]:
    return json.loads((ROOT / "artifacts" / "evaluation" / "phase1312_final_evaluation.json").read_text(encoding="utf-8"))


def test_phase1312_release_gates_all_pass() -> None:
    report = load_report()
    assert report["format"] == "fodci.phase1312_final_evaluation"
    assert report["phase"] == "13.12"
    assert report["release_gates"]["all_passed"] is True
    assert all(report["release_gates"].values())


def test_phase1312_audits_complete_checkpoint_lineage_and_stable_model() -> None:
    report = load_report()
    audit = report["checkpoint_audit"]
    assert audit["all_required_present"] is True
    assert len(audit["chain"]) == 9
    assert report["stable_model_version"] == "fodci-testing-qa-v1"
    assert report["stable_parameter_count"] == 11_424_400
    assert report["experimental_scaling_model"] == "fodci-scaling-25m-experimental-v1"
    assert report["experimental_scaling_activated"] is False


def test_phase1312_preserves_honest_benchmark_diagnostics() -> None:
    report = load_report()
    benchmarks = report["benchmarks"]
    assert len(benchmarks["reports"]) == 7
    assert benchmarks["all_have_items"] is True
    assert benchmarks["all_metrics_valid"] is True
    assert benchmarks["all_non_empty"] is False
    assert any(item["non_empty_rate"] == 0.0 for item in benchmarks["reports"])
    assert all(item["pass_rate"] == 0.0 for item in benchmarks["reports"])


def test_phase1312_verifies_runtime_synergy_and_full_regression() -> None:
    report = load_report()
    assert report["runtime_smoke"]["model_version"] == "fodci-testing-qa-v1"
    assert report["runtime_smoke"]["generated_token_count"] <= 4
    assert report["synergy"]["multi_agent_completed_subtasks"] == 4
    assert report["synergy"]["multi_agent_total_subtasks"] == 4
    assert report["synergy"]["memory_reload_retrieval"] is True
    assert report["synergy"]["autonomy_completed"] is True
    assert report["tests"]["passed"] >= 1064
    assert report["tests"]["compileall"] is True
