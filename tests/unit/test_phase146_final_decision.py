from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts" / "evaluation" / "phase146_final_decision.json"


def load_report() -> dict:
    assert REPORT.is_file(), f"Phase 14.6 report is missing: {REPORT}"
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_phase146_comparison_gates_pass_and_uses_same_benchmark() -> None:
    report = load_report()

    assert report["format"] == "fodci.phase146_final_decision"
    assert report["phase"] == "14.6"
    assert report["dataset_version"] == "phase141-v1"
    assert report["gates"]["same_benchmark_dataset"] is True
    assert report["gates"]["baseline_complete"] is True
    assert report["gates"]["qwen_complete"] is True
    assert report["phase_gates_passed"] is True


def test_phase146_records_clear_language_improvement_without_overclaiming_correctness() -> None:
    report = load_report()

    assert report["deltas"]["understandable_heuristic_rate"] > 0.9
    assert report["deltas"]["average_keyword_coverage"] > 0.6
    assert report["candidate"]["manual_quality_notes"] >= 5
    assert report["semantic_correctness_proven"] is False
    assert report["quantized_q4_validated"] is False
    assert report["qwen_1_5b_needed_now"] is False


def test_phase146_preserves_stable_runtime_and_release_identity() -> None:
    report = load_report()

    assert report["stable_runtime_replaced"] is False
    assert report["stable_runtime_action"] == "keep_fodci_testing_qa_v1"
    assert report["stable_checkpoint"]["matches_release"] is True
    assert report["gates"]["stable_runtime_not_replaced"] is True
    assert report["decision"] == "adopt_qwen_as_experimental_backend_with_backend_policy"
