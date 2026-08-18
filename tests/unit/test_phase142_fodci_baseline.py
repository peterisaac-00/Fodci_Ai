from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts" / "evaluation" / "phase142_fodci_baseline.json"


def load_report() -> dict:
    assert REPORT.is_file(), f"Phase 14.2 report is missing: {REPORT}"
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_phase142_completes_all_backend_cases_with_stable_identity() -> None:
    report = load_report()

    assert report["format"] == "fodci.phase142_fodci_baseline"
    assert report["phase"] == "14.2"
    assert report["model_version"] == "fodci-testing-qa-v1"
    assert report["parameter_count"] == 11_424_400
    assert report["dataset_version"] == "phase141-v1"
    assert report["case_count"] == 24
    assert report["completed_case_count"] == 24
    assert report["all_cases_completed"] is True
    assert report["phase_gates_passed"] is True
    assert report["stable_runtime_replaced"] is False


def test_phase142_records_honest_quality_limitation() -> None:
    report = load_report()
    aggregate = report["aggregate"]

    assert aggregate["manual_review_required"] is True
    assert aggregate["non_empty_rate"] == 1.0
    assert aggregate["understandable_heuristic_rate"] == 0.0
    assert aggregate["average_keyword_coverage"] == 0.0
    assert all(item["score"]["manual_review_required"] for item in report["results"])


def test_phase142_protocol_is_cpu_bounded_and_reproducible() -> None:
    report = load_report()
    protocol = report["protocol"]

    assert protocol["device"] == "cpu"
    assert protocol["seed"] == 2026
    assert protocol["max_new_tokens"] == 32
    assert protocol["benchmark_only"] is True
    assert report["failures"] == []
