from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts" / "evaluation" / "phase144_qwen_benchmark.json"


def load_report() -> dict:
    assert REPORT.is_file(), f"Phase 14.4 report is missing: {REPORT}"
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_phase144_qwen_completes_same_benchmark_without_replacing_stable_runtime() -> None:
    report = load_report()

    assert report["format"] == "fodci.phase144_qwen_benchmark"
    assert report["phase"] == "14.4"
    assert report["model_id"] == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert report["case_count"] == 24
    assert report["completed_case_count"] == 24
    assert report["all_cases_completed"] is True
    assert report["phase_gates_passed"] is True
    assert report["stable_runtime_replaced"] is False
    assert report["default_fodci_checkpoint_untouched"] is True
    assert report["failures"] == []


def test_phase144_shows_improvement_but_preserves_human_review_and_quantization_caveat() -> None:
    report = load_report()
    aggregate = report["aggregate"]

    assert aggregate["non_empty_rate"] == 1.0
    assert aggregate["understandable_heuristic_rate"] > 0.9
    assert aggregate["average_keyword_coverage"] > 0.6
    assert aggregate["manual_review_required"] is True
    assert len(report["manual_quality_notes"]) >= 5
    assert report["quantization"] == "none-fp16-safetensors"


def test_phase144_protocol_is_cpu_local_and_remote_code_disabled() -> None:
    report = load_report()
    protocol = report["protocol"]

    assert protocol["device"] == "cpu"
    assert protocol["local_files_only"] is True
    assert protocol["trust_remote_code"] is False
    assert protocol["do_sample"] is False
    assert protocol["benchmark_only"] is True
