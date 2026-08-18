from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts" / "evaluation" / "phase155_heldout_evaluation.json"


def load_report() -> dict:
    assert REPORT.is_file(), f"Phase 15.5 report is missing: {REPORT}"
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_phase155_heldout_evaluation_is_complete_and_same_dataset() -> None:
    report = load_report()

    assert report["format"] == "fodci.phase155_heldout_evaluation"
    assert report["phase"] == "15.5"
    assert report["dataset_version"] == "phase141-v1"
    assert report["same_benchmark"] is True
    assert report["all_cases_completed"] is True
    assert report["phase_gates_passed"] is True
    assert report["phase_gates"]["distilled_checkpoint_lineage_present"] is True


def test_phase155_does_not_promote_a_candidate_that_is_not_better() -> None:
    report = load_report()

    assert report["candidate_promoted"] is False
    assert report["response_quality_accepted"] is False
    assert report["stable_runtime_replaced"] is False
    assert report["distilled"]["understandable_heuristic_rate"] == 0.0
    assert report["deltas"]["repeated_token_rate"] > 0.0
    assert report["phase_gates"]["candidate_not_promoted_without_quality_proof"] is True
