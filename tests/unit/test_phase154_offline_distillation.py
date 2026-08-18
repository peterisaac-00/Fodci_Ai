from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts" / "evaluation" / "phase154_offline_distillation.json"


def load_report() -> dict:
    assert REPORT.is_file(), f"Phase 15.4 report is missing: {REPORT}"
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_phase154_training_gates_pass() -> None:
    report = load_report()

    assert report["format"] == "fodci.phase154_offline_distillation"
    assert report["phase"] == "15.4"
    assert report["parameter_count"] == 11_424_400
    assert report["train_records"] == 8
    assert report["validation_records"] == 4
    assert report["global_step"] == 32
    assert report["training_gates_passed"] is True
    assert report["finite_loss"] is True
    assert report["parameters_changed"] is True
    assert report["checkpoint_exists"] is True
    assert report["checkpoint_reload"] is True
    assert report["non_empty_splits"] is True


def test_phase154_validation_improved_but_stable_runtime_was_preserved() -> None:
    report = load_report()

    assert report["trained_validation_loss"] < report["baseline_validation_loss"]
    assert report["validation_quality_gate_passed"] is True
    assert report["automatic_online_training"] is False
    assert report["stable_runtime_replaced"] is False
