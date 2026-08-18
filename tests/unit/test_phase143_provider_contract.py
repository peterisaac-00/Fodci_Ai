from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts" / "evaluation" / "phase143_provider_contract.json"


def load_report() -> dict:
    assert REPORT.is_file(), f"Phase 14.3 report is missing: {REPORT}"
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_phase143_provider_contract_gates_pass() -> None:
    report = load_report()

    assert report["format"] == "fodci.phase143_provider_contract"
    assert report["phase"] == "14.3"
    assert report["provider"] == "PretrainedCodeProvider"
    assert report["phase_gates_passed"] is True
    assert all(report["phase_gates"].values())


def test_phase143_does_not_change_default_runtime_or_call_external_services() -> None:
    report = load_report()

    assert report["stable_runtime_replaced"] is False
    assert report["default_fodci_provider_changed"] is False
    assert report["model_downloaded"] is False
    assert report["external_api_used"] is False
    assert report["dependency_policy"] == "optional-lazy-local-files-only"
