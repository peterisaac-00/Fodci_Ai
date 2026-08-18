from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts" / "evaluation" / "phase145_backend_scope.json"


def load_report() -> dict:
    assert REPORT.is_file(), f"Phase 14.5 report is missing: {REPORT}"
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_phase145_all_scope_probes_pass() -> None:
    report = load_report()

    assert report["format"] == "fodci.phase145_backend_scope"
    assert report["phase"] == "14.5"
    assert report["probe_count"] == 5
    assert report["passed_probe_count"] == 5
    assert report["phase_gates_passed"] is True
    assert all(report["phase_gates"].values())


def test_phase145_blocks_out_of_scope_before_inner_provider() -> None:
    report = load_report()

    assert report["out_of_scope_calls_blocked_before_provider"] is True
    assert report["inner_provider_calls"] == 3
    assert report["stable_runtime_replaced"] is False
    assert report["default_fodci_provider_changed"] is False
