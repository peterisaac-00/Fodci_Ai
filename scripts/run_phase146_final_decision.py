#!/usr/bin/env python3
"""Build the final Phase 14.6 comparison and non-activation decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "artifacts" / "evaluation" / "phase142_fodci_baseline.json"
DEFAULT_QWEN = ROOT / "artifacts" / "evaluation" / "phase144_qwen_benchmark.json"
DEFAULT_SCOPE = ROOT / "artifacts" / "evaluation" / "phase145_backend_scope.json"
DEFAULT_STABLE = ROOT / "artifacts" / "checkpoints" / "fodci-testing-qa-v1.pt"
DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase146_final_decision.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase146_final_decision.md"
EXPECTED_STABLE_SHA256 = "3af5d2b6009f5a0fd0ff98644d9666bd0c30f0dfe8994f91524ae6df11433bfa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Fodci 11M and Qwen evidence and record the Phase 14.6 decision.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--stable-checkpoint", type=Path, default=DEFAULT_STABLE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required Phase 14 evidence is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    baseline = read_json(args.baseline)
    qwen = read_json(args.qwen)
    scope = read_json(args.scope)
    stable = args.stable_checkpoint.resolve()
    stable_hash = sha256(stable) if stable.is_file() else None
    b = baseline["aggregate"]
    q = qwen["aggregate"]
    same_dataset = baseline["dataset_version"] == qwen["dataset_version"] == "phase141-v1" and baseline["dataset_fingerprint"] == qwen["dataset_fingerprint"]
    deltas = {
        "non_empty_rate": round(q["non_empty_rate"] - b["non_empty_rate"], 6),
        "understandable_heuristic_rate": round(q["understandable_heuristic_rate"] - b["understandable_heuristic_rate"], 6),
        "average_keyword_coverage": round(q["average_keyword_coverage"] - b["average_keyword_coverage"], 6),
        "repeated_token_rate": round(q["average_repeated_token_rate"] - b["average_repeated_token_rate"], 6),
    }
    gates = {
        "same_benchmark_dataset": same_dataset,
        "baseline_complete": baseline["all_cases_completed"] is True,
        "qwen_complete": qwen["all_cases_completed"] is True,
        "qwen_readability_improved": q["understandable_heuristic_rate"] > b["understandable_heuristic_rate"],
        "qwen_keyword_coverage_improved": q["average_keyword_coverage"] > b["average_keyword_coverage"],
        "scope_policy_passed": scope["phase_gates_passed"] is True,
        "stable_checkpoint_present": stable.is_file(),
        "stable_checkpoint_release_hash_matches": stable_hash == EXPECTED_STABLE_SHA256,
        "stable_runtime_not_replaced": qwen["stable_runtime_replaced"] is False,
        "quantized_q4_evidence_not_overclaimed": qwen["quantization"] != "q4",
        "manual_review_required": qwen["human_review_required"] is True,
    }
    report = {
        "format": "fodci.phase146_final_decision",
        "schema_version": "1.0",
        "phase": "14.6",
        "baseline_model": baseline["model_version"],
        "candidate_model": qwen["model_id"],
        "dataset_version": baseline["dataset_version"],
        "dataset_fingerprint": baseline["dataset_fingerprint"],
        "baseline": {"non_empty_rate": b["non_empty_rate"], "understandable_heuristic_rate": b["understandable_heuristic_rate"], "average_keyword_coverage": b["average_keyword_coverage"], "average_repeated_token_rate": b["average_repeated_token_rate"]},
        "candidate": {"non_empty_rate": q["non_empty_rate"], "understandable_heuristic_rate": q["understandable_heuristic_rate"], "average_keyword_coverage": q["average_keyword_coverage"], "average_repeated_token_rate": q["average_repeated_token_rate"], "quantization": qwen["quantization"], "manual_quality_notes": len(qwen.get("manual_quality_notes", []))},
        "deltas": deltas,
        "stable_checkpoint": {"path": str(stable), "sha256": stable_hash, "expected_release_sha256": EXPECTED_STABLE_SHA256, "matches_release": stable_hash == EXPECTED_STABLE_SHA256},
        "scope_report": {"phase_gates_passed": scope["phase_gates_passed"], "passed_probe_count": scope["passed_probe_count"]},
        "gates": gates,
        "decision": "adopt_qwen_as_experimental_backend_with_backend_policy",
        "stable_runtime_action": "keep_fodci_testing_qa_v1",
        "stable_runtime_replaced": False,
        "semantic_correctness_proven": False,
        "quantized_q4_validated": False,
        "qwen_1_5b_needed_now": False,
        "recommendation": "Use Qwen 0.5B only as an experimental language provider behind BackendScopedProvider. Do not replace the stable Fodci runtime and do not claim semantic correctness until execution-aware correctness tests pass.",
    }
    report["phase_gates_passed"] = all(gates.values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": report["phase"], "decision": report["decision"], "phase_gates_passed": report["phase_gates_passed"], "stable_runtime_replaced": report["stable_runtime_replaced"], "deltas": report["deltas"], "report": str(args.report)}, ensure_ascii=False, indent=2))
    return 0 if report["phase_gates_passed"] else 1


def render_markdown(report: dict) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    lines = [
        "# Phase 14.6 — Final Comparison and Decision",
        "",
        "> The final decision compares the exact same held-out benchmark and keeps the stable Fodci runtime unchanged.",
        "",
        "## Comparison",
        "",
        "| Metric | Fodci 11M | Qwen 0.5B | Delta |",
        "|---|---:|---:|---:|",
        f"| Non-empty rate | {baseline['non_empty_rate']:.4f} | {candidate['non_empty_rate']:.4f} | {report['deltas']['non_empty_rate']:+.4f} |",
        f"| Understandable heuristic rate | {baseline['understandable_heuristic_rate']:.4f} | {candidate['understandable_heuristic_rate']:.4f} | {report['deltas']['understandable_heuristic_rate']:+.4f} |",
        f"| Average keyword coverage | {baseline['average_keyword_coverage']:.4f} | {candidate['average_keyword_coverage']:.4f} | {report['deltas']['average_keyword_coverage']:+.4f} |",
        f"| Repeated-token rate | {baseline['average_repeated_token_rate']:.4f} | {candidate['average_repeated_token_rate']:.4f} | {report['deltas']['repeated_token_rate']:+.4f} |",
        "",
        "## Decision",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Stable runtime action: `{report['stable_runtime_action']}`",
        f"- Stable runtime replaced: `{report['stable_runtime_replaced']}`",
        f"- Semantic correctness proven: `{report['semantic_correctness_proven']}`",
        f"- Q4 quantization validated: `{report['quantized_q4_validated']}`",
        f"- All Phase 14.6 gates: `{report['phase_gates_passed']}`",
        "",
        report["recommendation"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
