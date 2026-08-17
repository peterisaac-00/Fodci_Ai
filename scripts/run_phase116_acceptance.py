#!/usr/bin/env python3
"""Evaluate persisted Phase 11.5 evidence through the Phase 11.6 acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.evaluation.acceptance import (  # noqa: E402
    AcceptanceDecision,
    AcceptancePolicy,
    AcceptanceRequest,
    AcceptanceStore,
    ModelAcceptanceEvaluator,
    render_acceptance_report,
)
from backend_ai.evaluation.benchmark import (  # noqa: E402
    BenchmarkComparisonStore,
    BenchmarkRunStore,
    load_benchmark_dataset,
)
from backend_ai.model_artifact import ModelArtifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the Phase 11.6 fail-closed acceptance gate to persisted Phase 11.5 evidence.")
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--benchmark-dataset", type=Path, default=None)
    parser.add_argument("--comparison-store", type=Path, default=Path("artifacts/evaluation/benchmark_comparisons.json"))
    parser.add_argument("--runs-store", type=Path, default=Path("artifacts/evaluation/benchmark_runs.json"))
    parser.add_argument("--acceptance-store", type=Path, default=Path("artifacts/evaluation/acceptance_reports.json"))
    parser.add_argument("--candidate-artifact", type=Path, default=None)
    parser.add_argument("--training-config", type=Path, default=None, help="Canonical JSON training configuration when no Model Artifact supplies it.")
    parser.add_argument("--training-dataset-fingerprint", default=None)
    parser.add_argument("--validation-success-rate", type=float, default=None)
    parser.add_argument("--policy", type=Path, default=None, help="JSON object overriding AcceptancePolicy fields.")
    parser.add_argument("--held-out-test", action="store_true", help="Explicitly confirm that the benchmark is a held-out test set.")
    parser.add_argument("--human-report", type=Path, default=None)
    parser.add_argument("--json-report", type=Path, default=None)
    return parser.parse_args()


def _json_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def main() -> int:
    args = parse_args()
    try:
        dataset = load_benchmark_dataset(args.benchmark_dataset) if args.benchmark_dataset is not None else load_benchmark_dataset()
        comparisons = BenchmarkComparisonStore(args.comparison_store)
        comparison = comparisons.get(args.evaluation_id)
        if comparison is None:
            raise ValueError(f"comparison not found: {args.evaluation_id}")
        runs = BenchmarkRunStore(args.runs_store)
        base_run = runs.get(comparison.base_run_id)
        candidate_run = runs.get(comparison.candidate_run_id)
        if base_run is None or candidate_run is None:
            raise ValueError("comparison references missing benchmark run evidence")
        candidate_artifact = ModelArtifact.load(args.candidate_artifact) if args.candidate_artifact is not None else None
        training_config = _json_file(args.training_config) if args.training_config is not None else None
        policy_values = _json_file(args.policy) if args.policy is not None else {}
        policy = AcceptancePolicy(**policy_values)
        request = AcceptanceRequest(evaluation_id=args.evaluation_id, comparison=comparison, base_run=base_run, candidate_run=candidate_run, dataset=dataset, policy=policy, candidate_artifact=candidate_artifact, candidate_training_config=training_config, training_dataset_fingerprint=args.training_dataset_fingerprint, validation_success_rate=args.validation_success_rate, held_out_test=args.held_out_test)
        report = ModelAcceptanceEvaluator().evaluate(request)
        AcceptanceStore(args.acceptance_store).save(report)
        human = render_acceptance_report(report)
        if args.human_report is not None:
            args.human_report.parent.mkdir(parents=True, exist_ok=True)
            args.human_report.write_text(human + "\n", encoding="utf-8")
        if args.json_report is not None:
            args.json_report.parent.mkdir(parents=True, exist_ok=True)
            args.json_report.write_text(report.to_json() + "\n", encoding="utf-8")
        print(human)
        print("\nAcceptance store:", args.acceptance_store)
        return 0 if report.decision is AcceptanceDecision.ACCEPT else 2
    except Exception as exc:
        print(f"Phase 11.6 acceptance failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
