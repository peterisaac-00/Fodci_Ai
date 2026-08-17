#!/usr/bin/env python3
"""Run one explicit offline Phase 11.5 benchmark comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.evaluation.benchmark import (  # noqa: E402
    BenchmarkComparisonRunner,
    BenchmarkComparisonStore,
    BenchmarkModelSpec,
    BenchmarkProtocolConfig,
    BenchmarkRunStore,
    FodciBenchmarkRuntimeFactory,
    load_benchmark_dataset,
    render_comparison_report,
)
from backend_ai.model_artifact import ModelArtifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated Phase 11.5 Base-vs-Candidate benchmark; this does not train or accept a model.")
    parser.add_argument("--benchmark", type=Path, default=None, help="Versioned benchmark dataset JSON; defaults to the repository Phase 11.5 dataset.")
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate-checkpoint", type=Path)
    candidate.add_argument("--candidate-artifact", type=Path)
    parser.add_argument("--base-version", default="base")
    parser.add_argument("--candidate-version", default="candidate-v1")
    parser.add_argument("--base-tokenizer-version", type=int, default=1)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--system-prompt-version", default="system-v1")
    parser.add_argument("--agent-version", default="0.1.0")
    parser.add_argument("--tool-version", default="ToolRegistry.default-v1")
    parser.add_argument("--runs-per-task", type=int, default=1)
    parser.add_argument("--training-dataset-fingerprint", default=None)
    parser.add_argument("--runs-store", type=Path, default=Path("artifacts/evaluation/benchmark_runs.json"))
    parser.add_argument("--comparison-store", type=Path, default=Path("artifacts/evaluation/benchmark_comparisons.json"))
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset = load_benchmark_dataset(args.benchmark) if args.benchmark is not None else load_benchmark_dataset()
        base = BenchmarkModelSpec.from_checkpoint(args.base_checkpoint, model_version=args.base_version, tokenizer_version=args.base_tokenizer_version)
        if args.candidate_artifact is not None:
            candidate = BenchmarkModelSpec.from_artifact(ModelArtifact.load(args.candidate_artifact))
            if args.candidate_version != candidate.model_version:
                raise ValueError("--candidate-version must match candidate artifact model_version")
        else:
            candidate = BenchmarkModelSpec.from_checkpoint(args.candidate_checkpoint, model_version=args.candidate_version, tokenizer_version=args.base_tokenizer_version)
        protocol = BenchmarkProtocolConfig(seed=args.seed, temperature=args.temperature, max_tokens=args.max_tokens, max_iterations=args.max_iterations, timeout_seconds=args.timeout_seconds, system_prompt_version=args.system_prompt_version, agent_version=args.agent_version, tool_version=args.tool_version, runs_per_task=args.runs_per_task, store_path=args.runs_store)
        runner = BenchmarkComparisonRunner(runtime_factory=FodciBenchmarkRuntimeFactory(), protocol=protocol, run_store=BenchmarkRunStore(args.runs_store), comparison_store=BenchmarkComparisonStore(args.comparison_store))
        comparison = runner.run(dataset, base_model=base, candidate_model=candidate, comparison_id=args.comparison_id, training_dataset_fingerprint=args.training_dataset_fingerprint)
        report = render_comparison_report(comparison)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report + "\n", encoding="utf-8")
        print(report)
        print("\nRaw run store:", args.runs_store)
        print("Comparison store:", args.comparison_store)
        return 0
    except Exception as exc:
        print(f"Phase 11.5 benchmark failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
