#!/usr/bin/env python3
"""Evaluate the experimental distilled checkpoint on the fixed held-out benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.core.contracts import LLMRequest  # noqa: E402
from backend_ai.evaluation.backend_response_benchmark import load_backend_response_benchmark, score_response  # noqa: E402
from backend_ai.inference import InferenceConfig  # noqa: E402
from backend_ai.llm.fodci_provider import FodciLocalProvider  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-distilled-phase154-v1.pt"
DEFAULT_STABLE = ROOT / "artifacts" / "checkpoints" / "fodci-testing-qa-v1.pt"
DEFAULT_PHASE154 = ROOT / "artifacts" / "evaluation" / "phase154_offline_distillation.json"
DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase155_heldout_evaluation.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase155_heldout_evaluation.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 15.4 distilled checkpoint on the fixed backend benchmark.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stable", type=Path, default=DEFAULT_STABLE)
    parser.add_argument("--phase154-report", type=Path, default=DEFAULT_PHASE154)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def summarize(scores: list[dict]) -> dict:
    if not scores:
        return {"case_count": 0, "non_empty_rate": 0.0, "understandable_heuristic_rate": 0.0, "average_keyword_coverage": 0.0, "average_repeated_token_rate": 0.0, "forbidden_hit_rate": 0.0, "manual_review_required": True}
    return {
        "case_count": len(scores),
        "non_empty_rate": sum(item["score"]["non_empty"] for item in scores) / len(scores),
        "understandable_heuristic_rate": sum(item["score"]["understandable_heuristic"] for item in scores) / len(scores),
        "average_keyword_coverage": statistics.fmean(item["score"]["keyword_coverage"] for item in scores),
        "average_repeated_token_rate": statistics.fmean(item["score"]["repeated_token_rate"] for item in scores),
        "forbidden_hit_rate": sum(item["score"]["forbidden_hit"] for item in scores) / len(scores),
        "manual_review_required": True,
    }


def evaluate(checkpoint: Path, benchmark, max_new_tokens: int, model_version: str) -> tuple[list[dict], list[dict], float]:
    config = InferenceConfig(max_new_tokens=max_new_tokens, device="cpu", seed=2026, model_version=model_version, checkpoint_path=checkpoint)
    provider = FodciLocalProvider.from_checkpoint(checkpoint, inference_config=config)
    results: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    for case in benchmark.cases:
        try:
            response = provider.generate(LLMRequest.from_prompt(case.prompt))
            results.append({"case_id": case.case_id, "category": case.category, "prompt": case.prompt, "response": response.text, "score": score_response(case, response.text).to_dict()})
        except Exception as exc:
            failures.append({"case_id": case.case_id, "error": str(exc)})
    return results, failures, time.perf_counter() - started


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_new_tokens <= 128:
        raise ValueError("max_new_tokens must be between 1 and 128")
    for path in (args.checkpoint, args.stable, args.phase154_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    phase154 = json.loads(args.phase154_report.read_text(encoding="utf-8"))
    benchmark = load_backend_response_benchmark()
    started = time.perf_counter()
    distilled_results, distilled_failures, distilled_seconds = evaluate(args.checkpoint.resolve(), benchmark, args.max_new_tokens, "fodci-distilled-phase154-v1")
    stable_results, stable_failures, stable_seconds = evaluate(args.stable.resolve(), benchmark, args.max_new_tokens, "fodci-testing-qa-v1")
    distilled = summarize(distilled_results)
    stable = summarize(stable_results)
    report = {
        "format": "fodci.phase155_heldout_evaluation",
        "schema_version": "1.0",
        "phase": "15.5",
        "dataset_version": benchmark.dataset_version,
        "dataset_fingerprint": benchmark.dataset_fingerprint,
        "checkpoint": str(args.checkpoint.resolve()),
        "stable_checkpoint": str(args.stable.resolve()),
        "phase154_training_report": str(args.phase154_report.resolve()),
        "protocol": {"device": "cpu", "seed": 2026, "max_new_tokens": args.max_new_tokens, "benchmark_only": True, "same_dataset_for_both_models": True},
        "distilled": distilled,
        "stable": stable,
        "distilled_results": distilled_results,
        "distilled_failures": distilled_failures,
        "stable_failures": stable_failures,
        "elapsed_seconds": time.perf_counter() - started,
        "distilled_seconds": distilled_seconds,
        "stable_seconds": stable_seconds,
        "all_cases_completed": len(distilled_results) == len(benchmark.cases) and not distilled_failures and len(stable_results) == len(benchmark.cases) and not stable_failures,
        "same_benchmark": True,
        "stable_runtime_replaced": False,
        "candidate_promoted": False,
        "response_quality_accepted": False,
        "phase154_training_gates_passed": phase154["training_gates_passed"],
    }
    report["deltas"] = {
        "understandable_heuristic_rate": distilled["understandable_heuristic_rate"] - stable["understandable_heuristic_rate"],
        "average_keyword_coverage": distilled["average_keyword_coverage"] - stable["average_keyword_coverage"],
        "repeated_token_rate": distilled["average_repeated_token_rate"] - stable["average_repeated_token_rate"],
    }
    report["phase_gates"] = {
        "same_dataset": report["same_benchmark"] and benchmark.dataset_version == "phase141-v1",
        "both_models_complete": report["all_cases_completed"],
        "distilled_checkpoint_lineage_present": phase154["base_checkpoint"].endswith("fodci-testing-qa-v1.pt"),
        "stable_runtime_preserved": report["stable_runtime_replaced"] is False,
        "candidate_not_promoted_without_quality_proof": report["candidate_promoted"] is False and report["response_quality_accepted"] is False,
        "manual_review_required": distilled["manual_review_required"] is True,
    }
    report["phase_gates_passed"] = all(report["phase_gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": report["phase"], "distilled": report["distilled"], "stable": report["stable"], "deltas": report["deltas"], "phase_gates_passed": report["phase_gates_passed"], "candidate_promoted": report["candidate_promoted"], "report": str(args.report)}, ensure_ascii=False, indent=2))
    return 0 if report["phase_gates_passed"] else 1


def render_markdown(report: dict) -> str:
    d = report["distilled"]
    s = report["stable"]
    return "\n".join([
        "# Phase 15.5 — Held-out Evaluation and Regression",
        "",
        "> The distilled candidate and stable runtime were evaluated on the same fixed benchmark. The candidate was not promoted automatically.",
        "",
        "| Metric | Distilled Fodci | Stable Fodci | Delta |",
        "|---|---:|---:|---:|",
        f"| Non-empty rate | {d['non_empty_rate']:.4f} | {s['non_empty_rate']:.4f} | {report['deltas']['non_empty_rate'] if 'non_empty_rate' in report['deltas'] else d['non_empty_rate'] - s['non_empty_rate']:+.4f} |",
        f"| Understandable heuristic rate | {d['understandable_heuristic_rate']:.4f} | {s['understandable_heuristic_rate']:.4f} | {report['deltas']['understandable_heuristic_rate']:+.4f} |",
        f"| Average keyword coverage | {d['average_keyword_coverage']:.4f} | {s['average_keyword_coverage']:.4f} | {report['deltas']['average_keyword_coverage']:+.4f} |",
        f"| Repeated-token rate | {d['average_repeated_token_rate']:.4f} | {s['average_repeated_token_rate']:.4f} | {report['deltas']['repeated_token_rate']:+.4f} |",
        "",
        f"- All cases completed: `{report['all_cases_completed']}`",
        f"- Candidate promoted: `{report['candidate_promoted']}`",
        f"- Response quality accepted: `{report['response_quality_accepted']}`",
        f"- Stable runtime replaced: `{report['stable_runtime_replaced']}`",
        f"- Phase gates: `{report['phase_gates_passed']}`",
        "",
        "A non-empty response or loss improvement is not enough to promote a student model. The candidate remains experimental until response quality and execution-aware correctness are demonstrated.",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
