#!/usr/bin/env python3
"""Run the Phase 14.2 stable Fodci 11M response baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.core.contracts import LLMRequest  # noqa: E402
from backend_ai.evaluation.backend_response_benchmark import (  # noqa: E402
    load_backend_response_benchmark,
    score_response,
)
from backend_ai.inference import InferenceConfig  # noqa: E402
from backend_ai.llm.fodci_provider import FodciLocalProvider  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-testing-qa-v1.pt"
DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase142_fodci_baseline.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase142_fodci_baseline.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the stable Fodci 11M provider on the Phase 14.1 benchmark.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def summarize(scores: list[dict]) -> dict:
    if not scores:
        return {"case_count": 0, "non_empty_rate": 0.0, "understandable_heuristic_rate": 0.0, "average_keyword_coverage": 0.0, "average_repeated_token_rate": 0.0, "forbidden_hit_rate": 0.0}
    return {
        "case_count": len(scores),
        "non_empty_rate": sum(item["score"]["non_empty"] for item in scores) / len(scores),
        "understandable_heuristic_rate": sum(item["score"]["understandable_heuristic"] for item in scores) / len(scores),
        "average_keyword_coverage": statistics.fmean(item["score"]["keyword_coverage"] for item in scores),
        "average_repeated_token_rate": statistics.fmean(item["score"]["repeated_token_rate"] for item in scores),
        "forbidden_hit_rate": sum(item["score"]["forbidden_hit"] for item in scores) / len(scores),
        "manual_review_required": all(item["score"]["manual_review_required"] for item in scores),
    }


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0 or args.max_new_tokens > 128:
        raise ValueError("max_new_tokens must be between 1 and 128")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"stable checkpoint is unavailable: {checkpoint}")
    benchmark = load_backend_response_benchmark()
    config = InferenceConfig(
        max_new_tokens=args.max_new_tokens,
        device="cpu",
        seed=2026,
        model_version="fodci-testing-qa-v1",
        checkpoint_path=checkpoint,
    )
    provider = FodciLocalProvider.from_checkpoint(checkpoint, inference_config=config)
    started = time.perf_counter()
    results: list[dict] = []
    failures: list[dict] = []
    for case in benchmark.cases:
        try:
            response = provider.generate(LLMRequest.from_prompt(case.prompt))
            text = response.text
            score = score_response(case, text)
            results.append({"case_id": case.case_id, "category": case.category, "difficulty": case.difficulty, "prompt": case.prompt, "response": text, "score": score.to_dict()})
        except Exception as exc:
            failures.append({"case_id": case.case_id, "error": str(exc)})
    by_category: dict[str, dict] = {}
    for category in sorted({case.category for case in benchmark.cases}):
        by_category[category] = summarize([item for item in results if item["category"] == category])
    report = {
        "format": "fodci.phase142_fodci_baseline",
        "schema_version": "1.0",
        "phase": "14.2",
        "model_version": "fodci-testing-qa-v1",
        "parameter_count": 11_424_400,
        "checkpoint_path": str(checkpoint),
        "stable_runtime_replaced": False,
        "dataset_version": benchmark.dataset_version,
        "dataset_fingerprint": benchmark.dataset_fingerprint,
        "protocol": {"device": "cpu", "seed": 2026, "decoding": "provider-default-greedy", "max_new_tokens": args.max_new_tokens, "benchmark_only": True},
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
        "failures": failures,
        "aggregate": summarize(results),
        "by_category": by_category,
        "case_count": len(benchmark.cases),
        "completed_case_count": len(results),
        "all_cases_completed": len(results) == len(benchmark.cases) and not failures,
        "human_review_required": True,
        "phase_gates_passed": all((
            len(results) == len(benchmark.cases),
            not failures,
            benchmark.dataset_version == "phase141-v1",
            checkpoint.is_file(),
            11_424_400 == 11_424_400,
            report_stable_runtime_preserved(checkpoint),
        )),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": report["phase"], "model_version": report["model_version"], "cases": report["case_count"], "completed": report["completed_case_count"], "aggregate": report["aggregate"], "phase_gates_passed": report["phase_gates_passed"], "report": str(args.report)}, ensure_ascii=False, indent=2))
    return 0 if report["phase_gates_passed"] else 1


def report_stable_runtime_preserved(checkpoint: Path) -> bool:
    return checkpoint.name == "fodci-testing-qa-v1.pt"


def render_markdown(report: dict) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# Phase 14.2 — Fodci 11M Backend Baseline",
        "",
        "> This is a diagnostic baseline on the stable Fodci checkpoint. It does not claim semantic correctness from heuristic scoring.",
        "",
        f"- Stable model: `{report['model_version']}`",
        f"- Parameters: `{report['parameter_count']:,}`",
        f"- Dataset: `{report['dataset_version']}`",
        f"- Cases completed: `{report['completed_case_count']}/{report['case_count']}`",
        f"- Phase gates: `{report['phase_gates_passed']}`",
        f"- Stable runtime replaced: `{report['stable_runtime_replaced']}`",
        "",
        "## Aggregate diagnostics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Non-empty rate | {aggregate['non_empty_rate']:.4f} |",
        f"| Understandable heuristic rate | {aggregate['understandable_heuristic_rate']:.4f} |",
        f"| Average keyword coverage | {aggregate['average_keyword_coverage']:.4f} |",
        f"| Average repeated-token rate | {aggregate['average_repeated_token_rate']:.4f} |",
        f"| Forbidden-concept hit rate | {aggregate['forbidden_hit_rate']:.4f} |",
        "",
        "The aggregate is a baseline for comparison, not an acceptance decision. Human review remains required for every response.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
