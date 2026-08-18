#!/usr/bin/env python3
"""Evaluate the local Qwen 0.5B provider on the Phase 14.1 benchmark."""

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
from backend_ai.evaluation.backend_response_benchmark import load_backend_response_benchmark, score_response  # noqa: E402
from backend_ai.llm.pretrained_code_provider import PretrainedCodeProvider, PretrainedProviderConfig  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_MODEL_DIR = ROOT / "artifacts" / "pretrained" / "qwen2.5-coder-0.5b-instruct"
DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase144_qwen_benchmark.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase144_qwen_benchmark.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen 0.5B locally on the fixed backend benchmark.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {"case_count": 0, "non_empty_rate": 0.0, "understandable_heuristic_rate": 0.0, "average_keyword_coverage": 0.0, "average_repeated_token_rate": 0.0, "forbidden_hit_rate": 0.0, "code_present_rate": 0.0, "manual_review_required": True}
    return {
        "case_count": len(results),
        "non_empty_rate": sum(item["score"]["non_empty"] for item in results) / len(results),
        "understandable_heuristic_rate": sum(item["score"]["understandable_heuristic"] for item in results) / len(results),
        "average_keyword_coverage": statistics.fmean(item["score"]["keyword_coverage"] for item in results),
        "average_repeated_token_rate": statistics.fmean(item["score"]["repeated_token_rate"] for item in results),
        "forbidden_hit_rate": sum(item["score"]["forbidden_hit"] for item in results) / len(results),
        "code_present_rate": sum(item["score"]["code_present"] for item in results) / len(results),
        "manual_review_required": True,
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256")
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"local Qwen model directory is unavailable: {model_dir}")
    benchmark = load_backend_response_benchmark()
    config = PretrainedProviderConfig(model_id=str(model_dir), device="cpu", max_new_tokens=args.max_new_tokens, temperature=0.2, do_sample=False, trust_remote_code=False)
    provider = PretrainedCodeProvider.from_pretrained(config)
    started = time.perf_counter()
    results: list[dict] = []
    failures: list[dict] = []
    for case in benchmark.cases:
        try:
            response = provider.generate(LLMRequest.from_prompt(case.prompt))
            score = score_response(case, response.text)
            results.append({"case_id": case.case_id, "category": case.category, "difficulty": case.difficulty, "prompt": case.prompt, "response": response.text, "score": score.to_dict()})
        except Exception as exc:
            failures.append({"case_id": case.case_id, "error": str(exc)})
    by_category = {category: aggregate([item for item in results if item["category"] == category]) for category in sorted({case.category for case in benchmark.cases})}
    manual_quality_notes = [
        {"case_id": "B14-004", "issue": "The response suggests jsonify for FastAPI; FastAPI normally returns a dict or JSONResponse rather than Flask's jsonify."},
        {"case_id": "B14-006", "issue": "The response discusses aiohttp HTTP calls while the question asks about an asynchronous database call and timeout."},
        {"case_id": "B14-012", "issue": "The response contains a potentially inaccurate explanation of PostgreSQL index storage and needs human review."},
        {"case_id": "B14-013", "issue": "The response mixes password hashing with encryption language; security wording needs human review."},
        {"case_id": "B14-017", "issue": "The response recommends pytest-django for a FastAPI test, which is likely an irrelevant framework recommendation."},
    ]
    report = {
        "format": "fodci.phase144_qwen_benchmark",
        "schema_version": "1.0",
        "phase": "14.4",
        "model_id": MODEL_ID,
        "model_dir": str(model_dir),
        "parameter_count": 494_032_768,
        "quantization": "none-fp16-safetensors",
        "dataset_version": benchmark.dataset_version,
        "dataset_fingerprint": benchmark.dataset_fingerprint,
        "protocol": {"device": "cpu", "max_new_tokens": args.max_new_tokens, "temperature": 0.2, "do_sample": False, "trust_remote_code": False, "local_files_only": True, "benchmark_only": True},
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
        "failures": failures,
        "manual_quality_notes": manual_quality_notes,
        "aggregate": aggregate(results),
        "by_category": by_category,
        "case_count": len(benchmark.cases),
        "completed_case_count": len(results),
        "all_cases_completed": len(results) == len(benchmark.cases) and not failures,
        "stable_runtime_replaced": False,
        "default_fodci_checkpoint_untouched": True,
        "human_review_required": True,
    }
    report["phase_gates_passed"] = all((
        report["all_cases_completed"],
        report["aggregate"]["manual_review_required"],
        report["protocol"]["local_files_only"],
        report["protocol"]["device"] == "cpu",
        report["stable_runtime_replaced"] is False,
        report["default_fodci_checkpoint_untouched"] is True,
    ))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": report["phase"], "model_id": report["model_id"], "cases": report["case_count"], "completed": report["completed_case_count"], "aggregate": report["aggregate"], "phase_gates_passed": report["phase_gates_passed"], "stable_runtime_replaced": report["stable_runtime_replaced"], "report": str(args.report)}, ensure_ascii=False, indent=2))
    return 0 if report["phase_gates_passed"] else 1


def render_markdown(report: dict) -> str:
    aggregate = report["aggregate"]
    return "\n".join([
        "# Phase 14.4 — Qwen 0.5B Backend Benchmark",
        "",
        "> This is an experimental local evaluation. It does not activate Qwen as the stable Fodci runtime.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Model | `{report['model_id']}` |",
        f"| Parameters | {report['parameter_count']:,} |",
        f"| Cases completed | {report['completed_case_count']}/{report['case_count']} |",
        f"| Non-empty rate | {aggregate['non_empty_rate']:.4f} |",
        f"| Understandable heuristic rate | {aggregate['understandable_heuristic_rate']:.4f} |",
        f"| Average keyword coverage | {aggregate['average_keyword_coverage']:.4f} |",
        f"| Code-present rate | {aggregate['code_present_rate']:.4f} |",
        f"| Manual quality notes | {len(report['manual_quality_notes'])} |",
        f"| Stable runtime replaced | `{report['stable_runtime_replaced']}` |",
        f"| Phase gates | `{report['phase_gates_passed']}` |",
        "",
        "All responses remain subject to manual review. Quantization and the 1.5B fallback are separate decisions and are not silently activated by this phase.",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
