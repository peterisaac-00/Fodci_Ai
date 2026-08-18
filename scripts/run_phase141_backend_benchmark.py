#!/usr/bin/env python3
"""Validate and publish the Phase 14.1 backend-response benchmark manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.evaluation.backend_response_benchmark import (  # noqa: E402
    BENCHMARK_FORMAT,
    BENCHMARK_VERSION,
    load_backend_response_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Phase 14.1 backend-response benchmark.")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation/phase141_backend_benchmark.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_backend_response_benchmark(args.benchmark) if args.benchmark else load_backend_response_benchmark()
    categories = Counter(case.category for case in benchmark.cases)
    difficulties = Counter(case.difficulty for case in benchmark.cases)
    report = {
        "format": "fodci.phase141_backend_benchmark_report",
        "schema_version": "1.0",
        "phase": "14.1",
        "benchmark_format": BENCHMARK_FORMAT,
        "benchmark_version": BENCHMARK_VERSION,
        "dataset_version": benchmark.dataset_version,
        "dataset_fingerprint": benchmark.dataset_fingerprint,
        "benchmark_only": benchmark.benchmark_only,
        "training_source_paths": list(benchmark.training_source_paths),
        "case_count": len(benchmark.cases),
        "category_counts": dict(sorted(categories.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "requires_code_count": sum(case.requires_code for case in benchmark.cases),
        "all_cases_have_rubrics": all(case.expected_concepts for case in benchmark.cases),
        "all_case_ids_unique": len({case.case_id for case in benchmark.cases}) == len(benchmark.cases),
        "all_required_categories_present": len(categories) == 8,
        "data_contamination_checked": not benchmark.training_source_paths,
        "human_review_required": True,
        "model_executed": False,
        "training_performed": False,
        "phase_gates_passed": all((
            benchmark.benchmark_only,
            len(benchmark.cases) >= 16,
            len(categories) == 8,
            len({case.case_id for case in benchmark.cases}) == len(benchmark.cases),
            not benchmark.training_source_paths,
        )),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
