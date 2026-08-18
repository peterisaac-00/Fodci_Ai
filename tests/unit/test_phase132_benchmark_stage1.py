from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_stage1 import load_benchmark, score_response


DATASET = Path(__file__).parents[2] / "training_data" / "fundamentals" / "evaluation" / "stage_01.jsonl"


def test_stage1_benchmark_is_held_out_and_unique() -> None:
    records = load_benchmark(DATASET)
    assert len(records) == 24
    assert len({record["benchmark_id"] for record in records}) == len(records)
    assert len({record["question"] for record in records}) == len(records)
    assert all(record["split"] == "benchmark" for record in records)


def test_score_response_is_deterministic_and_reports_missing_keywords() -> None:
    record = load_benchmark(DATASET)[0]
    result = score_response(record, "A server receives a request, applies business logic, reads data, and returns a response.")
    assert result["passed"] is True
    assert result["keyword_coverage"] == 1.0
    assert result["missing_keywords"] == []
    assert score_response(record, "") == {
        "matched_keywords": [],
        "missing_keywords": record["required_keywords"],
        "keyword_coverage": 0.0,
        "minimum_keyword_coverage": 0.75,
        "non_empty": False,
        "passed": False,
    }


def test_load_benchmark_rejects_duplicate_questions(tmp_path: Path) -> None:
    records = load_benchmark(DATASET)
    records[1]["question"] = records[0]["question"]
    path = tmp_path / "duplicate.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate benchmark identity"):
        load_benchmark(path)


def test_baseline_report_contains_reproducibility_metadata() -> None:
    report_path = Path(__file__).parents[2] / "artifacts" / "evaluation" / "stage1_baseline.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["format"] == "fodci.stage1_baseline"
    assert report["model"]["parameter_count"] == 11_424_400
    assert report["dataset"]["records"] == 24
    assert report["protocol"]["decoding"] == "greedy_argmax"
    assert report["evaluation"]["aggregate"]["items"] == 24
    assert report["run_id"].startswith("stage1-baseline-")
