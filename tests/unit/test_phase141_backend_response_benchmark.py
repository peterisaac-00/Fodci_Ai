from __future__ import annotations

from backend_ai.evaluation.backend_response_benchmark import (
    BENCHMARK_FORMAT,
    BENCHMARK_VERSION,
    load_backend_response_benchmark,
    score_response,
)


def test_phase141_dataset_is_versioned_backend_only_and_training_separate() -> None:
    dataset = load_backend_response_benchmark()

    assert dataset.format == BENCHMARK_FORMAT
    assert dataset.benchmark_version == BENCHMARK_VERSION
    assert dataset.dataset_version == "phase141-v1"
    assert dataset.benchmark_only is True
    assert len(dataset.cases) == 24
    assert not dataset.training_source_paths
    assert dataset.dataset_fingerprint.startswith("sha256:")


def test_phase141_covers_all_required_backend_categories() -> None:
    dataset = load_backend_response_benchmark()
    categories = {case.category for case in dataset.cases}

    assert categories == {
        "python-backend", "fastapi", "rest-http", "sql-postgresql",
        "auth-security", "testing", "debugging", "architecture",
    }
    assert all(case.case_id.startswith("B14-") for case in dataset.cases)
    assert len({case.case_id for case in dataset.cases}) == len(dataset.cases)


def test_phase141_scoring_recognizes_concepts_but_requires_manual_review() -> None:
    case = load_backend_response_benchmark().cases[0]
    score = score_response(
        case,
        "A list is mutable, while a tuple is immutable. Choose based on whether the backend data should change.",
    )

    assert score.non_empty is True
    assert score.keyword_coverage == 1.0
    assert score.understandable_heuristic is True
    assert score.manual_review_required is True


def test_phase141_scoring_flags_repetition_and_empty_output() -> None:
    case = load_backend_response_benchmark().cases[0]
    repeated = score_response(case, "Input Input Input Input Input Input")
    empty = score_response(case, "")

    assert repeated.non_empty is True
    assert repeated.repeated_token_rate > 0.55
    assert repeated.understandable_heuristic is False
    assert empty.non_empty is False
    assert empty.understandable_heuristic is False


def test_phase141_scoring_flags_forbidden_concepts() -> None:
    case = load_backend_response_benchmark().cases[12]
    score = score_response(case, "Store the password in plaintext so it is easy to recover.")

    assert score.forbidden_hit is True
    assert score.understandable_heuristic is False
