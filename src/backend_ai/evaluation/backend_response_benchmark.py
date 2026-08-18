"""Phase 14.1 backend-response benchmark and transparent scoring helpers.

The benchmark is deliberately separate from training data and from the existing
agent/tool benchmark. It measures the language provider on short backend
questions before tool execution is evaluated in later phases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


BENCHMARK_FORMAT = "fodci.phase141_backend_response_benchmark"
BENCHMARK_VERSION = "backend-response-v1"
SCHEMA_VERSION = "1.0"
DEFAULT_DATASET_PATH = Path(__file__).with_name("datasets") / "phase141_backend_response_benchmark.json"
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")


class BackendResponseBenchmarkError(ValueError):
    """Raised when the Phase 14.1 benchmark is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class BackendBenchmarkCase:
    """One immutable backend question and its transparent scoring rubric."""

    case_id: str
    category: str
    difficulty: str
    prompt: str
    expected_concepts: tuple[str, ...]
    forbidden_concepts: tuple[str, ...] = ()
    requires_code: bool = False
    max_expected_tokens: int = 180

    def __post_init__(self) -> None:
        if not re.fullmatch(r"B14-\d{3}", self.case_id):
            raise BackendResponseBenchmarkError(f"invalid case_id: {self.case_id!r}")
        if self.category not in {
            "python-backend", "fastapi", "rest-http", "sql-postgresql",
            "auth-security", "testing", "debugging", "architecture",
        }:
            raise BackendResponseBenchmarkError(f"unsupported category: {self.category!r}")
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise BackendResponseBenchmarkError(f"unsupported difficulty: {self.difficulty!r}")
        if not self.prompt.strip() or not self.expected_concepts:
            raise BackendResponseBenchmarkError("prompt and expected_concepts are required")
        if self.max_expected_tokens < 16 or self.max_expected_tokens > 512:
            raise BackendResponseBenchmarkError("max_expected_tokens is outside the benchmark bound")
        object.__setattr__(self, "expected_concepts", tuple(sorted(set(self.expected_concepts))))
        object.__setattr__(self, "forbidden_concepts", tuple(sorted(set(self.forbidden_concepts))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "expected_concepts": list(self.expected_concepts),
            "forbidden_concepts": list(self.forbidden_concepts),
            "requires_code": self.requires_code,
            "max_expected_tokens": self.max_expected_tokens,
        }


@dataclass(frozen=True, slots=True)
class BackendResponseBenchmark:
    """Versioned benchmark dataset that is never used as training data."""

    format: str
    schema_version: str
    benchmark_version: str
    dataset_version: str
    benchmark_only: bool
    training_source_paths: tuple[str, ...]
    cases: tuple[BackendBenchmarkCase, ...]
    dataset_fingerprint: str

    def __post_init__(self) -> None:
        if self.format != BENCHMARK_FORMAT or self.schema_version != SCHEMA_VERSION:
            raise BackendResponseBenchmarkError("benchmark format/schema is invalid")
        if self.benchmark_version != BENCHMARK_VERSION or not self.dataset_version.strip():
            raise BackendResponseBenchmarkError("benchmark version is invalid")
        if self.benchmark_only is not True:
            raise BackendResponseBenchmarkError("benchmark_only must be true")
        ordered = tuple(sorted(self.cases, key=lambda item: item.case_id))
        if len(ordered) < 16:
            raise BackendResponseBenchmarkError("benchmark must contain at least 16 cases")
        if len({case.case_id for case in ordered}) != len(ordered):
            raise BackendResponseBenchmarkError("benchmark case IDs must be unique")
        categories = {case.category for case in ordered}
        if categories != {
            "python-backend", "fastapi", "rest-http", "sql-postgresql",
            "auth-security", "testing", "debugging", "architecture",
        }:
            raise BackendResponseBenchmarkError("benchmark must cover all required backend categories")
        object.__setattr__(self, "cases", ordered)
        object.__setattr__(self, "training_source_paths", tuple(sorted(set(self.training_source_paths))))
        expected = compute_fingerprint(self.benchmark_version, self.dataset_version, ordered)
        if self.dataset_fingerprint != expected:
            raise BackendResponseBenchmarkError("dataset fingerprint does not match canonical cases")

    @classmethod
    def from_cases(
        cls,
        cases: Sequence[BackendBenchmarkCase],
        *,
        dataset_version: str = "phase141-v1",
        training_source_paths: Sequence[str] = (),
    ) -> "BackendResponseBenchmark":
        ordered = tuple(sorted(cases, key=lambda item: item.case_id))
        return cls(
            BENCHMARK_FORMAT,
            SCHEMA_VERSION,
            BENCHMARK_VERSION,
            dataset_version,
            True,
            tuple(training_source_paths),
            ordered,
            compute_fingerprint(BENCHMARK_VERSION, dataset_version, ordered),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "benchmark_version": self.benchmark_version,
            "dataset_version": self.dataset_version,
            "benchmark_only": self.benchmark_only,
            "training_source_paths": list(self.training_source_paths),
            "dataset_fingerprint": self.dataset_fingerprint,
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class BackendResponseScore:
    """Transparent, non-LLM score for one generated response."""

    case_id: str
    non_empty: bool
    word_count: int
    keyword_coverage: float
    forbidden_hit: bool
    repeated_token_rate: float
    understandable_heuristic: bool
    code_present: bool
    manual_review_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def compute_fingerprint(
    benchmark_version: str,
    dataset_version: str,
    cases: Sequence[BackendBenchmarkCase],
) -> str:
    payload = {
        "format": BENCHMARK_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": benchmark_version,
        "dataset_version": dataset_version,
        "benchmark_only": True,
        "cases": [case.to_dict() for case in sorted(cases, key=lambda item: item.case_id)],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def load_backend_response_benchmark(path: Path | str = DEFAULT_DATASET_PATH) -> BackendResponseBenchmark:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendResponseBenchmarkError(f"benchmark dataset is unavailable or malformed: {path}") from exc
    required = {
        "format", "schema_version", "benchmark_version", "dataset_version",
        "benchmark_only", "training_source_paths", "dataset_fingerprint", "cases",
    }
    if not isinstance(payload, Mapping) or set(payload) != required or not isinstance(payload["cases"], list):
        raise BackendResponseBenchmarkError("benchmark dataset fields are invalid")
    cases = tuple(
        BackendBenchmarkCase(
            case_id=item["case_id"],
            category=item["category"],
            difficulty=item["difficulty"],
            prompt=item["prompt"],
            expected_concepts=tuple(item["expected_concepts"]),
            forbidden_concepts=tuple(item.get("forbidden_concepts", ())),
            requires_code=bool(item.get("requires_code", False)),
            max_expected_tokens=int(item.get("max_expected_tokens", 180)),
        )
        for item in payload["cases"]
    )
    return BackendResponseBenchmark(
        payload["format"], payload["schema_version"], payload["benchmark_version"],
        payload["dataset_version"], payload["benchmark_only"],
        tuple(payload["training_source_paths"]), cases, payload["dataset_fingerprint"],
    )


def score_response(case: BackendBenchmarkCase, response: str) -> BackendResponseScore:
    text = response.strip() if isinstance(response, str) else ""
    words = [word.lower() for word in _WORD_PATTERN.findall(text)]
    normalized = " ".join(words)
    expected_hits = sum(1 for concept in case.expected_concepts if _concept_in_text(concept, normalized))
    keyword_coverage = expected_hits / len(case.expected_concepts)
    forbidden_hit = any(_concept_in_text(concept, normalized) for concept in case.forbidden_concepts)
    repeated_token_rate = _repeated_token_rate(words)
    code_present = bool(re.search(r"```|\b(def|class|async|SELECT|INSERT|UPDATE|curl|pytest)\b", response, re.IGNORECASE))
    understandable = bool(
        text
        and 3 <= len(words) <= case.max_expected_tokens
        and repeated_token_rate <= 0.55
        and keyword_coverage >= 0.5
        and not forbidden_hit
    )
    return BackendResponseScore(
        case.case_id,
        bool(text),
        len(words),
        round(keyword_coverage, 6),
        forbidden_hit,
        round(repeated_token_rate, 6),
        understandable,
        code_present,
        True,
    )


def _concept_in_text(concept: str, normalized: str) -> bool:
    tokens = " ".join(_WORD_PATTERN.findall(concept.lower()))
    return bool(tokens) and tokens in normalized


def _repeated_token_rate(words: Sequence[str]) -> float:
    if not words:
        return 0.0
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(words)


__all__ = [
    "BENCHMARK_FORMAT",
    "BENCHMARK_VERSION",
    "BackendBenchmarkCase",
    "BackendResponseBenchmark",
    "BackendResponseBenchmarkError",
    "BackendResponseScore",
    "DEFAULT_DATASET_PATH",
    "compute_fingerprint",
    "load_backend_response_benchmark",
    "score_response",
]
