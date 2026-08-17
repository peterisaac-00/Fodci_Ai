"""Phase 11.5 reproducible benchmark and Base-vs-Candidate comparison.

This module evaluates immutable model identities through the existing bounded
BenchmarkRunner.  It never trains, changes weights, edits the source
repository, or decides model acceptance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from types import MappingProxyType
from typing import Any, Protocol

from backend_ai.evaluation.baseline import (
    BaselineEvaluationConfig,
    ModelIdentity,
    _task_from_dict,
    model_identity_from_checkpoint,
    create_current_model_runtime,
)
from backend_ai.evaluation.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkExecutionResult,
    BenchmarkRequest,
    BenchmarkResult,
    BenchmarkRuntime,
    BenchmarkRunner as EvidenceBenchmarkRunner,
    BenchmarkTaskRun,
    BenchmarkTaskStatus,
)
from backend_ai.evaluation.metrics import BenchmarkMetrics, collect_metrics
from backend_ai.evaluation.task_model import EvaluationTask, EvaluationTaskValidator
from backend_ai.evaluation.version_comparison import compare_evaluation_metrics
from backend_ai.checkpoint import CheckpointManager
from backend_ai.model_artifact import ModelArtifact


BENCHMARK_FORMAT = "fodci.benchmark_run"
BENCHMARK_PROTOCOL_VERSION = "11.5"
BENCHMARK_DATASET_FORMAT = "fodci.backend_benchmark_dataset"
BENCHMARK_DATASET_VERSION = "benchmark-v1"
BENCHMARK_VERSION = "backend-v1"
BENCHMARK_SCHEMA_VERSION = "1.0"
DEFAULT_BENCHMARK_DATASET_PATH = Path(__file__).with_name("datasets") / "phase115_backend_benchmark.json"
DEFAULT_BENCHMARK_STORE_PATH = Path("artifacts") / "evaluation" / "benchmark_runs.json"
DEFAULT_COMPARISON_STORE_PATH = Path("artifacts") / "evaluation" / "benchmark_comparisons.json"
_FINGERPRINT_PATTERN = r"^sha256:[0-9a-f]{64}$"


class BenchmarkError(ValueError):
    """Base Phase 11.5 benchmark error."""


class BenchmarkContaminationError(BenchmarkError):
    """Raised when a benchmark overlaps training data or source identities."""


class BenchmarkConflictError(BenchmarkError):
    """Raised when immutable benchmark storage would be overwritten."""


class BenchmarkStatus(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    INVALID = "INVALID"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """Versioned benchmark-only task collection, separate from training."""

    format: str
    benchmark_version: str
    dataset_version: str
    protocol_version: str
    benchmark_only: bool
    training_dataset_fingerprints: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    tasks: tuple[EvaluationTask, ...]
    dataset_fingerprint: str

    def __post_init__(self) -> None:
        if self.format != BENCHMARK_DATASET_FORMAT or self.protocol_version != BENCHMARK_PROTOCOL_VERSION or self.benchmark_version != BENCHMARK_VERSION:
            raise BenchmarkError("benchmark dataset format/version/protocol is invalid")
        if not self.benchmark_only:
            raise BenchmarkError("benchmark dataset must be marked benchmark_only")
        object.__setattr__(self, "training_dataset_fingerprints", tuple(sorted(set(self.training_dataset_fingerprints))))
        object.__setattr__(self, "source_record_ids", tuple(sorted(set(self.source_record_ids))))
        object.__setattr__(self, "tasks", tuple(sorted(self.tasks, key=lambda item: item.task_id)))
        if not self.tasks:
            raise BenchmarkError("benchmark dataset must contain tasks")
        validator = EvaluationTaskValidator()
        task_ids: set[str] = set()
        for task in self.tasks:
            if not isinstance(task, EvaluationTask) or task.task_id in task_ids:
                raise BenchmarkError("benchmark task IDs must be unique valid EvaluationTasks")
            task_ids.add(task.task_id)
            if not validator.validate(task).valid:
                raise BenchmarkError(f"benchmark task is invalid: {task.task_id}")
        if any(not isinstance(value, str) or not value.startswith("sha256:") for value in self.training_dataset_fingerprints):
            raise BenchmarkError("training dataset fingerprints are invalid")
        expected = compute_benchmark_dataset_fingerprint(self.benchmark_version, self.dataset_version, self.protocol_version, self.tasks)
        if self.dataset_fingerprint != expected:
            raise BenchmarkError("benchmark dataset fingerprint does not match canonical tasks")
        self.validate_contamination()

    @classmethod
    def from_tasks(cls, tasks: Sequence[EvaluationTask], *, benchmark_version: str = BENCHMARK_VERSION, dataset_version: str = BENCHMARK_DATASET_VERSION, training_dataset_fingerprints: Sequence[str] = (), source_record_ids: Sequence[str] = ()) -> "BenchmarkDataset":
        ordered = tuple(sorted(tasks, key=lambda item: item.task_id))
        return cls(BENCHMARK_DATASET_FORMAT, benchmark_version, dataset_version, BENCHMARK_PROTOCOL_VERSION, True, tuple(training_dataset_fingerprints), tuple(source_record_ids), ordered, compute_benchmark_dataset_fingerprint(benchmark_version, dataset_version, BENCHMARK_PROTOCOL_VERSION, ordered))

    def validate_contamination(self, *, training_dataset_fingerprint: str | None = None, training_source_record_ids: Sequence[str] = ()) -> None:
        fingerprints = set(self.training_dataset_fingerprints)
        if training_dataset_fingerprint:
            fingerprints.add(training_dataset_fingerprint)
        if self.dataset_fingerprint in fingerprints:
            raise BenchmarkContaminationError("benchmark dataset fingerprint overlaps a training dataset fingerprint")
        overlap = set(self.source_record_ids) & set(training_source_record_ids)
        if overlap:
            raise BenchmarkContaminationError(f"benchmark source record IDs overlap training IDs: {sorted(overlap)[:8]}")

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "benchmark_version": self.benchmark_version, "dataset_version": self.dataset_version, "protocol_version": self.protocol_version, "benchmark_only": self.benchmark_only, "training_dataset_fingerprints": list(self.training_dataset_fingerprints), "source_record_ids": list(self.source_record_ids), "dataset_fingerprint": self.dataset_fingerprint, "tasks": [task.to_dict() for task in self.tasks]}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class BenchmarkProtocolConfig:
    """Same inference/runtime protocol applied to base and candidate models."""

    seed: int = 2026
    temperature: float = 1.0
    max_tokens: int = 32
    max_iterations: int = 16
    timeout_seconds: float = 60.0
    system_prompt_version: str = "system-v1"
    agent_version: str = "0.1.0"
    tool_version: str = "ToolRegistry.default-v1"
    runs_per_task: int = 1
    deterministic: bool = True
    store_path: Path | str | None = DEFAULT_BENCHMARK_STORE_PATH

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise BenchmarkError("seed must be a non-negative integer")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) or self.temperature <= 0:
            raise BenchmarkError("temperature must be positive")
        for name in ("max_tokens", "max_iterations", "runs_per_task"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise BenchmarkError(f"{name} must be a positive integer")
        if self.runs_per_task > 8:
            raise BenchmarkError("runs_per_task exceeds the benchmark safety bound")
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(self.timeout_seconds, bool) or self.timeout_seconds <= 0:
            raise BenchmarkError("timeout_seconds must be positive")
        for name in ("system_prompt_version", "agent_version", "tool_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise BenchmarkError(f"{name} must contain text")
        if not isinstance(self.deterministic, bool):
            raise BenchmarkError("deterministic must be boolean")
        if self.store_path is not None:
            object.__setattr__(self, "store_path", Path(self.store_path).expanduser())

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "temperature": float(self.temperature), "max_tokens": self.max_tokens, "max_iterations": self.max_iterations, "timeout_seconds": float(self.timeout_seconds), "system_prompt_version": self.system_prompt_version, "agent_version": self.agent_version, "tool_version": self.tool_version, "runs_per_task": self.runs_per_task, "deterministic": self.deterministic, "store_path": str(self.store_path) if self.store_path is not None else None}


@dataclass(frozen=True, slots=True)
class BenchmarkModelSpec:
    """Exact model identity and checkpoint used by one benchmark arm."""

    model_version: str
    model_identity: ModelIdentity
    checkpoint_path: Path
    artifact_fingerprint: str | None = None
    model_artifact_id: str | None = None
    checkpoint_model_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_version, str) or not self.model_version.strip() or not isinstance(self.model_identity, ModelIdentity):
            raise BenchmarkError("model specification identity is invalid")
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path).expanduser())
        if not self.checkpoint_path.is_file():
            raise BenchmarkError(f"model checkpoint is unavailable: {self.checkpoint_path}")
        if self.artifact_fingerprint is not None and not _fingerprint(self.artifact_fingerprint):
            raise BenchmarkError("artifact fingerprint is invalid")
        if self.checkpoint_model_version is not None and (not isinstance(self.checkpoint_model_version, str) or not self.checkpoint_model_version.strip()):
            raise BenchmarkError("checkpoint_model_version is invalid")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path | str, *, model_version: str, tokenizer_version: int = 1) -> "BenchmarkModelSpec":
        path = Path(checkpoint_path).expanduser()
        try:
            checkpoint_info = CheckpointManager(path.parent).inspect(path)
        except Exception as exc:
            raise BenchmarkError(f"unable to inspect benchmark checkpoint: {path}") from exc
        return cls(model_version, model_identity_from_checkpoint(path, model_version=model_version, tokenizer_version=tokenizer_version), path, checkpoint_model_version=checkpoint_info.metadata.model_version)

    @classmethod
    def from_artifact(cls, artifact: ModelArtifact) -> "BenchmarkModelSpec":
        if not isinstance(artifact, ModelArtifact):
            raise BenchmarkError("candidate model must be a ModelArtifact")
        artifact.assert_valid()
        identity = ModelIdentity(artifact.metadata.base_model.model_name, artifact.model_version, str(artifact.checkpoint_path), artifact.metadata.checkpoint_fingerprint, artifact.metadata.base_model.tokenizer_version)
        return cls(artifact.model_version, identity, artifact.checkpoint_path, artifact.fingerprint, artifact.model_id, artifact.metadata.model_version)

    def to_dict(self) -> dict[str, Any]:
        return {"model_version": self.model_version, "model_identity": self.model_identity.to_dict(), "checkpoint_path": str(self.checkpoint_path), "artifact_fingerprint": self.artifact_fingerprint, "model_artifact_id": self.model_artifact_id, "checkpoint_model_version": self.checkpoint_model_version}


class BenchmarkRuntimeFactory(Protocol):
    def create(self, model: BenchmarkModelSpec, protocol: BenchmarkProtocolConfig) -> BenchmarkRuntime:
        """Create a runtime with no access to the other comparison arm."""


class FodciBenchmarkRuntimeFactory:
    """Adapter for the existing local Fodci inference and read-only tools."""

    def create(self, model: BenchmarkModelSpec, protocol: BenchmarkProtocolConfig) -> BenchmarkRuntime:
        config = BaselineEvaluationConfig(seed=protocol.seed, temperature=protocol.temperature, max_tokens=protocol.max_tokens, max_iterations=protocol.max_iterations, timeout_seconds=protocol.timeout_seconds, tool_configuration=protocol.tool_version, store_path=None)
        return create_current_model_runtime(model.checkpoint_path, model_version=model.checkpoint_model_version or model.model_version, tokenizer_version=model.model_identity.tokenizer_version or 1, config=config)


@dataclass(frozen=True, slots=True)
class BenchmarkTaskResult:
    task_id: str
    category: str
    difficulty: str
    status: str
    success: bool
    attempts: int | None
    tests_passed: int
    tests_failed: int
    total_tests: int
    tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    recovery_encountered: bool
    recovery_success: bool
    duration_seconds: float
    failure_reason: str | None
    final_state: Mapping[str, Any]
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("attempts", "tests_passed", "tests_failed", "total_tests", "tool_calls", "successful_tool_calls", "failed_tool_calls"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise BenchmarkError(f"{name} is invalid")
        if self.tests_passed + self.tests_failed != self.total_tests:
            raise BenchmarkError("test counters do not reconcile")
        if self.duration_seconds < 0:
            raise BenchmarkError("duration_seconds cannot be negative")
        object.__setattr__(self, "final_state", _freeze(self.final_state))
        object.__setattr__(self, "errors", tuple(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BenchmarkAggregate:
    total_tasks: int
    successful_tasks: int
    task_success_rate: float | None
    test_pass_rate: float | None
    tool_success_rate: float | None
    error_recovery_rate: float | None
    average_attempts: float | None
    failure_rate: float | None
    by_category: Mapping[str, Mapping[str, Any]]
    by_difficulty: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_category", _freeze(self.by_category))
        object.__setattr__(self, "by_difficulty", _freeze(self.by_difficulty))

    def to_dict(self) -> dict[str, Any]:
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    format: str
    protocol_version: str
    benchmark_version: str
    run_id: str
    model: BenchmarkModelSpec
    dataset_version: str
    dataset_fingerprint: str
    protocol: Mapping[str, Any]
    status: BenchmarkStatus
    task_results: tuple[BenchmarkTaskResult, ...]
    aggregate: BenchmarkAggregate
    raw_benchmark: Mapping[str, Any]
    metrics: Mapping[str, Any]
    environment: Mapping[str, Any]
    timestamp: str
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.format != BENCHMARK_FORMAT or self.protocol_version != BENCHMARK_PROTOCOL_VERSION or not self.run_id.strip():
            raise BenchmarkError("benchmark run identity is invalid")
        if self.benchmark_version != BENCHMARK_VERSION or not _fingerprint(self.dataset_fingerprint):
            raise BenchmarkError("benchmark run version/dataset identity is invalid")
        if not isinstance(self.model, BenchmarkModelSpec) or not isinstance(self.status, BenchmarkStatus):
            raise BenchmarkError("benchmark run model/status is invalid")
        object.__setattr__(self, "task_results", tuple(sorted(self.task_results, key=lambda item: item.task_id)))
        for name in ("protocol", "raw_benchmark", "metrics", "environment"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise BenchmarkError(f"{name} must be a mapping")
            object.__setattr__(self, name, _freeze(value))

    def to_dict(self) -> dict[str, Any]:
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class MetricDelta:
    name: str
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    direction: MetricDirection
    classification: str

    def to_dict(self) -> dict[str, Any]:
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BenchmarkGroupComparison:
    group_name: str
    task_count_base: int
    task_count_candidate: int
    metrics: tuple[MetricDelta, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"group_name": self.group_name, "task_count_base": self.task_count_base, "task_count_candidate": self.task_count_candidate, "metrics": [item.to_dict() for item in self.metrics]}


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    format: str
    protocol_version: str
    comparison_id: str
    benchmark_version: str
    dataset_version: str
    dataset_fingerprint: str
    base_run_id: str
    candidate_run_id: str
    base_model: BenchmarkModelSpec
    candidate_model: BenchmarkModelSpec
    overall_metrics: tuple[MetricDelta, ...]
    by_category: tuple[BenchmarkGroupComparison, ...]
    by_difficulty: tuple[BenchmarkGroupComparison, ...]
    status: str
    warnings: tuple[str, ...] = ()
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.format != "fodci.benchmark_comparison" or self.protocol_version != BENCHMARK_PROTOCOL_VERSION:
            raise BenchmarkError("comparison identity is invalid")
        if not isinstance(self.base_model, BenchmarkModelSpec) or not isinstance(self.candidate_model, BenchmarkModelSpec):
            raise BenchmarkError("comparison model identities are invalid")
        if self.base_run_id == self.candidate_run_id or not _fingerprint(self.dataset_fingerprint):
            raise BenchmarkError("comparison run IDs/dataset identity are invalid")
        if self.status not in {"IMPROVED", "REGRESSED", "EQUIVALENT", "INCONCLUSIVE"}:
            raise BenchmarkError("comparison status is invalid")
        object.__setattr__(self, "overall_metrics", tuple(self.overall_metrics))
        object.__setattr__(self, "by_category", tuple(self.by_category))
        object.__setattr__(self, "by_difficulty", tuple(self.by_difficulty))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "protocol_version": self.protocol_version, "comparison_id": self.comparison_id, "benchmark_version": self.benchmark_version, "dataset_version": self.dataset_version, "dataset_fingerprint": self.dataset_fingerprint, "base_run_id": self.base_run_id, "candidate_run_id": self.candidate_run_id, "base_model": self.base_model.to_dict(), "candidate_model": self.candidate_model.to_dict(), "overall_metrics": [item.to_dict() for item in self.overall_metrics], "by_category": [item.to_dict() for item in self.by_category], "by_difficulty": [item.to_dict() for item in self.by_difficulty], "status": self.status, "warnings": list(self.warnings), "timestamp": self.timestamp}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


class BenchmarkRunStore:
    """Atomic append-only local store for raw model benchmark runs."""

    def __init__(self, path: Path | str | None = DEFAULT_BENCHMARK_STORE_PATH) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self._runs: dict[str, BenchmarkRun] = {}
        self._loaded_digest: str | None = None
        self.reload()

    def reload(self) -> None:
        if self.path is None:
            self._runs = {}
            self._loaded_digest = None
            return
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise BenchmarkError("benchmark run store must not use symlinks")
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            self._runs = {}
            self._loaded_digest = None
            return
        except OSError as exc:
            raise BenchmarkError("benchmark run store is unavailable") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
            if set(payload) != {"format", "protocol_version", "runs"} or payload["format"] != BENCHMARK_FORMAT or payload["protocol_version"] != BENCHMARK_PROTOCOL_VERSION:
                raise BenchmarkError("benchmark run store header is invalid")
            self._runs = {key: _run_from_dict(value) for key, value in sorted(payload["runs"].items())}
        except BenchmarkError:
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BenchmarkError("benchmark run store is malformed") from exc
        self._loaded_digest = _digest(raw)

    def list_runs(self) -> tuple[BenchmarkRun, ...]:
        return tuple(self._runs[key] for key in sorted(self._runs))

    def get(self, run_id: str) -> BenchmarkRun | None:
        return self._runs.get(run_id)

    def save(self, run: BenchmarkRun) -> BenchmarkRun:
        existing = self._runs.get(run.run_id)
        if existing is not None:
            if existing.to_json() == run.to_json():
                return existing
            raise BenchmarkConflictError("benchmark run ID already has different immutable results")
        self._runs[run.run_id] = run
        try:
            self._persist()
        except Exception:
            self._runs.pop(run.run_id, None)
            raise
        return run

    def _persist(self) -> None:
        if self.path is None:
            return
        if self.path.exists() and (self._loaded_digest is None or _digest(self.path.read_bytes()) != self._loaded_digest):
            raise BenchmarkConflictError("benchmark run store changed since it was loaded")
        payload = {"format": BENCHMARK_FORMAT, "protocol_version": BENCHMARK_PROTOCOL_VERSION, "runs": {key: value.to_dict() for key, value in sorted(self._runs.items())}}
        _atomic_write_json(self.path, payload)
        self._loaded_digest = _digest(self.path.read_bytes())


class BenchmarkComparisonStore:
    """Atomic local store for immutable Base-vs-Candidate comparisons."""

    def __init__(self, path: Path | str | None = DEFAULT_COMPARISON_STORE_PATH) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self._comparisons: dict[str, BenchmarkComparison] = {}
        self._loaded_digest: str | None = None
        self.reload()

    def reload(self) -> None:
        if self.path is None:
            self._comparisons = {}
            self._loaded_digest = None
            return
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            self._comparisons = {}
            self._loaded_digest = None
            return
        except OSError as exc:
            raise BenchmarkError("comparison store is unavailable") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
            if set(payload) != {"format", "protocol_version", "comparisons"} or payload["format"] != "fodci.benchmark_comparison" or payload["protocol_version"] != BENCHMARK_PROTOCOL_VERSION:
                raise BenchmarkError("comparison store header is invalid")
            self._comparisons = {key: _comparison_from_dict(value) for key, value in sorted(payload["comparisons"].items())}
        except BenchmarkError:
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BenchmarkError("comparison store is malformed") from exc
        self._loaded_digest = _digest(raw)

    def get(self, comparison_id: str) -> BenchmarkComparison | None:
        return self._comparisons.get(comparison_id)

    def list_comparisons(self) -> tuple[BenchmarkComparison, ...]:
        return tuple(self._comparisons[key] for key in sorted(self._comparisons))

    def save(self, comparison: BenchmarkComparison) -> BenchmarkComparison:
        existing = self._comparisons.get(comparison.comparison_id)
        if existing is not None:
            if existing.to_json() == comparison.to_json():
                return existing
            raise BenchmarkConflictError("comparison ID already has different immutable results")
        self._comparisons[comparison.comparison_id] = comparison
        try:
            if self.path is not None and self.path.exists() and (self._loaded_digest is None or _digest(self.path.read_bytes()) != self._loaded_digest):
                raise BenchmarkConflictError("comparison store changed since it was loaded")
            if self.path is not None:
                _atomic_write_json(self.path, {"format": "fodci.benchmark_comparison", "protocol_version": BENCHMARK_PROTOCOL_VERSION, "comparisons": {key: value.to_dict() for key, value in sorted(self._comparisons.items())}})
                self._loaded_digest = _digest(self.path.read_bytes())
        except Exception:
            self._comparisons.pop(comparison.comparison_id, None)
            raise
        return comparison


class BenchmarkComparisonRunner:
    """Run both model arms under one immutable dataset and protocol."""

    def __init__(self, *, runtime_factory: BenchmarkRuntimeFactory, protocol: BenchmarkProtocolConfig | None = None, run_store: BenchmarkRunStore | None = None, comparison_store: BenchmarkComparisonStore | None = None) -> None:
        self.runtime_factory = runtime_factory
        self.protocol = protocol or BenchmarkProtocolConfig()
        self.run_store = run_store or BenchmarkRunStore(self.protocol.store_path)
        self.comparison_store = comparison_store or BenchmarkComparisonStore()

    def run(self, dataset: BenchmarkDataset, *, base_model: BenchmarkModelSpec, candidate_model: BenchmarkModelSpec, comparison_id: str, fixture_provider: Callable[[EvaluationTask, Path], None] | None = None, project_root: Path | str | None = None, training_dataset_fingerprint: str | None = None, training_source_record_ids: Sequence[str] = ()) -> BenchmarkComparison:
        dataset.validate_contamination(training_dataset_fingerprint=training_dataset_fingerprint, training_source_record_ids=training_source_record_ids)
        if base_model.model_version == candidate_model.model_version:
            raise BenchmarkError("base and candidate model versions must differ")
        base_run = self._run_one(dataset, base_model, f"{comparison_id}-base", fixture_provider=fixture_provider, project_root=project_root)
        candidate_run = self._run_one(dataset, candidate_model, f"{comparison_id}-candidate", fixture_provider=fixture_provider, project_root=project_root)
        comparison = build_comparison(comparison_id, base_run, candidate_run, dataset)
        return self.comparison_store.save(comparison)

    def _run_one(self, dataset: BenchmarkDataset, model: BenchmarkModelSpec, run_id: str, *, fixture_provider: Callable[[EvaluationTask, Path], None] | None, project_root: Path | str | None) -> BenchmarkRun:
        runtime = self.runtime_factory.create(model, self.protocol)
        config = BenchmarkConfig(max_tasks=len(dataset.tasks), max_task_wall_time=self.protocol.timeout_seconds, max_total_wall_time=self.protocol.timeout_seconds * len(dataset.tasks), benchmark_id=run_id, benchmark_version=BENCHMARK_PROTOCOL_VERSION, deterministic_mode=self.protocol.deterministic, fail_fast=False, continue_on_task_failure=True, cleanup_workspaces=True)
        result = EvidenceBenchmarkRunner().run(BenchmarkRequest(tasks=dataset.tasks, project_root=project_root, config=config, runtime=runtime, fixture_provider=fixture_provider))
        metrics = collect_metrics(result, dataset.tasks)
        run = _build_run(run_id, model, dataset, self.protocol, result, metrics)
        return self.run_store.save(run)


def load_benchmark_dataset(path: Path | str = DEFAULT_BENCHMARK_DATASET_PATH) -> BenchmarkDataset:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"benchmark dataset is unavailable or malformed: {path}") from exc
    required = {"format", "benchmark_version", "dataset_version", "protocol_version", "benchmark_only", "training_dataset_fingerprints", "source_record_ids", "dataset_fingerprint", "tasks"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise BenchmarkError("benchmark dataset fields are invalid")
    tasks = tuple(_task_from_dict(item) for item in payload["tasks"])
    dataset = BenchmarkDataset(payload["format"], payload["benchmark_version"], payload["dataset_version"], payload["protocol_version"], payload["benchmark_only"], tuple(payload["training_dataset_fingerprints"]), tuple(payload["source_record_ids"]), tasks, payload["dataset_fingerprint"])
    return dataset


def compute_benchmark_dataset_fingerprint(benchmark_version: str, dataset_version: str, protocol_version: str, tasks: Sequence[EvaluationTask]) -> str:
    payload = {"format": BENCHMARK_DATASET_FORMAT, "benchmark_version": benchmark_version, "dataset_version": dataset_version, "protocol_version": protocol_version, "benchmark_only": True, "tasks": [task.to_dict() for task in sorted(tasks, key=lambda item: item.task_id)]}
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_comparison(comparison_id: str, base_run: BenchmarkRun, candidate_run: BenchmarkRun, dataset: BenchmarkDataset) -> BenchmarkComparison:
    if base_run.dataset_fingerprint != candidate_run.dataset_fingerprint or base_run.dataset_fingerprint != dataset.dataset_fingerprint:
        raise BenchmarkContaminationError("base and candidate did not use the same benchmark dataset")
    if base_run.protocol != candidate_run.protocol:
        raise BenchmarkError("base and candidate did not use the same benchmark protocol")
    overall = tuple(_metric_delta(name, _aggregate_value(base_run.aggregate, name), _aggregate_value(candidate_run.aggregate, name), _direction(name)) for name in ("task_success_rate", "test_pass_rate", "tool_success_rate", "error_recovery_rate", "average_attempts", "failure_rate"))
    categories = _group_comparisons(base_run.aggregate.by_category, candidate_run.aggregate.by_category)
    difficulties = _group_comparisons(base_run.aggregate.by_difficulty, candidate_run.aggregate.by_difficulty)
    classifications = [item.classification for item in overall if item.classification != "INCONCLUSIVE"]
    if not classifications:
        status = "INCONCLUSIVE"
    elif all(item == "IMPROVED" for item in classifications):
        status = "IMPROVED"
    elif all(item == "REGRESSED" for item in classifications):
        status = "REGRESSED"
    elif any(item == "REGRESSED" for item in classifications):
        status = "REGRESSED"
    elif any(item == "IMPROVED" for item in classifications):
        status = "IMPROVED"
    else:
        status = "EQUIVALENT"
    warnings: list[str] = []
    if base_run.status is not BenchmarkStatus.COMPLETED or candidate_run.status is not BenchmarkStatus.COMPLETED:
        warnings.append("one or both model runs are not fully completed")
    return BenchmarkComparison("fodci.benchmark_comparison", BENCHMARK_PROTOCOL_VERSION, comparison_id, dataset.benchmark_version, dataset.dataset_version, dataset.dataset_fingerprint, base_run.run_id, candidate_run.run_id, base_run.model, candidate_run.model, overall, categories, difficulties, status, tuple(warnings), _utc_now())


def render_comparison_report(comparison: BenchmarkComparison) -> str:
    lines = ["=" * 56, "FODCI MODEL BENCHMARK", "=" * 56, "", f"Benchmark: {comparison.benchmark_version}", f"Dataset: {comparison.dataset_version}", f"Base Model: {comparison.base_model.model_version}", f"Candidate Model: {comparison.candidate_model.model_version}", "", "Overall", "-" * 56, f"{'Metric':<24} {'Base':>12} {'Candidate':>12} {'Delta':>12}"]
    for item in comparison.overall_metrics:
        lines.append(f"{item.name:<24} {_format_value(item.baseline_value):>12} {_format_value(item.candidate_value):>12} {_format_delta(item.delta):>12}")
    for title, groups in (("By Category", comparison.by_category), ("By Difficulty", comparison.by_difficulty)):
        lines.extend(["", title, "-" * 56])
        for group in groups:
            lines.append(f"[{group.group_name}] base_tasks={group.task_count_base} candidate_tasks={group.task_count_candidate}")
            for item in group.metrics:
                lines.append(f"  {item.name:<22} {_format_value(item.baseline_value):>12} {_format_value(item.candidate_value):>12} {_format_delta(item.delta):>12}")
    lines.extend(["", "Result", "-" * 56, f"Comparison evidence status: {comparison.status}"])
    if comparison.warnings:
        lines.append("Warnings: " + "; ".join(comparison.warnings))
    lines.append("=" * 56)
    return "\n".join(lines)


def _build_run(run_id: str, model: BenchmarkModelSpec, dataset: BenchmarkDataset, protocol: BenchmarkProtocolConfig, result: BenchmarkResult, metrics: BenchmarkMetrics) -> BenchmarkRun:
    task_results = tuple(_task_result(run) for run in result.task_runs)
    aggregate = _aggregate(task_results)
    environment = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "device": "cpu", "runtime": "local", "tool_registry": protocol.tool_version}
    return BenchmarkRun(BENCHMARK_FORMAT, BENCHMARK_PROTOCOL_VERSION, dataset.benchmark_version, run_id, model, dataset.dataset_version, dataset.dataset_fingerprint, protocol.to_dict(), BenchmarkStatus.COMPLETED if result.status.value == "COMPLETED" else BenchmarkStatus.PARTIAL, task_results, aggregate, result.to_dict(), metrics.to_dict(), environment, _utc_now())


def _task_result(run: BenchmarkTaskRun) -> BenchmarkTaskResult:
    evidence = run.evidence
    budget = evidence.budget_state or {}
    test = evidence.test_result or {}
    total_tests = 1 if evidence.tests_executed else 0
    passed_tests = int(total_tests and str(test.get("status", "")).upper() in {"PASS", "PASSED", "COMPLETED"})
    tools = int(budget.get("tool_calls", 0) or 0)
    successful = int(budget.get("successful_tool_calls", 0) or 0)
    failed = int(budget.get("failed_tool_calls", max(0, tools - successful)) or 0)
    recovery = evidence.recovery_state or {}
    return BenchmarkTaskResult(run.task_id, run.category, run.difficulty, run.status.value, run.status is BenchmarkTaskStatus.PASSED, _int_or_none(budget.get("attempts")), passed_tests, total_tests - passed_tests, total_tests, tools, successful, failed, bool(recovery.get("encountered_failure", False)), bool(recovery.get("recovered", False)), float(evidence.duration_seconds), run.failure_information[0] if run.failure_information else None, {"execution_status": evidence.execution_status, "termination_reason": evidence.termination_reason, "tests_executed": evidence.tests_executed, "evidence_complete": evidence.evidence_complete}, run.failure_information)


def _aggregate(results: Sequence[BenchmarkTaskResult]) -> BenchmarkAggregate:
    total = len(results)
    successful = sum(item.success for item in results)
    test_total = sum(item.total_tests for item in results)
    test_passed = sum(item.tests_passed for item in results)
    tool_total = sum(item.tool_calls for item in results)
    tool_success = sum(item.successful_tool_calls for item in results)
    recovery_attempted = sum(item.recovery_encountered for item in results)
    recovery_success = sum(item.recovery_success for item in results)
    attempts = [float(item.attempts) for item in results if item.attempts is not None]
    by_category = _group_aggregate(results, "category")
    by_difficulty = _group_aggregate(results, "difficulty")
    return BenchmarkAggregate(total, successful, _rate(successful, total), _rate(test_passed, test_total), _rate(tool_success, tool_total), _rate(recovery_success, recovery_attempted), sum(attempts) / len(attempts) if attempts else None, _rate(total - successful, total), by_category, by_difficulty)


def _group_aggregate(results: Sequence[BenchmarkTaskResult], field: str) -> Mapping[str, Mapping[str, Any]]:
    groups: dict[str, list[BenchmarkTaskResult]] = {}
    for result in results:
        groups.setdefault(str(getattr(result, field)), []).append(result)
    return {key: _group_values(items) for key, items in sorted(groups.items())}


def _group_values(items: Sequence[BenchmarkTaskResult]) -> Mapping[str, Any]:
    total = len(items)
    test_total = sum(item.total_tests for item in items)
    tools = sum(item.tool_calls for item in items)
    successes = sum(item.successful_tool_calls for item in items)
    recovery_attempted = sum(item.recovery_encountered for item in items)
    recovery_success = sum(item.recovery_success for item in items)
    attempts = [float(item.attempts) for item in items if item.attempts is not None]
    return {"task_count": total, "task_success_rate": _rate(sum(item.success for item in items), total), "test_pass_rate": _rate(sum(item.tests_passed for item in items), test_total), "tool_success_rate": _rate(successes, tools), "error_recovery_rate": _rate(recovery_success, recovery_attempted), "average_attempts": sum(attempts) / len(attempts) if attempts else None, "failure_rate": _rate(sum(not item.success for item in items), total)}


def _group_comparisons(base: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Mapping[str, Any]]) -> tuple[BenchmarkGroupComparison, ...]:
    groups: list[BenchmarkGroupComparison] = []
    names = sorted(set(base) | set(candidate))
    metric_names = ("task_success_rate", "test_pass_rate", "tool_success_rate", "error_recovery_rate", "average_attempts", "failure_rate")
    for name in names:
        b = base.get(name, {})
        c = candidate.get(name, {})
        groups.append(BenchmarkGroupComparison(name, int(b.get("task_count", 0)), int(c.get("task_count", 0)), tuple(_metric_delta(metric, b.get(metric), c.get(metric), _direction(metric)) for metric in metric_names)))
    return tuple(groups)


def _metric_delta(name: str, baseline: Any, candidate: Any, direction: MetricDirection) -> MetricDelta:
    b = float(baseline) if isinstance(baseline, (int, float)) and not isinstance(baseline, bool) else None
    c = float(candidate) if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) else None
    delta = c - b if b is not None and c is not None else None
    if delta is None:
        classification = "INCONCLUSIVE"
    elif abs(delta) <= 1e-12:
        classification = "EQUIVALENT"
    elif (direction is MetricDirection.HIGHER_IS_BETTER and delta > 0) or (direction is MetricDirection.LOWER_IS_BETTER and delta < 0):
        classification = "IMPROVED"
    else:
        classification = "REGRESSED"
    return MetricDelta(name, b, c, delta, direction, classification)


def _aggregate_value(aggregate: BenchmarkAggregate, name: str) -> Any:
    return getattr(aggregate, name)


def _direction(name: str) -> MetricDirection:
    return MetricDirection.LOWER_IS_BETTER if name in {"average_attempts", "failure_rate"} else MetricDirection.HIGHER_IS_BETTER


def _run_from_dict(value: Mapping[str, Any]) -> BenchmarkRun:
    model_payload = value["model"]
    identity = ModelIdentity(**model_payload["model_identity"])
    model = BenchmarkModelSpec(model_payload["model_version"], identity, Path(model_payload["checkpoint_path"]), model_payload.get("artifact_fingerprint"), model_payload.get("model_artifact_id"), model_payload.get("checkpoint_model_version"))
    task_results = tuple(BenchmarkTaskResult(**item) for item in value["task_results"])
    aggregate_payload = value["aggregate"]
    aggregate = BenchmarkAggregate(**aggregate_payload)
    return BenchmarkRun(value["format"], value["protocol_version"], value["benchmark_version"], value["run_id"], model, value["dataset_version"], value["dataset_fingerprint"], value["protocol"], BenchmarkStatus(value["status"]), task_results, aggregate, value["raw_benchmark"], value["metrics"], value["environment"], value["timestamp"], value.get("failure_reason"))


def _comparison_from_dict(value: Mapping[str, Any]) -> BenchmarkComparison:
    def model(payload: Mapping[str, Any]) -> BenchmarkModelSpec:
        return BenchmarkModelSpec(payload["model_version"], ModelIdentity(**payload["model_identity"]), Path(payload["checkpoint_path"]), payload.get("artifact_fingerprint"), payload.get("model_artifact_id"), payload.get("checkpoint_model_version"))
    def delta(payload: Mapping[str, Any]) -> MetricDelta:
        return MetricDelta(payload["name"], payload["baseline_value"], payload["candidate_value"], payload["delta"], MetricDirection(payload["direction"]), payload["classification"])
    def group(payload: Mapping[str, Any]) -> BenchmarkGroupComparison:
        return BenchmarkGroupComparison(payload["group_name"], payload["task_count_base"], payload["task_count_candidate"], tuple(delta(item) for item in payload["metrics"]))
    return BenchmarkComparison(value["format"], value["protocol_version"], value["comparison_id"], value["benchmark_version"], value["dataset_version"], value["dataset_fingerprint"], value["base_run_id"], value["candidate_run_id"], model(value["base_model"]), model(value["candidate_model"]), tuple(delta(item) for item in value["overall_metrics"]), tuple(group(item) for item in value["by_category"]), tuple(group(item) for item in value["by_difficulty"]), value["status"], tuple(value.get("warnings", ())), value.get("timestamp", ""))


def _fingerprint(value: str) -> bool:
    import re
    return isinstance(value, str) and bool(re.fullmatch(_FINGERPRINT_PATTERN, value))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _format_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _format_delta(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.3f}"


def _canonical_json(value: Any) -> str:
    return json.dumps(_serialize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if hasattr(value, "to_dict"):
        return _serialize(value.to_dict())
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BENCHMARK_DATASET_FORMAT",
    "BENCHMARK_DATASET_VERSION",
    "BENCHMARK_FORMAT",
    "BENCHMARK_PROTOCOL_VERSION",
    "BENCHMARK_VERSION",
    "BenchmarkAggregate",
    "BenchmarkComparison",
    "BenchmarkComparisonRunner",
    "BenchmarkComparisonStore",
    "BenchmarkContaminationError",
    "BenchmarkDataset",
    "BenchmarkError",
    "BenchmarkGroupComparison",
    "BenchmarkModelSpec",
    "BenchmarkProtocolConfig",
    "BenchmarkRun",
    "BenchmarkRunStore",
    "BenchmarkRuntimeFactory",
    "BenchmarkStatus",
    "BenchmarkTaskResult",
    "FodciBenchmarkRuntimeFactory",
    "MetricDelta",
    "MetricDirection",
    "build_comparison",
    "compute_benchmark_dataset_fingerprint",
    "load_benchmark_dataset",
    "render_comparison_report",
]
