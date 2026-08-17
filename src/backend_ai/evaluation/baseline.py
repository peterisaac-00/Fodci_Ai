"""Phase 11.1 reproducible baseline evaluation for the current Fodci Agent.

This module evaluates the existing bounded Agent/runtime before any fine-tuning.
It reuses the declarative EvaluationTask model, BenchmarkRunner evidence, and
Phase 8.5 metrics.  It never changes model weights or training state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

from backend_ai.agent.autonomous_tool_loop import (
    AutonomousLoopConfig,
    AutonomousLoopRequest,
    AutonomousToolLoop,
    LoopStatus,
)
from backend_ai.agent.execution_budget import ExecutionBudget
from backend_ai.agent.registry import ToolRegistry
from backend_ai.evaluation.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkExecutionResult,
    BenchmarkRequest,
    BenchmarkResult,
    BenchmarkRuntime,
    BenchmarkRunner,
    BenchmarkTaskRun,
    BenchmarkTaskStatus,
)
from backend_ai.evaluation.metrics import MetricStatus, collect_metrics
from backend_ai.evaluation.task_model import (
    AllowedScope,
    EvaluationConstraint,
    EvaluationDifficulty,
    EvaluationTask,
    EvaluationTaskCategory,
    EvaluationTaskValidator,
    ExpectedArea,
    ExpectedAreaType,
    ExpectedBehavior,
    ForbiddenChange,
    ForbiddenChangeType,
    GroundTruth,
    ProjectDefinition,
    Requirement,
    SuccessCriterion,
    SuccessCriterionType,
)
from backend_ai.tools.test_runner import TestRunResult


BASELINE_EVALUATION_PROTOCOL_VERSION = "11.1"
BASELINE_EVALUATION_FORMAT = "fodci.baseline_evaluation"
BASELINE_DATASET_FORMAT = "fodci.backend_evaluation_dataset"
BASELINE_DATASET_VERSION = "evaluation-v1"
DEFAULT_BASELINE_DATASET_PATH = Path(__file__).with_name("datasets") / "phase111_backend_tasks.json"
DEFAULT_BASELINE_STORE_PATH = Path("artifacts") / "evaluation" / "baseline_runs.json"


class BaselineEvaluationError(ValueError):
    """Invalid evaluation input, result, or configuration."""


class BaselineEvaluationConflictError(BaselineEvaluationError):
    """Raised when a historical evaluation ID would be overwritten."""


class BaselineEvaluationStatus(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class BaselineEvaluationConfig:
    """Finite reproducibility and persistence settings for one baseline run."""

    protocol_version: str = BASELINE_EVALUATION_PROTOCOL_VERSION
    seed: int = 2026
    temperature: float = 1.0
    max_tokens: int = 32
    max_iterations: int = 16
    timeout_seconds: float = 60.0
    tool_configuration: str = "ToolRegistry.default"
    store_path: Path | str | None = DEFAULT_BASELINE_STORE_PATH

    def __post_init__(self) -> None:
        if self.protocol_version != BASELINE_EVALUATION_PROTOCOL_VERSION:
            raise BaselineEvaluationError("unsupported baseline evaluation protocol")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise BaselineEvaluationError("seed must be a non-negative integer")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) or float(self.temperature) <= 0:
            raise BaselineEvaluationError("temperature must be positive")
        for name in ("max_tokens", "max_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise BaselineEvaluationError(f"{name} must be a positive integer")
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(self.timeout_seconds, bool) or float(self.timeout_seconds) <= 0:
            raise BaselineEvaluationError("timeout_seconds must be positive")
        if not isinstance(self.tool_configuration, str) or not self.tool_configuration.strip():
            raise BaselineEvaluationError("tool_configuration must contain text")
        if self.store_path is not None:
            object.__setattr__(self, "store_path", Path(self.store_path).expanduser())

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_version": self.protocol_version, "seed": self.seed, "temperature": self.temperature, "max_tokens": self.max_tokens, "max_iterations": self.max_iterations, "timeout_seconds": self.timeout_seconds, "tool_configuration": self.tool_configuration, "store_path": str(self.store_path) if self.store_path is not None else None}


@dataclass(frozen=True, slots=True)
class BaselineEvaluationDataset:
    """Evaluation-only task set, deliberately separate from training datasets."""

    format: str
    dataset_version: str
    protocol_version: str
    evaluation_only: bool
    tasks: tuple[EvaluationTask, ...]
    dataset_fingerprint: str

    def __post_init__(self) -> None:
        if self.format != BASELINE_DATASET_FORMAT or not self.evaluation_only:
            raise BaselineEvaluationError("baseline dataset format/separation flag is invalid")
        if not re.fullmatch(r"evaluation-v[0-9]+(?:\.[0-9]+)?", self.dataset_version):
            raise BaselineEvaluationError("dataset_version must use evaluation-vN format")
        if self.protocol_version != BASELINE_EVALUATION_PROTOCOL_VERSION:
            raise BaselineEvaluationError("unsupported dataset protocol version")
        object.__setattr__(self, "tasks", tuple(sorted(self.tasks, key=lambda task: task.task_id)))
        if not self.tasks:
            raise BaselineEvaluationError("evaluation dataset must contain at least one task")
        validator = EvaluationTaskValidator()
        ids: set[str] = set()
        for task in self.tasks:
            if not isinstance(task, EvaluationTask):
                raise BaselineEvaluationError("evaluation dataset contains a non-task value")
            if task.task_id in ids:
                raise BaselineEvaluationError("evaluation dataset task IDs must be unique")
            ids.add(task.task_id)
            result = validator.validate(task)
            if not result.valid:
                raise BaselineEvaluationError(f"evaluation task {task.task_id} is invalid")
        expected = _dataset_fingerprint(self.dataset_version, self.protocol_version, self.tasks)
        if self.dataset_fingerprint != expected:
            raise BaselineEvaluationError("dataset_fingerprint does not match canonical evaluation tasks")

    @classmethod
    def from_tasks(cls, tasks: Sequence[EvaluationTask], *, dataset_version: str = BASELINE_DATASET_VERSION) -> "BaselineEvaluationDataset":
        ordered = tuple(sorted(tasks, key=lambda task: task.task_id))
        return cls(BASELINE_DATASET_FORMAT, dataset_version, BASELINE_EVALUATION_PROTOCOL_VERSION, True, ordered, _dataset_fingerprint(dataset_version, BASELINE_EVALUATION_PROTOCOL_VERSION, ordered))

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "dataset_version": self.dataset_version, "protocol_version": self.protocol_version, "evaluation_only": self.evaluation_only, "dataset_fingerprint": self.dataset_fingerprint, "tasks": [task.to_dict() for task in self.tasks]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def category_counts(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks:
            key = _enum_value(task.category)
            counts[key] = counts.get(key, 0) + 1
        return MappingProxyType(dict(sorted(counts.items())))

    @property
    def difficulty_counts(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks:
            key = _enum_value(task.difficulty)
            counts[key] = counts.get(key, 0) + 1
        return MappingProxyType(dict(sorted(counts.items())))


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Only model metadata that can be derived reliably from the runtime."""

    model_name: str
    model_version: str
    model_path: str | None
    model_fingerprint: str | None
    tokenizer_version: int | None

    def __post_init__(self) -> None:
        for name in ("model_name", "model_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise BaselineEvaluationError(f"{name} must contain text")
        if self.model_path is not None and not isinstance(self.model_path, str):
            raise BaselineEvaluationError("model_path must be text or None")
        if self.model_fingerprint is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", self.model_fingerprint):
            raise BaselineEvaluationError("model_fingerprint must use sha256 format")
        if self.tokenizer_version is not None and (not isinstance(self.tokenizer_version, int) or self.tokenizer_version < 0):
            raise BaselineEvaluationError("tokenizer_version must be a non-negative integer or None")

    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "model_version": self.model_version, "model_path": self.model_path, "model_fingerprint": self.model_fingerprint, "tokenizer_version": self.tokenizer_version}


@dataclass(frozen=True, slots=True)
class BaselineTaskResult:
    task_id: str
    category: str
    difficulty: str
    status: str
    success: bool
    tests_evaluated: bool
    tests_passed: int
    tests_failed: int
    total_tests: int
    test_pass_rate: float | None
    code_correctness_evaluated: bool
    code_correctness_passed: bool | None
    tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    tool_success_rate: float | None
    attempts: int | None
    recovery_encountered: bool
    recovery_success: bool
    duration_seconds: float
    failure_reason: str | None
    benchmark_run: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.tests_passed < 0 or self.tests_failed < 0 or self.total_tests < 0 or self.tool_calls < 0 or self.successful_tool_calls < 0 or self.failed_tool_calls < 0:
            raise BaselineEvaluationError("task result counters must be non-negative")
        if self.total_tests != self.tests_passed + self.tests_failed:
            raise BaselineEvaluationError("test counters do not reconcile")
        for value in (self.test_pass_rate, self.tool_success_rate):
            if value is not None and not 0.0 <= value <= 1.0:
                raise BaselineEvaluationError("rates must be within [0, 1]")
        object.__setattr__(self, "benchmark_run", _freeze(self.benchmark_run))

    def to_dict(self) -> dict[str, Any]:
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BaselineAggregateReport:
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    task_success_rate: float | None
    tests_passed: int
    tests_failed: int
    total_tests: int
    test_pass_rate: float | None
    successful_tool_operations: int
    failed_tool_operations: int
    total_tool_operations: int
    tool_success_rate: float | None
    recovery_attempted_tasks: int
    recovery_successful_tasks: int
    recovery_success_rate: float | None
    code_correctness_evaluated_tasks: int
    code_correctness_passed_tasks: int
    code_correctness_rate: float | None
    average_attempts: float | None
    average_duration_seconds: float | None
    failure_rate: float | None
    failure_reasons: Mapping[str, int]
    success_rate_by_category: Mapping[str, float | None]
    success_rate_by_difficulty: Mapping[str, float | None]

    def __post_init__(self) -> None:
        for name in ("failure_reasons", "success_rate_by_category", "success_rate_by_difficulty"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BaselineEvaluationRun:
    format: str
    evaluation_id: str
    model_identity: ModelIdentity
    agent_version: str
    dataset_version: str
    dataset_fingerprint: str
    evaluation_protocol_version: str
    timestamp: str
    configuration: Mapping[str, Any]
    status: BaselineEvaluationStatus
    task_results: tuple[BaselineTaskResult, ...]
    aggregate: BaselineAggregateReport
    benchmark_result: Mapping[str, Any]
    environment: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.format != BASELINE_EVALUATION_FORMAT or not isinstance(self.evaluation_id, str) or not self.evaluation_id.strip():
            raise BaselineEvaluationError("baseline evaluation identity is invalid")
        if not isinstance(self.model_identity, ModelIdentity) or self.evaluation_protocol_version != BASELINE_EVALUATION_PROTOCOL_VERSION:
            raise BaselineEvaluationError("baseline evaluation metadata is invalid")
        object.__setattr__(self, "task_results", tuple(sorted(self.task_results, key=lambda item: item.task_id)))
        object.__setattr__(self, "configuration", _freeze(self.configuration))
        object.__setattr__(self, "benchmark_result", _freeze(self.benchmark_result))
        object.__setattr__(self, "environment", _freeze(self.environment))

    def to_dict(self) -> dict[str, Any]:
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class AutonomousToolLoopBenchmarkRuntime:
    """Explicit adapter that evaluates the existing AgentLoop and tools."""

    def __init__(self, engine: Any, *, registry: ToolRegistry | None = None, config: AutonomousLoopConfig | None = None) -> None:
        self.registry = registry or ToolRegistry.default()
        self.loop = AutonomousToolLoop(engine, registry=self.registry, config=config)

    def execute(self, task: EvaluationTask, workspace_root: Path, *, max_wall_time: float) -> BenchmarkExecutionResult:
        request = AutonomousLoopRequest(task=task.user_intent, project_root=workspace_root)
        result = self.loop.run(request)
        calls = tuple(result.tool_calls)
        tool_results = tuple(result.tool_results)
        tests = [(call, tool_result) for call, tool_result in zip(calls, tool_results) if call.name == "run_tests"]
        tests_requested = bool(tests)
        test_evidence = _test_evidence(tests)
        recovery_state = _recovery_evidence(result)
        budget_state = result.execution_budget.to_dict() if result.execution_budget is not None else {"steps": result.usage.steps, "tool_calls": result.usage.tool_calls}
        successful_tools = sum(1 for item in tool_results if item.success)
        failed_tools = len(tool_results) - successful_tools
        budget_state = dict(budget_state)
        budget_state.update({"tool_calls": len(tool_results), "successful_tool_calls": successful_tools, "failed_tool_calls": failed_tools, "attempts": result.usage.steps})
        status = _benchmark_status(result.status)
        failure_information = tuple(result.errors) + tuple(result.warnings)
        return BenchmarkExecutionResult(
            status=status,
            execution_status=result.status.value,
            termination_reason=_termination_reason(result),
            test_evidence=test_evidence,
            completion_evidence=result.completion.to_dict() if result.completion else None,
            final_verification_evidence=result.final_verification.to_dict() if result.final_verification else None,
            stop_condition_evidence=result.stop_evaluation.to_dict() if result.stop_evaluation else None,
            failure_information=failure_information,
            recovery_state=recovery_state,
            budget_state=budget_state,
            policy_safety_blocks=tuple(result.errors),
            tests_requested=tests_requested,
            tests_executed=tests_requested,
        )


class BaselineEvaluationStore:
    """Atomic local historical store; evaluation IDs are never silently overwritten."""

    def __init__(self, path: Path | str | None = DEFAULT_BASELINE_STORE_PATH) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self._runs: dict[str, BaselineEvaluationRun] = {}
        self._loaded_digest: str | None = None
        self.reload()

    def reload(self) -> None:
        if self.path is None:
            self._runs = {}
            self._loaded_digest = None
            return
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise BaselineEvaluationError("baseline store must not use symlinks")
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            self._runs = {}
            self._loaded_digest = None
            return
        except OSError as exc:
            raise BaselineEvaluationError("baseline store is unavailable") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
            if set(payload) != {"format", "protocol_version", "runs"} or payload["format"] != BASELINE_EVALUATION_FORMAT or payload["protocol_version"] != BASELINE_EVALUATION_PROTOCOL_VERSION:
                raise BaselineEvaluationError("baseline store header is invalid")
            self._runs = {str(key): _run_from_dict(value) for key, value in sorted(payload["runs"].items())}
        except (TypeError, ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BaselineEvaluationError("baseline store is malformed") from exc
        self._loaded_digest = _digest(raw)

    def list_runs(self) -> tuple[BaselineEvaluationRun, ...]:
        return tuple(self._runs[key] for key in sorted(self._runs))

    def get(self, evaluation_id: str) -> BaselineEvaluationRun | None:
        return self._runs.get(evaluation_id)

    def save(self, run: BaselineEvaluationRun) -> BaselineEvaluationRun:
        if not isinstance(run, BaselineEvaluationRun):
            raise BaselineEvaluationError("store requires BaselineEvaluationRun")
        existing = self._runs.get(run.evaluation_id)
        if existing is not None:
            if existing.to_json() == run.to_json():
                return existing
            raise BaselineEvaluationConflictError("evaluation_id already exists with different results")
        self._runs[run.evaluation_id] = run
        try:
            self._persist()
        except Exception:
            self._runs.pop(run.evaluation_id, None)
            raise
        return run

    def _persist(self) -> None:
        if self.path is None:
            return
        if self.path.exists() and (self._loaded_digest is None or _digest(self.path.read_bytes()) != self._loaded_digest):
            raise BaselineEvaluationConflictError("baseline store changed since it was loaded")
        payload = json.dumps({"format": BASELINE_EVALUATION_FORMAT, "protocol_version": BASELINE_EVALUATION_PROTOCOL_VERSION, "runs": {key: self._runs[key].to_dict() for key in sorted(self._runs)}}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=directory, prefix=".baseline.", suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            self._loaded_digest = _digest(payload)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass


class BaselineEvaluationRunner:
    """Run one explicit baseline benchmark and aggregate objective evidence."""

    def __init__(self, *, runtime: BenchmarkRuntime, model_identity: ModelIdentity, agent_version: str = "0.1.0", config: BaselineEvaluationConfig | None = None, store: BaselineEvaluationStore | None = None) -> None:
        self.runtime = runtime
        self.model_identity = model_identity
        self.agent_version = agent_version
        self.config = config or BaselineEvaluationConfig()
        self.store = store if store is not None else BaselineEvaluationStore(self.config.store_path)

    def run(self, dataset: BaselineEvaluationDataset, *, evaluation_id: str, project_root: Path | str | None = None, timestamp: str = "") -> BaselineEvaluationRun:
        if not isinstance(dataset, BaselineEvaluationDataset):
            raise BaselineEvaluationError("dataset must be BaselineEvaluationDataset")
        if not evaluation_id.strip():
            raise BaselineEvaluationError("evaluation_id must contain text")
        benchmark = BenchmarkRunner().run(BenchmarkRequest(tasks=dataset.tasks, project_root=project_root, config=BenchmarkConfig(max_tasks=len(dataset.tasks), max_task_wall_time=self.config.timeout_seconds, max_total_wall_time=self.config.timeout_seconds * len(dataset.tasks), benchmark_id=evaluation_id, benchmark_version=self.config.protocol_version, deterministic_mode=True, fail_fast=False, continue_on_task_failure=True), runtime=self.runtime))
        metrics = collect_metrics(benchmark, dataset.tasks)
        task_results = tuple(_task_result(run) for run in benchmark.task_runs)
        aggregate = _aggregate(task_results, dataset.tasks)
        status = BaselineEvaluationStatus.COMPLETED if benchmark.termination_reason.value == "COMPLETED" and len(task_results) == len(dataset.tasks) else BaselineEvaluationStatus.PARTIAL if task_results else BaselineEvaluationStatus.FAILED
        run = BaselineEvaluationRun(BASELINE_EVALUATION_FORMAT, evaluation_id, self.model_identity, self.agent_version, dataset.dataset_version, dataset.dataset_fingerprint, self.config.protocol_version, timestamp or "NOT_RECORDED", self.config.to_dict() | {"dataset_category_counts": dict(dataset.category_counts), "dataset_difficulty_counts": dict(dataset.difficulty_counts), "metrics_available": [metric.name for metric in metrics.metrics if metric.status is MetricStatus.AVAILABLE]}, status, task_results, aggregate, benchmark.to_dict(), {"runtime": "local", "project_root_supplied": project_root is not None})
        return self.store.save(run)


def load_evaluation_dataset(path: Path | str = DEFAULT_BASELINE_DATASET_PATH) -> BaselineEvaluationDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != {"format", "dataset_version", "protocol_version", "evaluation_only", "dataset_fingerprint", "tasks"}:
        raise BaselineEvaluationError("evaluation dataset fields are invalid")
    tasks = tuple(_task_from_dict(item) for item in payload["tasks"])
    dataset = BaselineEvaluationDataset.from_tasks(tasks, dataset_version=payload["dataset_version"])
    if payload["format"] != dataset.format or payload["protocol_version"] != dataset.protocol_version or payload["evaluation_only"] is not True or payload["dataset_fingerprint"] != dataset.dataset_fingerprint:
        raise BaselineEvaluationError("evaluation dataset header or fingerprint is invalid")
    return dataset


def create_current_model_runtime(
    checkpoint_path: Path | str,
    *,
    model_version: str,
    tokenizer_version: int,
    config: BaselineEvaluationConfig | None = None,
    registry: ToolRegistry | None = None,
) -> AutonomousToolLoopBenchmarkRuntime:
    """Load the current local Fodci model and expose it through an explicit runtime."""

    active = config or BaselineEvaluationConfig()
    from backend_ai.inference import InferenceConfig, InferenceEngine
    from backend_ai.model import FodciModel
    from backend_ai.tokenizer import FodciTokenizer

    inference = InferenceConfig(
        max_new_tokens=active.max_tokens,
        temperature=active.temperature,
        do_sample=False,
        stop_on_eos=True,
        device="cpu",
        seed=active.seed,
        model_version=model_version,
        tokenizer_version=tokenizer_version,
        checkpoint_path=Path(checkpoint_path),
    )
    engine = InferenceEngine(FodciModel(), FodciTokenizer(), inference)
    budget = ExecutionBudget(
        max_iterations=active.max_iterations,
        max_tool_calls=active.max_iterations,
        max_wall_time_seconds=active.timeout_seconds,
        max_action_steps=active.max_iterations,
    )
    loop_config = AutonomousLoopConfig(execution_budget=budget)
    return AutonomousToolLoopBenchmarkRuntime(engine, registry=registry, config=loop_config)


def model_identity_from_checkpoint(checkpoint_path: Path | str, *, model_version: str, tokenizer_version: int) -> ModelIdentity:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise BaselineEvaluationError(f"checkpoint is unavailable: {path}")
    return ModelIdentity("FodciModel", model_version, str(path), "sha256:" + _file_sha256(path), tokenizer_version)


def _dataset_fingerprint(version: str, protocol: str, tasks: Sequence[EvaluationTask]) -> str:
    payload = {"format": BASELINE_DATASET_FORMAT, "dataset_version": version, "protocol_version": protocol, "evaluation_only": True, "tasks": [task.to_dict() for task in sorted(tasks, key=lambda item: item.task_id)]}
    return "sha256:" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _task_result(run: BenchmarkTaskRun) -> BaselineTaskResult:
    evidence = run.evidence
    budget = evidence.budget_state or {}
    test_data = evidence.test_result or {}
    total_tests = 1 if evidence.tests_executed else 0
    tests_passed = int(total_tests and str(test_data.get("status", "")).upper() in {"PASS", "PASSED", "COMPLETED"})
    tests_failed = total_tests - tests_passed
    tool_calls = int(budget.get("tool_calls", 0) or 0)
    successful_tools = int(budget.get("successful_tool_calls", 0) or 0)
    failed_tools = int(budget.get("failed_tool_calls", max(0, tool_calls - successful_tools)) or 0)
    recovery = evidence.recovery_state or {}
    recovery_encountered = bool(recovery.get("encountered_failure", False))
    recovery_success = bool(recovery.get("recovered", False)) if recovery_encountered else False
    attempts = _int_or_none(budget.get("attempts"))
    return BaselineTaskResult(run.task_id, run.category, run.difficulty, run.status.value, run.status is BenchmarkTaskStatus.PASSED, bool(evidence.tests_executed), tests_passed, tests_failed, total_tests, tests_passed / total_tests if total_tests else None, bool(evidence.tests_executed), bool(tests_passed) if total_tests else None, tool_calls, successful_tools, failed_tools, successful_tools / tool_calls if tool_calls else None, attempts, recovery_encountered, recovery_success, float(evidence.duration_seconds), run.failure_information[0] if run.failure_information else None, run.to_dict())


def _aggregate(results: Sequence[BaselineTaskResult], tasks: Sequence[EvaluationTask]) -> BaselineAggregateReport:
    total = len(results)
    successes = sum(item.success for item in results)
    tests_passed = sum(item.tests_passed for item in results)
    tests_failed = sum(item.tests_failed for item in results)
    total_tests = sum(item.total_tests for item in results)
    successful_tools = sum(item.successful_tool_calls for item in results)
    failed_tools = sum(item.failed_tool_calls for item in results)
    attempted_recovery = sum(item.recovery_encountered for item in results)
    successful_recovery = sum(item.recovery_success for item in results)
    code_evaluated = sum(item.code_correctness_evaluated for item in results)
    code_passed = sum(item.code_correctness_passed is True for item in results)
    attempts = [float(item.attempts) for item in results if item.attempts is not None]
    durations = [item.duration_seconds for item in results]
    failure_reasons: dict[str, int] = {}
    for item in results:
        if not item.success:
            reason = item.failure_reason or "UNSPECIFIED_FAILURE"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    by_category = _group_success(results, "category")
    by_difficulty = _group_success(results, "difficulty")
    return BaselineAggregateReport(total, successes, total - successes, _rate(successes, total), tests_passed, tests_failed, total_tests, _rate(tests_passed, total_tests), successful_tools, failed_tools, successful_tools + failed_tools, _rate(successful_tools, successful_tools + failed_tools), attempted_recovery, successful_recovery, _rate(successful_recovery, attempted_recovery), code_evaluated, code_passed, _rate(code_passed, code_evaluated), sum(attempts) / len(attempts) if attempts else None, sum(durations) / len(durations) if durations else None, _rate(total - successes, total), failure_reasons, by_category, by_difficulty)


def _group_success(results: Sequence[BaselineTaskResult], field: str) -> Mapping[str, float | None]:
    groups: dict[str, list[BaselineTaskResult]] = {}
    for item in results:
        groups.setdefault(str(getattr(item, field)), []).append(item)
    return {key: _rate(sum(item.success for item in values), len(values)) for key, values in sorted(groups.items())}


def _task_from_dict(value: Mapping[str, Any]) -> EvaluationTask:
    project = ProjectDefinition(**value.get("project_definition", {}))
    expected_behaviors = tuple(ExpectedBehavior(**item) for item in value.get("expected_behaviors", ()))
    requirements = tuple(Requirement(**item) for item in value.get("requirements", ()))
    areas = tuple(ExpectedArea(**item) for item in value.get("expected_areas", ()))
    criteria = tuple(SuccessCriterion(**item) for item in value.get("success_criteria", ()))
    forbidden = tuple(ForbiddenChange(**item) for item in value.get("forbidden_changes", ()))
    scope = AllowedScope(**value.get("allowed_scope", {}))
    constraints = EvaluationConstraint(**value.get("constraints", {}))
    ground_truth = GroundTruth(**value.get("ground_truth", {}))
    fields = {"task_id": value["task_id"], "title": value["title"], "description": value["description"], "version": value["version"], "category": value["category"], "difficulty": value["difficulty"], "project_definition": project, "user_intent": value["user_intent"], "expected_behaviors": expected_behaviors, "requirements": requirements, "allowed_scope": scope, "expected_areas": areas, "tests": (), "success_criteria": criteria, "forbidden_changes": forbidden, "constraints": constraints, "ground_truth": ground_truth, "metadata": value.get("metadata", {})}
    return EvaluationTask(**fields)


def _test_evidence(tests: Sequence[tuple[Any, Any]]) -> Mapping[str, Any] | None:
    if not tests:
        return None
    last = tests[-1][1].data
    if isinstance(last, TestRunResult):
        return {"status": "PASS" if last.exit_code == 0 and last.status.value == "COMPLETED" else "FAIL", "raw_status": last.status.value, "exit_code": last.exit_code}
    if isinstance(last, Mapping):
        return dict(last)
    return {"status": "FAIL", "raw_type": type(last).__name__}


def _recovery_evidence(result: Any) -> Mapping[str, Any] | None:
    if result.recovery is None:
        return {"encountered_failure": False, "recovered": False, "history_count": 0}
    history_count = len(result.recovery.history)
    encountered = history_count > 0
    return {"encountered_failure": encountered, "recovered": encountered and result.status is LoopStatus.COMPLETED, "history_count": history_count}


def _benchmark_status(status: LoopStatus) -> BenchmarkTaskStatus:
    if status is LoopStatus.COMPLETED:
        return BenchmarkTaskStatus.PASSED
    if status is LoopStatus.BLOCKED:
        return BenchmarkTaskStatus.BLOCKED
    if status in {LoopStatus.CONTEXT_LIMIT_REACHED, LoopStatus.LOOP_BOUND_REACHED}:
        return BenchmarkTaskStatus.TIMED_OUT
    return BenchmarkTaskStatus.FAILED


def _termination_reason(result: Any) -> str:
    if result.stop_evaluation is not None:
        reason = getattr(result.stop_evaluation, "reason", None)
        if reason is not None:
            return getattr(reason, "value", str(reason))
    return result.status.value


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _enum_value(value: Any) -> str:
    return getattr(value, "value", str(value))


def _canonical(value: Any) -> str:
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


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_from_dict(value: Mapping[str, Any]) -> BaselineEvaluationRun:
    identity = ModelIdentity(**value["model_identity"])
    aggregate = BaselineAggregateReport(**value["aggregate"])
    tasks = tuple(BaselineTaskResult(**item) for item in value["task_results"])
    status = BaselineEvaluationStatus(value["status"])
    return BaselineEvaluationRun(value["format"], value["evaluation_id"], identity, value["agent_version"], value["dataset_version"], value["dataset_fingerprint"], value["evaluation_protocol_version"], value["timestamp"], value["configuration"], status, tasks, aggregate, value["benchmark_result"], value["environment"])


__all__ = [
    "BASELINE_DATASET_FORMAT",
    "BASELINE_DATASET_VERSION",
    "BASELINE_EVALUATION_FORMAT",
    "BASELINE_EVALUATION_PROTOCOL_VERSION",
    "BaselineAggregateReport",
    "BaselineEvaluationConfig",
    "BaselineEvaluationConflictError",
    "BaselineEvaluationDataset",
    "BaselineEvaluationError",
    "BaselineEvaluationRun",
    "BaselineEvaluationStatus",
    "BaselineEvaluationStore",
    "BaselineEvaluationRunner",
    "BaselineTaskResult",
    "DEFAULT_BASELINE_DATASET_PATH",
    "DEFAULT_BASELINE_STORE_PATH",
    "ModelIdentity",
    "AutonomousToolLoopBenchmarkRuntime",
    "load_evaluation_dataset",
    "model_identity_from_checkpoint",
    "create_current_model_runtime",
]
