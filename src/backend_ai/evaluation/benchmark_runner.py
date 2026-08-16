"""Bounded sequential benchmark execution over declarative EvaluationTask definitions.

Phase 8.2 deliberately collects execution evidence and does not score quality.
The runner delegates task execution to an explicitly supplied runtime adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import fnmatch
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from types import MappingProxyType
from typing import Any, Protocol

from backend_ai.evaluation.task_model import EvaluationTask, EvaluationTaskValidationResult, EvaluationTaskValidator


class BenchmarkTaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    TIMED_OUT = "TIMED_OUT"
    SKIPPED = "SKIPPED"
    UNAVAILABLE = "UNAVAILABLE"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    INCOMPLETE_EVIDENCE = "INCOMPLETE_EVIDENCE"


class BenchmarkStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


class BenchmarkTerminationReason(str, Enum):
    COMPLETED = "COMPLETED"
    FAIL_FAST = "FAIL_FAST"
    MAX_TASKS = "MAX_TASKS"
    WALL_TIME = "WALL_TIME"
    INVALID_REQUEST = "INVALID_REQUEST"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Host-controlled finite limits and artifact policy for one benchmark."""

    max_tasks: int = 32
    max_total_wall_time: float = 300.0
    max_task_wall_time: float = 60.0
    max_artifact_bytes: int = 65_536
    max_evidence_bytes: int = 65_536
    max_log_chars: int = 8_192
    fail_fast: bool = False
    continue_on_task_failure: bool = True
    collect_artifacts: bool = False
    environment_policy: str = "isolated-temporary-workspace"
    deterministic_mode: bool = True
    benchmark_id: str = "BENCHMARK-LOCAL"
    benchmark_version: str = "1.0"
    cleanup_workspaces: bool = True

    def __post_init__(self) -> None:
        for name in ("max_tasks", "max_artifact_bytes", "max_evidence_bytes", "max_log_chars"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_total_wall_time", "max_task_wall_time"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        if self.max_tasks > 10_000:
            raise ValueError("max_tasks exceeds benchmark safety ceiling")
        if self.max_artifact_bytes > 10_000_000 or self.max_evidence_bytes > 10_000_000:
            raise ValueError("artifact/evidence limit exceeds benchmark safety ceiling")
        for name in ("benchmark_id", "benchmark_version", "environment_policy"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must contain text")
        if not isinstance(self.fail_fast, bool) or not isinstance(self.continue_on_task_failure, bool):
            raise ValueError("fail_fast and continue_on_task_failure must be boolean")
        if not isinstance(self.collect_artifacts, bool) or not isinstance(self.deterministic_mode, bool) or not isinstance(self.cleanup_workspaces, bool):
            raise ValueError("artifact, deterministic, and cleanup flags must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BenchmarkEvidence:
    """Bounded raw evidence collected from one task run; it is not a score."""

    execution_started: bool = False
    execution_completed: bool = False
    execution_status: str = ""
    duration_seconds: float = 0.0
    termination_reason: str = ""
    workspace_identity: str = ""
    project_definition_identity: str = ""
    task_identity: str = ""
    cleanup_status: str = ""
    changed_paths: tuple[str, ...] = ()
    expected_paths_touched: tuple[str, ...] = ()
    unexpected_modifications: tuple[str, ...] = ()
    forbidden_changes_detected: tuple[str, ...] = ()
    mutation_count: int = 0
    mutation_verification: Mapping[str, Any] | None = None
    tests_requested: bool = False
    tests_executed: bool = False
    test_result: Mapping[str, Any] | None = None
    completion_evidence: Mapping[str, Any] | None = None
    final_verification_evidence: Mapping[str, Any] | None = None
    stop_condition_evidence: Mapping[str, Any] | None = None
    failure_information: tuple[str, ...] = ()
    recovery_state: Mapping[str, Any] | None = None
    budget_state: Mapping[str, Any] | None = None
    policy_safety_blocks: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    bounded_logs: tuple[str, ...] = ()
    evidence_complete: bool = True
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "changed_paths", "expected_paths_touched", "unexpected_modifications",
            "forbidden_changes_detected", "failure_information", "policy_safety_blocks",
            "artifacts", "bounded_logs", "warnings",
        ):
            object.__setattr__(self, name, _bounded_tuple(getattr(self, name), 256))
        for name in ("mutation_verification", "test_result", "completion_evidence", "final_verification_evidence", "stop_condition_evidence", "recovery_state", "budget_state"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping or None")
            if value is not None:
                object.__setattr__(self, name, _freeze_mapping(value))
        if self.mutation_count < 0:
            raise ValueError("mutation_count must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionResult:
    """Adapter result supplied by the existing bounded Fodci runtime."""

    status: BenchmarkTaskStatus
    execution_status: str = ""
    termination_reason: str = ""
    test_evidence: Mapping[str, Any] | None = None
    mutation_evidence: Mapping[str, Any] | None = None
    completion_evidence: Mapping[str, Any] | None = None
    final_verification_evidence: Mapping[str, Any] | None = None
    stop_condition_evidence: Mapping[str, Any] | None = None
    failure_information: tuple[str, ...] = ()
    recovery_state: Mapping[str, Any] | None = None
    budget_state: Mapping[str, Any] | None = None
    policy_safety_blocks: tuple[str, ...] = ()
    logs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    tests_requested: bool = False
    tests_executed: bool = False

    def __post_init__(self) -> None:
        for name in ("failure_information", "policy_safety_blocks", "logs", "warnings"):
            object.__setattr__(self, name, _bounded_tuple(getattr(self, name), 256))
        for name in ("test_evidence", "mutation_evidence", "completion_evidence", "final_verification_evidence", "stop_condition_evidence", "recovery_state", "budget_state"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise ValueError(f"{name} must be a mapping or None")
                object.__setattr__(self, name, _freeze_mapping(value))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


class BenchmarkRuntime(Protocol):
    """Explicit adapter for the existing bounded Fodci execution runtime."""

    def execute(self, task: EvaluationTask, workspace_root: Path, *, max_wall_time: float) -> BenchmarkExecutionResult:
        """Run one task through an existing runtime; do not implement a runner here."""


@dataclass(frozen=True, slots=True)
class BenchmarkTaskRun:
    task_id: str
    task_version: str
    category: str
    difficulty: str
    status: BenchmarkTaskStatus
    start_time: float | None
    end_time: float | None
    duration_seconds: float
    execution_result: BenchmarkExecutionResult | None
    evidence: BenchmarkEvidence
    validation: EvaluationTaskValidationResult
    failure_information: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_information", _bounded_tuple(self.failure_information, 256))
        object.__setattr__(self, "artifacts", _bounded_tuple(self.artifacts, 256))
        object.__setattr__(self, "warnings", _bounded_tuple(self.warnings, 256))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class BenchmarkRunSummary:
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    blocked_tasks: int
    timed_out_tasks: int
    skipped_tasks: int
    unavailable_tasks: int
    infrastructure_failures: int
    evidence_incomplete_tasks: int

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    benchmark_id: str
    benchmark_version: str
    status: BenchmarkStatus
    task_runs: tuple[BenchmarkTaskRun, ...]
    summary: BenchmarkRunSummary
    total_duration_seconds: float
    termination_reason: BenchmarkTerminationReason
    warnings: tuple[str, ...] = ()
    deterministic_metadata: Mapping[str, Any] | None = None
    fail_fast_enabled: bool = False
    fail_fast_triggered: bool = False
    fail_fast_task_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_runs", tuple(self.task_runs))
        object.__setattr__(self, "warnings", _bounded_tuple(self.warnings, 256))
        if self.deterministic_metadata is not None:
            object.__setattr__(self, "deterministic_metadata", _freeze_mapping(self.deterministic_metadata))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    tasks: tuple[EvaluationTask, ...]
    project_root: Path | None = None
    config: BenchmarkConfig = BenchmarkConfig()
    runtime: BenchmarkRuntime | None = None
    fixture_provider: Callable[[EvaluationTask, Path], None] | None = None
    baseline_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        if self.project_root is not None:
            object.__setattr__(self, "project_root", Path(self.project_root))
        if not isinstance(self.config, BenchmarkConfig):
            raise ValueError("config must be BenchmarkConfig")
        if self.baseline_metadata is not None and not isinstance(self.baseline_metadata, Mapping):
            raise ValueError("baseline_metadata must be a mapping or None")
        if self.baseline_metadata is not None:
            object.__setattr__(self, "baseline_metadata", _freeze_mapping(self.baseline_metadata))


class BenchmarkRunner:
    """Sequential, bounded orchestration layer over an explicit runtime adapter."""

    def __init__(self, *, validator: EvaluationTaskValidator | None = None) -> None:
        self.validator = validator or EvaluationTaskValidator()

    def run(self, request: BenchmarkRequest) -> BenchmarkResult:
        started = time.monotonic()
        config = request.config
        warnings: list[str] = []
        if not request.tasks:
            return self._empty_result(config, BenchmarkTerminationReason.INVALID_REQUEST, ("task collection must not be empty",))
        if len(request.tasks) > config.max_tasks:
            return self._empty_result(config, BenchmarkTerminationReason.INVALID_REQUEST, ("task collection exceeds max_tasks",))
        ids = [task.task_id if isinstance(task, EvaluationTask) else "<invalid>" for task in request.tasks]
        if len(set(ids)) != len(ids):
            return self._empty_result(config, BenchmarkTerminationReason.INVALID_REQUEST, ("task IDs must be unique",))
        if request.runtime is None:
            return self._empty_result(config, BenchmarkTerminationReason.INVALID_REQUEST, ("an explicit benchmark runtime adapter is required",))

        runs: list[BenchmarkTaskRun] = []
        fail_fast_triggered = False
        fail_fast_task_id: str | None = None
        termination = BenchmarkTerminationReason.COMPLETED
        for index, task in enumerate(request.tasks):
            elapsed = time.monotonic() - started
            if elapsed >= config.max_total_wall_time:
                termination = BenchmarkTerminationReason.WALL_TIME
                warnings.append("benchmark wall-time bound reached before remaining tasks")
                break
            run = self._run_task(request, task, index, started)
            runs.append(run)
            if config.fail_fast and run.status in _TERMINAL_FAILURES:
                fail_fast_triggered = True
                fail_fast_task_id = run.task_id
                termination = BenchmarkTerminationReason.FAIL_FAST
                break
            if not config.continue_on_task_failure and run.status in _TERMINAL_FAILURES:
                fail_fast_triggered = True
                fail_fast_task_id = run.task_id
                termination = BenchmarkTerminationReason.FAIL_FAST
                break

        if len(runs) < len(request.tasks) and termination is BenchmarkTerminationReason.COMPLETED:
            termination = BenchmarkTerminationReason.MAX_TASKS
        if len(runs) < len(request.tasks) and termination is not BenchmarkTerminationReason.INVALID_REQUEST:
            runs.extend(_skipped_run(task, "benchmark terminated before task started") for task in request.tasks[len(runs):])
        total_duration = time.monotonic() - started
        summary = _summary(runs)
        status = _benchmark_status(runs, len(request.tasks), termination)
        deterministic_metadata = {
            "task_order": [task.task_id for task in request.tasks],
            "project_root_supplied": request.project_root is not None,
            "deterministic_mode": config.deterministic_mode,
            "scoring": "NOT_IMPLEMENTED_IN_PHASE_8_2",
        }
        return BenchmarkResult(
            config.benchmark_id,
            config.benchmark_version,
            status,
            tuple(runs),
            summary,
            total_duration,
            termination,
            tuple(warnings),
            deterministic_metadata,
            config.fail_fast,
            fail_fast_triggered,
            fail_fast_task_id,
        )

    def _run_task(self, request: BenchmarkRequest, task: EvaluationTask, index: int, benchmark_started: float) -> BenchmarkTaskRun:
        config = request.config
        validation = self.validator.validate(task)
        if not validation.valid:
            failure = tuple(f"{issue.code}: {issue.message}" for issue in validation.errors)
            evidence = BenchmarkEvidence(task_identity=_task_identity(task), execution_status="VALIDATION_FAILED", failure_information=failure, evidence_complete=False, warnings=("task was not executed because validation failed",))
            return BenchmarkTaskRun(_task_id(task), _task_version(task), _task_enum(task, "category"), _task_enum(task, "difficulty"), BenchmarkTaskStatus.FAILED, None, None, 0.0, None, evidence, validation, failure, (), evidence.warnings)

        workspace: _Workspace | None = None
        before: Mapping[str, str] = {}
        task_started = time.monotonic()
        start_metadata = task_started
        try:
            workspace = _Workspace.create(index, task.task_id, request.project_root, config)
            before = _snapshot(workspace.path, config.max_evidence_bytes)
            if request.fixture_provider is not None:
                request.fixture_provider(task, workspace.path)
            fixture_after = _snapshot(workspace.path, config.max_evidence_bytes)
            result = request.runtime.execute(task, workspace.path, max_wall_time=min(config.max_task_wall_time, max(0.001, config.max_total_wall_time - (task_started - benchmark_started))))
            after = _snapshot(workspace.path, config.max_evidence_bytes)
            changed, expected, unexpected, forbidden = _path_evidence(task, before, after, fixture_after)
            end = time.monotonic()
            logs = _redact_and_bound(result.logs, config.max_log_chars)
            failure = _redact_and_bound(result.failure_information, config.max_log_chars)
            artifacts = _collect_artifacts(workspace.path, config) if config.collect_artifacts else ()
            evidence = BenchmarkEvidence(
                execution_started=True,
                execution_completed=True,
                execution_status=result.execution_status or result.status.value,
                duration_seconds=end - task_started,
                termination_reason=result.termination_reason or result.status.value,
                workspace_identity=workspace.identity,
                project_definition_identity=_project_identity(task),
                task_identity=_task_identity(task),
                cleanup_status="cleaned" if config.cleanup_workspaces else "preserved",
                changed_paths=changed,
                expected_paths_touched=expected,
                unexpected_modifications=unexpected,
                forbidden_changes_detected=forbidden,
                mutation_count=len(changed),
                mutation_verification=result.mutation_evidence,
                tests_requested=result.tests_requested,
                tests_executed=result.tests_executed,
                test_result=result.test_evidence,
                completion_evidence=result.completion_evidence,
                final_verification_evidence=result.final_verification_evidence,
                stop_condition_evidence=result.stop_condition_evidence,
                failure_information=failure,
                recovery_state=result.recovery_state,
                budget_state=result.budget_state,
                policy_safety_blocks=_redact_and_bound(result.policy_safety_blocks, config.max_log_chars),
                artifacts=artifacts,
                bounded_logs=logs,
                evidence_complete=not bool(result.warnings and any("incomplete" in item.lower() for item in result.warnings)),
                warnings=_redact_and_bound(result.warnings, config.max_log_chars),
            )
            final_status = result.status
            if forbidden or unexpected:
                final_status = BenchmarkTaskStatus.INCOMPLETE_EVIDENCE
            return BenchmarkTaskRun(task.task_id, task.version, _task_enum(task, "category"), _task_enum(task, "difficulty"), final_status, start_metadata, end, end - task_started, result, evidence, validation, failure, artifacts, evidence.warnings)
        except TimeoutError as exc:
            end = time.monotonic()
            message = _redact(str(exc))
            evidence = BenchmarkEvidence(True, False, "TIMED_OUT", end - task_started, "TASK_WALL_TIME", workspace.identity if workspace else "", _project_identity(task), _task_identity(task), "not_started" if workspace is None else ("cleaned" if config.cleanup_workspaces else "preserved"), failure_information=(message,), evidence_complete=False)
            return BenchmarkTaskRun(task.task_id, task.version, _task_enum(task, "category"), _task_enum(task, "difficulty"), BenchmarkTaskStatus.TIMED_OUT, start_metadata, end, end - task_started, None, evidence, validation, (message,), (), (message,))
        except Exception as exc:
            end = time.monotonic()
            message = _redact(f"{type(exc).__name__}: {exc}")
            evidence = BenchmarkEvidence(True, False, "INFRASTRUCTURE_ERROR", end - task_started, "RUNTIME_EXCEPTION", workspace.identity if workspace else "", _project_identity(task), _task_identity(task), "not_started" if workspace is None else ("cleaned" if config.cleanup_workspaces else "preserved"), failure_information=(message,), evidence_complete=False)
            return BenchmarkTaskRun(task.task_id, task.version, _task_enum(task, "category"), _task_enum(task, "difficulty"), BenchmarkTaskStatus.INFRASTRUCTURE_ERROR, start_metadata, end, end - task_started, None, evidence, validation, (message,), (), (message,))
        finally:
            if workspace is not None:
                workspace.cleanup(config.cleanup_workspaces)

    @staticmethod
    def _empty_result(config: BenchmarkConfig, reason: BenchmarkTerminationReason, warnings: tuple[str, ...]) -> BenchmarkResult:
        return BenchmarkResult(config.benchmark_id, config.benchmark_version, BenchmarkStatus.FAILED, (), BenchmarkRunSummary(0, 0, 0, 0, 0, 0, 0, 0, 0), 0.0, reason, warnings, {"scoring": "NOT_IMPLEMENTED_IN_PHASE_8_2"}, config.fail_fast, False, None)


@dataclass
class _Workspace:
    path: Path
    identity: str
    _preserve: bool = False

    @classmethod
    def create(cls, index: int, task_id: str, project_root: Path | None, config: BenchmarkConfig) -> "_Workspace":
        base = Path(tempfile.mkdtemp(prefix="fodci-benchmark-"))
        path = base / f"task-{index + 1:03d}-{_safe_name(task_id)}"
        path.mkdir()
        if project_root is not None:
            source = Path(project_root)
            if not source.is_dir():
                shutil.rmtree(base, ignore_errors=True)
                raise FileNotFoundError(f"benchmark project_root is not a directory: {source}")
            shutil.copytree(source, path, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))
        return cls(path, f"task-{index + 1:03d}-{task_id}")

    def cleanup(self, cleanup: bool) -> None:
        if cleanup and not self._preserve:
            shutil.rmtree(self.path.parent, ignore_errors=True)


_TERMINAL_FAILURES = {
    BenchmarkTaskStatus.FAILED,
    BenchmarkTaskStatus.BLOCKED,
    BenchmarkTaskStatus.TIMED_OUT,
    BenchmarkTaskStatus.UNAVAILABLE,
    BenchmarkTaskStatus.INFRASTRUCTURE_ERROR,
    BenchmarkTaskStatus.INCOMPLETE_EVIDENCE,
}


def _skipped_run(task: EvaluationTask, reason: str) -> BenchmarkTaskRun:
    validation = EvaluationTaskValidator().validate(task)
    evidence = BenchmarkEvidence(
        execution_status="SKIPPED",
        termination_reason="BENCHMARK_TERMINATED",
        workspace_identity=f"task-skipped-{task.task_id}",
        project_definition_identity=_project_identity(task),
        task_identity=_task_identity(task),
        cleanup_status="not_started",
        failure_information=(reason,),
        evidence_complete=False,
        warnings=(reason,),
    )
    return BenchmarkTaskRun(
        task.task_id,
        task.version,
        _task_enum(task, "category"),
        _task_enum(task, "difficulty"),
        BenchmarkTaskStatus.SKIPPED,
        None,
        None,
        0.0,
        None,
        evidence,
        validation,
        (),
        (),
        evidence.warnings,
    )


def _summary(runs: Sequence[BenchmarkTaskRun]) -> BenchmarkRunSummary:
    statuses = [run.status for run in runs]
    return BenchmarkRunSummary(
        len(runs),
        statuses.count(BenchmarkTaskStatus.PASSED),
        statuses.count(BenchmarkTaskStatus.FAILED),
        statuses.count(BenchmarkTaskStatus.BLOCKED),
        statuses.count(BenchmarkTaskStatus.TIMED_OUT),
        statuses.count(BenchmarkTaskStatus.SKIPPED),
        statuses.count(BenchmarkTaskStatus.UNAVAILABLE),
        statuses.count(BenchmarkTaskStatus.INFRASTRUCTURE_ERROR),
        statuses.count(BenchmarkTaskStatus.INCOMPLETE_EVIDENCE),
    )


def _benchmark_status(runs: Sequence[BenchmarkTaskRun], requested: int, reason: BenchmarkTerminationReason) -> BenchmarkStatus:
    if reason is BenchmarkTerminationReason.WALL_TIME:
        return BenchmarkStatus.TIMED_OUT
    if reason is BenchmarkTerminationReason.INVALID_REQUEST:
        return BenchmarkStatus.FAILED
    if reason in {BenchmarkTerminationReason.FAIL_FAST, BenchmarkTerminationReason.MAX_TASKS}:
        return BenchmarkStatus.PARTIAL if runs else BenchmarkStatus.FAILED
    if not runs:
        return BenchmarkStatus.FAILED
    statuses = [run.status for run in runs]
    if all(status is BenchmarkTaskStatus.PASSED for status in statuses) and len(runs) == requested:
        return BenchmarkStatus.COMPLETED
    if all(status is BenchmarkTaskStatus.BLOCKED for status in statuses):
        return BenchmarkStatus.BLOCKED
    if all(status in _TERMINAL_FAILURES for status in statuses):
        return BenchmarkStatus.FAILED
    return BenchmarkStatus.PARTIAL


def _path_evidence(task: EvaluationTask, before: Mapping[str, str], after: Mapping[str, str], fixture_after: Mapping[str, str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    # Fixture/template materialization is initial state, not an agent mutation.
    baseline = fixture_after if fixture_after else before
    changed = tuple(sorted(path for path in set(baseline) | set(after) if baseline.get(path) != after.get(path)))
    expected_patterns = tuple(area_path for area in task.expected_areas for area_path in area.paths)
    allowed = tuple(task.allowed_scope.allowed_files) + tuple(task.allowed_scope.allowed_directories) + tuple(task.allowed_scope.allowed_patterns)
    forbidden = tuple(task.allowed_scope.forbidden_paths) + tuple(task.allowed_scope.forbidden_patterns) + tuple(pattern for item in task.forbidden_changes for pattern in item.paths + item.patterns)
    expected = tuple(path for path in changed if _matches_any(path, expected_patterns))
    unexpected = tuple(path for path in changed if allowed and not _matches_any(path, allowed))
    forbidden_hits = tuple(path for path in changed if _matches_any(path, forbidden))
    return changed, expected, unexpected, forbidden_hits


def _snapshot(root: Path, max_bytes: int) -> dict[str, str]:
    result: dict[str, str] = {}
    total = 0
    if not root.exists():
        return result
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(root)
        relative = relative_path.as_posix()
        if any(part in {".git", ".pytest_cache", "__pycache__"} for part in relative_path.parts):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        remaining = max_bytes - total
        if remaining <= 0:
            break
        bounded = data[:remaining]
        result[relative] = bounded.hex()
        total += len(bounded)
    return result


def _collect_artifacts(root: Path, config: BenchmarkConfig) -> tuple[str, ...]:
    artifacts: list[str] = []
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if total + size > config.max_artifact_bytes:
            break
        artifacts.append(relative)
        total += size
    return tuple(artifacts)


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern.replace("\\", "/")) or normalized.startswith(pattern.rstrip("/") + "/") for pattern in patterns)


def _task_id(task: object) -> str:
    return task.task_id if isinstance(task, EvaluationTask) else "<invalid>"


def _task_version(task: object) -> str:
    return task.version if isinstance(task, EvaluationTask) else ""


def _task_enum(task: object, name: str) -> str:
    if not isinstance(task, EvaluationTask):
        return ""
    value = getattr(task, name)
    return getattr(value, "value", str(value))


def _task_identity(task: object) -> str:
    if not isinstance(task, EvaluationTask):
        return "<invalid>"
    return f"{task.task_id}@{task.version}"


def _project_identity(task: object) -> str:
    if not isinstance(task, EvaluationTask):
        return "<invalid>"
    definition = task.project_definition
    return json.dumps(definition.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "task"


def _redact(value: str) -> str:
    if not isinstance(value, str):
        return "<NON_TEXT>"
    value = re.sub(r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", "<REDACTED_PRIVATE_KEY>", value, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"(?i)(password|token|api[_-]?key|secret|credential|authorization)\s*[:=]\s*[^\s,;]+", r"\1=<REDACTED>", value)


def _redact_and_bound(values: Sequence[str], max_chars: int) -> tuple[str, ...]:
    result: list[str] = []
    used = 0
    for item in values:
        redacted = _redact(str(item))
        if used + len(redacted) > max_chars:
            break
        result.append(redacted)
        used += len(redacted)
    return tuple(result)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))})


def _bounded_tuple(value: Sequence[Any], limit: int) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        value = tuple(value)
    return value[:limit]


def _serialize_dataclass(value: object) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in value.__dataclass_fields__:  # type: ignore[attr-defined]
        result[name] = _serialize(getattr(value, name))
    return result


def _serialize(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


__all__ = [
    "BenchmarkConfig",
    "BenchmarkEvidence",
    "BenchmarkExecutionResult",
    "BenchmarkRequest",
    "BenchmarkResult",
    "BenchmarkRunSummary",
    "BenchmarkRunner",
    "BenchmarkRuntime",
    "BenchmarkStatus",
    "BenchmarkTaskRun",
    "BenchmarkTaskStatus",
    "BenchmarkTerminationReason",
]
