"""Bounded, deterministic, task-scoped short-term memory for one active task.

This module is deliberately a working-context layer, not a retrieval or persistence
system.  It stores bounded summaries of the current task and exposes immutable
snapshots for orchestration and evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


_MAX_COLLECTION_ITEMS = 64
_SECRET_KEY_RE = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"((?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie)\s*(?:=|:)\s*)([^,\s}\]]+|\"[^\"]*\"|'[^']*')",
    re.IGNORECASE,
)
_ENV_SECRET_RE = re.compile(
    r"(?im)^\s*(?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|database_url)\s*=\s*[^\n]*"
)
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE | re.DOTALL)


class MemoryLifecycle(str, Enum):
    """Lifecycle of one task-scoped memory owner."""

    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    UPDATED = "UPDATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CLOSED = "CLOSED"


class MemoryStatus(str, Enum):
    """Operational status of the memory subsystem."""

    ACTIVE = "ACTIVE"
    UPDATED = "UPDATED"
    CLOSED = "CLOSED"
    MEMORY_LIMIT_REACHED = "MEMORY_LIMIT_REACHED"
    MEMORY_INVALID = "MEMORY_INVALID"


class MemoryImportance(IntEnum):
    """Simple deterministic retention priority; this is not semantic ranking."""

    LOW = 0
    NORMAL = 1
    IMPORTANT = 2
    CRITICAL = 3


class MemoryInformationKind(str, Enum):
    AUTHORITATIVE = "AUTHORITATIVE"
    DERIVED = "DERIVED"


class ShortTermMemoryError(RuntimeError):
    """Base error for controlled short-term-memory operations."""


class MemoryClosedError(ShortTermMemoryError):
    """A write was attempted after task memory was closed."""


class MemoryValidationError(ShortTermMemoryError, ValueError):
    """An invalid memory input or configuration was rejected."""


class MemorySerializationError(ShortTermMemoryError):
    """A snapshot could not be represented within its configured bound."""


@dataclass(frozen=True, slots=True)
class ShortTermMemoryLimits:
    """Host-controlled bounds; callers cannot increase them through memory data."""

    max_observations: int = 32
    max_tool_records: int = 24
    max_failure_records: int = 12
    max_test_records: int = 12
    max_fix_records: int = 12
    max_verification_records: int = 12
    max_memory_entries: int = 96
    max_text_length_per_entry: int = 512
    max_total_memory_bytes: int = 32_768

    def __post_init__(self) -> None:
        integer_fields = (
            "max_observations",
            "max_tool_records",
            "max_failure_records",
            "max_test_records",
            "max_fix_records",
            "max_verification_records",
            "max_memory_entries",
            "max_text_length_per_entry",
            "max_total_memory_bytes",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise MemoryValidationError(f"{name} must be a positive integer")
        ceilings = {
            "max_observations": 256,
            "max_tool_records": 256,
            "max_failure_records": 128,
            "max_test_records": 128,
            "max_fix_records": 128,
            "max_verification_records": 128,
            "max_memory_entries": 512,
            "max_text_length_per_entry": 16_384,
            "max_total_memory_bytes": 1_048_576,
        }
        for name, ceiling in ceilings.items():
            if getattr(self, name) > ceiling:
                raise MemoryValidationError(f"{name} exceeds the safety ceiling")
        if self.max_memory_entries < 6:
            raise MemoryValidationError("max_memory_entries must leave room for all memory categories")
        if self.max_total_memory_bytes < 1_024:
            raise MemoryValidationError("max_total_memory_bytes must be at least 1024 bytes")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_observations": self.max_observations,
            "max_tool_records": self.max_tool_records,
            "max_failure_records": self.max_failure_records,
            "max_test_records": self.max_test_records,
            "max_fix_records": self.max_fix_records,
            "max_verification_records": self.max_verification_records,
            "max_memory_entries": self.max_memory_entries,
            "max_text_length_per_entry": self.max_text_length_per_entry,
            "max_total_memory_bytes": self.max_total_memory_bytes,
        }


@dataclass(frozen=True, slots=True)
class MemoryPlanState:
    """Bounded plan snapshot; the planner remains authoritative for execution."""

    plan: str
    current_step: str | None = None
    completed_steps: tuple[str, ...] = ()
    blocked_steps: tuple[str, ...] = ()
    failed_steps: tuple[str, ...] = ()
    next_step: str | None = None

    def __post_init__(self) -> None:
        for name in ("plan", "current_step", "next_step"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise MemoryValidationError(f"{name} must be text or None")
        for name in ("completed_steps", "blocked_steps", "failed_steps"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, str) for item in values):
                raise MemoryValidationError(f"{name} must be a tuple of strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "current_step": self.current_step,
            "completed_steps": list(self.completed_steps),
            "blocked_steps": list(self.blocked_steps),
            "failed_steps": list(self.failed_steps),
            "next_step": self.next_step,
        }


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One bounded, redacted summary rather than an unbounded raw log."""

    sequence: int
    category: str
    summary: str
    source: str
    importance: MemoryImportance = MemoryImportance.NORMAL
    information_kind: MemoryInformationKind = MemoryInformationKind.DERIVED
    status: str | None = None
    operation: str | None = None
    related_step: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, int) or self.sequence <= 0:
            raise MemoryValidationError("record sequence must be a positive integer")
        if not isinstance(self.category, str) or not self.category.strip():
            raise MemoryValidationError("record category must contain text")
        if not isinstance(self.summary, str) or not self.summary:
            raise MemoryValidationError("record summary must contain text")
        if not isinstance(self.source, str) or not self.source.strip():
            raise MemoryValidationError("record source must contain text")
        if not isinstance(self.importance, MemoryImportance):
            raise MemoryValidationError("record importance must be MemoryImportance")
        if not isinstance(self.information_kind, MemoryInformationKind):
            raise MemoryValidationError("record information_kind must be MemoryInformationKind")
        if not isinstance(self.metadata, Mapping):
            raise MemoryValidationError("record metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_value(_redact_value(dict(self.metadata))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "category": self.category,
            "summary": self.summary,
            "source": self.source,
            "importance": self.importance.name,
            "information_kind": self.information_kind.value,
            "status": self.status,
            "operation": self.operation,
            "related_step": self.related_step,
            "metadata": _thaw_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Immutable, bounded, serializable view of one active or closed task memory."""

    task_id: str
    session_id: str
    project_id: str | None
    project_root: str | None
    lifecycle: MemoryLifecycle
    status: MemoryStatus
    objective: str
    requirements: tuple[str, ...]
    constraints: tuple[str, ...]
    expected_outcome: str
    plan_state: MemoryPlanState | None
    observations: tuple[MemoryRecord, ...]
    tool_records: tuple[MemoryRecord, ...]
    test_records: tuple[MemoryRecord, ...]
    failure_records: tuple[MemoryRecord, ...]
    fix_records: tuple[MemoryRecord, ...]
    verification_records: tuple[MemoryRecord, ...]
    warnings: tuple[str, ...]
    sequence: int
    evictions: int
    closed_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "requirements",
            "constraints",
            "observations",
            "tool_records",
            "test_records",
            "failure_records",
            "fix_records",
            "verification_records",
            "warnings",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise MemoryValidationError(f"snapshot {name} must be immutable")
        if self.project_root is not None and not isinstance(self.project_root, str):
            raise MemoryValidationError("snapshot project_root must be text or None")

    @property
    def total_entries(self) -> int:
        return sum(
            len(items)
            for items in (
                self.observations,
                self.tool_records,
                self.test_records,
                self.failure_records,
                self.fix_records,
                self.verification_records,
            )
        )

    @property
    def is_closed(self) -> bool:
        return self.lifecycle is MemoryLifecycle.CLOSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "project_root": self.project_root,
            "lifecycle": self.lifecycle.value,
            "status": self.status.value,
            "objective": self.objective,
            "requirements": list(self.requirements),
            "constraints": list(self.constraints),
            "expected_outcome": self.expected_outcome,
            "plan_state": self.plan_state.to_dict() if self.plan_state else None,
            "observations": [record.to_dict() for record in self.observations],
            "tool_records": [record.to_dict() for record in self.tool_records],
            "test_records": [record.to_dict() for record in self.test_records],
            "failure_records": [record.to_dict() for record in self.failure_records],
            "fix_records": [record.to_dict() for record in self.fix_records],
            "verification_records": [record.to_dict() for record in self.verification_records],
            "warnings": list(self.warnings),
            "sequence": self.sequence,
            "evictions": self.evictions,
            "closed_reason": self.closed_reason,
        }

    def to_json(self) -> str:
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return encoded


class ShortTermMemory:
    """The single mutable owner for one task's bounded working context.

    The owner is task-scoped and intentionally has no retrieve/search/persistence
    API.  All writes go through the typed record methods below; callers receive
    only immutable snapshots.
    """

    def __init__(
        self,
        task_id: str,
        objective: str,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        project_root: Path | str | None = None,
        requirements: Sequence[str] = (),
        constraints: Sequence[str] = (),
        expected_outcome: str = "",
        limits: ShortTermMemoryLimits | None = None,
    ) -> None:
        self.limits = limits or ShortTermMemoryLimits()
        self.task_id = _require_text(task_id, "task_id")
        self.session_id = _require_text(session_id or self.task_id, "session_id")
        self.project_id = _optional_text(project_id, "project_id")
        self.project_root = str(Path(project_root).expanduser().resolve(strict=False)) if project_root is not None else None
        self.objective = _bounded_text(_require_text(objective, "objective"), self.limits.max_text_length_per_entry)
        self.requirements = _bounded_text_sequence(requirements, self.limits.max_text_length_per_entry, "requirements")
        self.constraints = _bounded_text_sequence(constraints, self.limits.max_text_length_per_entry, "constraints")
        self.expected_outcome = _bounded_text(_optional_text(expected_outcome, "expected_outcome") or "", self.limits.max_text_length_per_entry)
        self._lifecycle = MemoryLifecycle.CREATED
        self._status = MemoryStatus.ACTIVE
        self._plan_state: MemoryPlanState | None = None
        self._records: dict[str, list[MemoryRecord]] = {category: [] for category in _CATEGORIES}
        self._warnings: list[str] = []
        self._sequence = 0
        self._evictions = 0
        self._closed_reason: str | None = None

    @classmethod
    def for_task(
        cls,
        task: str,
        project_root: Path | str | None = None,
        *,
        session_id: str | None = None,
        limits: ShortTermMemoryLimits | None = None,
    ) -> "ShortTermMemory":
        """Create deterministic task identity without persistence or random IDs."""

        root = str(Path(project_root).expanduser().resolve(strict=False)) if project_root is not None else ""
        task_id = "task-" + hashlib.sha256(f"{root}\n{task}".encode("utf-8")).hexdigest()[:16]
        return cls(task_id, task, session_id=session_id or task_id, project_root=root or None, limits=limits)

    @property
    def lifecycle(self) -> MemoryLifecycle:
        return self._lifecycle

    @property
    def status(self) -> MemoryStatus:
        return self._status

    @property
    def is_closed(self) -> bool:
        return self._lifecycle is MemoryLifecycle.CLOSED

    def activate(self) -> None:
        if self.is_closed:
            raise MemoryClosedError("short-term memory is already closed")
        if self._lifecycle is MemoryLifecycle.CREATED:
            self._lifecycle = MemoryLifecycle.ACTIVE
            self._status = MemoryStatus.ACTIVE
        elif self._lifecycle in {MemoryLifecycle.ACTIVE, MemoryLifecycle.UPDATED}:
            return
        else:
            raise MemoryValidationError(f"cannot activate memory from {self._lifecycle.value}")

    def update_plan_state(
        self,
        plan: object,
        *,
        current_step: str | None = None,
        completed_steps: Sequence[str] = (),
        blocked_steps: Sequence[str] = (),
        failed_steps: Sequence[str] = (),
        next_step: str | None = None,
    ) -> MemorySnapshot:
        self._ensure_writable()
        plan_text = _extract_plan_text(plan)
        self._plan_state = MemoryPlanState(
            _bounded_text(plan_text, self.limits.max_text_length_per_entry),
            _bounded_optional(current_step, self.limits.max_text_length_per_entry),
            _bounded_text_sequence(completed_steps, self.limits.max_text_length_per_entry, "completed_steps"),
            _bounded_text_sequence(blocked_steps, self.limits.max_text_length_per_entry, "blocked_steps"),
            _bounded_text_sequence(failed_steps, self.limits.max_text_length_per_entry, "failed_steps"),
            _bounded_optional(next_step, self.limits.max_text_length_per_entry),
        )
        self._mark_updated()
        return self.snapshot()

    def record_observation(
        self,
        observation: object,
        *,
        source: str = "agent",
        importance: MemoryImportance = MemoryImportance.NORMAL,
        confidence: str | None = None,
        operation: str | None = None,
        related_step: str | None = None,
        authoritative: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemorySnapshot:
        return self._record(
            "observations",
            observation,
            source=source,
            importance=importance,
            information_kind=MemoryInformationKind.AUTHORITATIVE if authoritative else MemoryInformationKind.DERIVED,
            operation=operation,
            related_step=related_step,
            status=confidence,
            metadata=metadata,
        )

    def record_tool_result(
        self,
        tool_result: object,
        *,
        tool_name: str | None = None,
        operation: str | None = None,
        related_step: str | None = None,
        importance: MemoryImportance = MemoryImportance.NORMAL,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemorySnapshot:
        inferred_name = tool_name or getattr(tool_result, "tool_name", None) or getattr(tool_result, "name", None) or "tool"
        success = getattr(tool_result, "success", None)
        status = "SUCCESS" if success is True else "FAILED" if success is False else None
        return self._record(
            "tool_records",
            tool_result,
            source=str(inferred_name),
            importance=importance,
            information_kind=MemoryInformationKind.DERIVED,
            status=status,
            operation=operation or str(inferred_name),
            related_step=related_step,
            metadata=metadata,
        )

    def record_test_result(
        self,
        test_result: object,
        *,
        status: str | None = None,
        tests_executed: int | None = None,
        important_failures: Sequence[str] = (),
        related_step: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemorySnapshot:
        result_status = status or _enum_value(getattr(test_result, "status", None) or getattr(test_result, "overall_status", None))
        details = dict(metadata or {})
        if tests_executed is not None:
            details["tests_executed"] = tests_executed
        details["important_failures"] = tuple(important_failures)
        return self._record("test_records", test_result, source="test_runner", importance=MemoryImportance.IMPORTANT if important_failures else MemoryImportance.NORMAL, status=result_status, related_step=related_step, metadata=details)

    def record_failure(
        self,
        failure: object,
        *,
        classification: str | None = None,
        message: str | None = None,
        location: str | None = None,
        hypothesis: str | None = None,
        related_step: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemorySnapshot:
        details = dict(metadata or {})
        for key, value in (("location", location), ("hypothesis", hypothesis)):
            if value is not None:
                details[key] = value
        summary = message if message is not None else failure
        return self._record("failure_records", summary, source="failure_analysis", importance=MemoryImportance.CRITICAL, information_kind=MemoryInformationKind.DERIVED, status=classification, related_step=related_step, metadata=details)

    def record_fix(
        self,
        fix: object,
        *,
        target: str | None = None,
        result: str | None = None,
        verification_status: str | None = None,
        related_step: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemorySnapshot:
        details = dict(metadata or {})
        for key, value in (("target", target), ("result", result)):
            if value is not None:
                details[key] = value
        return self._record("fix_records", fix, source="automatic_fix", importance=MemoryImportance.IMPORTANT, status=verification_status, related_step=related_step, metadata=details)

    def record_verification(
        self,
        verification: object,
        *,
        tests: str | None = None,
        regression: str | None = None,
        final: str | None = None,
        completion: str | None = None,
        status: str | None = None,
        related_step: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemorySnapshot:
        details = dict(metadata or {})
        for key, value in (("tests", tests), ("regression", regression), ("final", final), ("completion", completion)):
            if value is not None:
                details[key] = value
        verification_status = status or _enum_value(getattr(verification, "status", None)) or _enum_value(getattr(verification, "state", None))
        return self._record("verification_records", verification, source="verification", importance=MemoryImportance.CRITICAL, status=verification_status, related_step=related_step, metadata=details)

    def add_warning(self, warning: object) -> MemorySnapshot:
        self._ensure_writable()
        text = _bounded_text(_redact_text(str(warning)), self.limits.max_text_length_per_entry)
        if text and text not in self._warnings:
            self._warnings.append(text)
        self._mark_updated()
        self._compact()
        return self.snapshot()

    def close(self, outcome: MemoryLifecycle | str = MemoryLifecycle.COMPLETED, *, reason: str | None = None) -> MemorySnapshot:
        if self.is_closed:
            return self.snapshot()
        if isinstance(outcome, str):
            try:
                outcome = MemoryLifecycle(outcome.upper())
            except ValueError as exc:
                raise MemoryValidationError("close outcome must be COMPLETED, FAILED, or BLOCKED") from exc
        if outcome not in {MemoryLifecycle.COMPLETED, MemoryLifecycle.FAILED, MemoryLifecycle.BLOCKED}:
            raise MemoryValidationError("close outcome must be COMPLETED, FAILED, or BLOCKED")
        self._lifecycle = outcome
        self._closed_reason = _bounded_optional(reason, self.limits.max_text_length_per_entry)
        self._lifecycle = MemoryLifecycle.CLOSED
        self._status = MemoryStatus.CLOSED
        self._compact()
        return self.snapshot()

    def snapshot(self) -> MemorySnapshot:
        self._compact()
        return MemorySnapshot(
            task_id=self.task_id,
            session_id=self.session_id,
            project_id=self.project_id,
            project_root=self.project_root,
            lifecycle=self._lifecycle,
            status=self._status,
            objective=self.objective,
            requirements=self.requirements,
            constraints=self.constraints,
            expected_outcome=self.expected_outcome,
            plan_state=self._plan_state,
            observations=tuple(self._records["observations"]),
            tool_records=tuple(self._records["tool_records"]),
            test_records=tuple(self._records["test_records"]),
            failure_records=tuple(self._records["failure_records"]),
            fix_records=tuple(self._records["fix_records"]),
            verification_records=tuple(self._records["verification_records"]),
            warnings=tuple(self._warnings),
            sequence=self._sequence,
            evictions=self._evictions,
            closed_reason=self._closed_reason,
        )

    def to_json(self) -> str:
        return self.snapshot().to_json()

    def _record(
        self,
        category: str,
        value: object,
        *,
        source: str,
        importance: MemoryImportance,
        information_kind: MemoryInformationKind = MemoryInformationKind.DERIVED,
        status: str | None = None,
        operation: str | None = None,
        related_step: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemorySnapshot:
        if self.is_closed:
            raise MemoryClosedError("short-term memory is closed; writes are rejected")
        if category not in _CATEGORIES:
            raise MemoryValidationError(f"unknown memory category: {category}")
        if not isinstance(importance, MemoryImportance):
            raise MemoryValidationError("importance must be MemoryImportance")
        safe_source = _bounded_text(_require_text(source, "source"), self.limits.max_text_length_per_entry)
        safe_status = _bounded_optional(_enum_value(status), self.limits.max_text_length_per_entry)
        safe_operation = _bounded_optional(operation, self.limits.max_text_length_per_entry)
        safe_related_step = _bounded_optional(related_step, self.limits.max_text_length_per_entry)
        safe_metadata = metadata or {}
        if not isinstance(safe_metadata, Mapping):
            raise MemoryValidationError("metadata must be a mapping")
        summary = _bounded_text(_redact_text(_safe_summary(value)), self.limits.max_text_length_per_entry)
        next_sequence = self._sequence + 1
        record = MemoryRecord(
            sequence=next_sequence,
            category=category,
            summary=summary or "[empty summary]",
            source=safe_source,
            importance=importance,
            information_kind=information_kind,
            status=safe_status,
            operation=safe_operation,
            related_step=safe_related_step,
            metadata=safe_metadata,
        )
        self._ensure_writable()
        self._sequence = next_sequence
        self._records[category].append(record)
        self._mark_updated()
        self._compact()
        return self.snapshot()

    def _ensure_writable(self) -> None:
        if self.is_closed:
            raise MemoryClosedError("short-term memory is closed; writes are rejected")
        if self._lifecycle is MemoryLifecycle.CREATED:
            self.activate()

    def _mark_updated(self) -> None:
        self._lifecycle = MemoryLifecycle.UPDATED
        if self._status is not MemoryStatus.MEMORY_LIMIT_REACHED:
            self._status = MemoryStatus.UPDATED

    def _compact(self) -> None:
        changed = False
        category_limits = {
            "observations": self.limits.max_observations,
            "tool_records": self.limits.max_tool_records,
            "failure_records": self.limits.max_failure_records,
            "test_records": self.limits.max_test_records,
            "fix_records": self.limits.max_fix_records,
            "verification_records": self.limits.max_verification_records,
        }
        for category, limit in category_limits.items():
            while len(self._records[category]) > limit:
                self._evict_lowest(category)
                changed = True
        while self._record_count() > self.limits.max_memory_entries:
            self._evict_lowest()
            changed = True
        while self._serialized_length() > self.limits.max_total_memory_bytes and self._record_count() > 0:
            self._evict_lowest()
            changed = True
        if changed:
            self._evictions += 1
            self._status = MemoryStatus.MEMORY_LIMIT_REACHED
            if "bounded eviction applied" not in self._warnings:
                self._warnings.append("bounded eviction applied")

    def _evict_lowest(self, category: str | None = None) -> None:
        candidates: list[tuple[str, MemoryRecord]] = []
        categories = (category,) if category is not None else _CATEGORIES
        for name in categories:
            candidates.extend((name, item) for item in self._records[name])
        if not candidates:
            return
        name, record = min(candidates, key=lambda item: (_retention_priority(item[0], item[1]), item[1].importance, item[1].sequence))
        self._records[name].remove(record)

    def _record_count(self) -> int:
        return sum(len(items) for items in self._records.values())

    def _serialized_length(self) -> int:
        payload = self._payload()
        return len(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def _payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "project_root": self.project_root,
            "lifecycle": self._lifecycle.value,
            "status": self._status.value,
            "objective": self.objective,
            "requirements": list(self.requirements),
            "constraints": list(self.constraints),
            "expected_outcome": self.expected_outcome,
            "plan_state": self._plan_state.to_dict() if self._plan_state else None,
            "observations": [record.to_dict() for record in self._records["observations"]],
            "tool_records": [record.to_dict() for record in self._records["tool_records"]],
            "test_records": [record.to_dict() for record in self._records["test_records"]],
            "failure_records": [record.to_dict() for record in self._records["failure_records"]],
            "fix_records": [record.to_dict() for record in self._records["fix_records"]],
            "verification_records": [record.to_dict() for record in self._records["verification_records"]],
            "warnings": list(self._warnings),
            "sequence": self._sequence,
            "evictions": self._evictions,
            "closed_reason": self._closed_reason,
        }


_CATEGORIES = ("observations", "tool_records", "test_records", "failure_records", "fix_records", "verification_records")
_CATEGORY_PRIORITY = {"observations": 1, "tool_records": 2, "test_records": 4, "fix_records": 5, "failure_records": 6, "verification_records": 7}


def _retention_priority(category: str, record: MemoryRecord) -> tuple[int, int]:
    """Lower values are evicted first; critical current evidence is retained."""

    authoritative = 2 if record.information_kind is MemoryInformationKind.AUTHORITATIVE else 0
    return (_CATEGORY_PRIORITY.get(category, 0) + authoritative, int(record.importance))


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"{name} must contain non-empty text")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MemoryValidationError(f"{name} must be text or None")
    return value.strip()


def _bounded_text(value: str, limit: int) -> str:
    text = _redact_text(value)
    if len(text) <= limit:
        return text
    marker = f"\n[truncated: kept_first_{limit}_chars]"
    return text[: max(0, limit - len(marker))] + marker


def _bounded_optional(value: object, limit: int) -> str | None:
    if value is None:
        return None
    return _bounded_text(_require_text(value, "text"), limit)


def _bounded_text_sequence(values: Sequence[str], limit: int, name: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise MemoryValidationError(f"{name} must be a sequence of strings")
    result: list[str] = []
    for value in values[:_MAX_COLLECTION_ITEMS]:
        result.append(_bounded_text(_require_text(value, name), limit))
    return tuple(dict.fromkeys(result))


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _extract_plan_text(plan: object) -> str:
    if isinstance(plan, str):
        return _require_text(plan, "plan")
    if hasattr(plan, "to_dict"):
        return _safe_summary(plan.to_dict())
    return _safe_summary(plan)


def _safe_summary(value: object) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    safe = _redact_value(value)
    try:
        return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(safe)


def _redact_text(value: str) -> str:
    text = _PRIVATE_KEY_RE.sub("[REDACTED_PRIVATE_KEY]", value)
    text = _ENV_SECRET_RE.sub(lambda match: match.group(0).split("=", 1)[0] + "=[REDACTED]", text)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]", text)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze_value(item) for name, item in sorted(value.items(), key=lambda item: str(item[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in list(value)[:_MAX_COLLECTION_ITEMS])
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _thaw_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_value(item) for item in value]
    return value


def _redact_value(value: Any, *, key: str = "") -> Any:
    if _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(name): _redact_value(item, key=str(name)) for name, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, key=key) for item in list(value)[:_MAX_COLLECTION_ITEMS]]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _redact_value(value.to_dict(), key=key)
    return _redact_text(str(value))


__all__ = [
    "MemoryClosedError",
    "MemoryImportance",
    "MemoryInformationKind",
    "MemoryLifecycle",
    "MemoryPlanState",
    "MemoryRecord",
    "MemorySerializationError",
    "MemorySnapshot",
    "MemoryStatus",
    "MemoryValidationError",
    "ShortTermMemory",
    "ShortTermMemoryError",
    "ShortTermMemoryLimits",
]
