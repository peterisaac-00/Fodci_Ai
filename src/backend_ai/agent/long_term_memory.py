"""Bounded, explicit, global long-term memory for reusable knowledge.

Phase 9.3 is intentionally independent from task-scoped and project-scoped
memory.  It provides typed entries, explicit persistence, and deterministic
lexical retrieval without embeddings, RAG, or execution authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

from backend_ai.agent.short_term_memory import _redact_text, _redact_value


LONG_TERM_MEMORY_SCHEMA_VERSION = "9.3"
LONG_TERM_MEMORY_FORMAT = "fodci.long_term_memory"
_DEFAULT_DIRECTORY = ".fodci"
_DEFAULT_FILENAME = "long_term_memory.json"
_MAX_COLLECTION_ITEMS = 64
_WORD_RE = re.compile(r"\w+", re.UNICODE)


class LongTermMemoryError(RuntimeError):
    """Base error for controlled Long-Term Memory operations."""


class LongTermMemoryValidationError(LongTermMemoryError, ValueError):
    """An invalid entry, query, limit, or persisted document."""


class LongTermMemoryConflictError(LongTermMemoryError):
    """The global memory file changed after a store loaded it."""


class LongTermMemoryClosedError(LongTermMemoryError):
    """A write was attempted after a memory owner was closed."""


class LongTermMemoryLoadStatus(str, Enum):
    LOADED = "LOADED"
    MEMORY_MISSING = "MEMORY_MISSING"
    MEMORY_CORRUPTED = "MEMORY_CORRUPTED"
    MEMORY_INVALID = "MEMORY_INVALID"
    MEMORY_UNAVAILABLE = "MEMORY_UNAVAILABLE"


class LongTermMemoryCategory(str, Enum):
    KNOWLEDGE = "knowledge"
    PATTERN = "pattern"
    LESSON = "lesson"
    SOLUTION = "solution"
    PREFERENCE = "preference"
    WARNING = "warning"


class LongTermMemorySource(str, Enum):
    USER_PROVIDED = "USER_PROVIDED"
    VERIFIED_TASK = "VERIFIED_TASK"
    PROJECT_EVIDENCE = "PROJECT_EVIDENCE"
    EXPLICIT_HOST_CONFIGURATION = "EXPLICIT_HOST_CONFIGURATION"
    STRUCTURED_OBSERVATION = "STRUCTURED_OBSERVATION"


class LongTermMemoryConfidence(IntEnum):
    UNKNOWN = 0
    INFERRED = 1
    OBSERVED = 2
    VERIFIED = 3
    USER_CONFIRMED = 4


class LongTermMemoryStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    INVALIDATED = "invalidated"
    CONFLICTED = "conflicted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class LongTermMemoryLimits:
    """Host-controlled bounds; model text cannot increase them."""

    max_memories: int = 128
    max_content_length: int = 1_024
    max_metadata_size: int = 4_096
    max_total_memory_bytes: int = 262_144
    max_key_length: int = 128

    def __post_init__(self) -> None:
        fields = ("max_memories", "max_content_length", "max_metadata_size", "max_total_memory_bytes", "max_key_length")
        for name in fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise LongTermMemoryValidationError(f"{name} must be a positive integer")
        ceilings = {
            "max_memories": 1_024,
            "max_content_length": 65_536,
            "max_metadata_size": 65_536,
            "max_total_memory_bytes": 8 * 1024 * 1024,
            "max_key_length": 512,
        }
        for name, ceiling in ceilings.items():
            if getattr(self, name) > ceiling:
                raise LongTermMemoryValidationError(f"{name} exceeds the safety ceiling")

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in ("max_memories", "max_content_length", "max_metadata_size", "max_total_memory_bytes", "max_key_length")}


@dataclass(frozen=True, slots=True)
class LongTermMemoryEntry:
    """Immutable reusable knowledge entry with access metadata."""

    entry_id: str
    content: str
    category: LongTermMemoryCategory
    source: LongTermMemorySource
    confidence: LongTermMemoryConfidence
    created_at: str
    updated_at: str
    last_accessed_at: str | None
    access_count: int
    status: LongTermMemoryStatus
    metadata: Mapping[str, Any] = field(default_factory=dict)
    conflict_with: tuple[str, ...] = ()
    schema_version: str = LONG_TERM_MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.entry_id, str) or not self.entry_id.startswith("ltm-"):
            raise LongTermMemoryValidationError("entry_id must be canonical")
        if not isinstance(self.content, str) or not self.content.strip():
            raise LongTermMemoryValidationError("content must contain text")
        if not isinstance(self.category, LongTermMemoryCategory):
            raise LongTermMemoryValidationError("category must be LongTermMemoryCategory")
        if not isinstance(self.source, LongTermMemorySource):
            raise LongTermMemoryValidationError("source must be LongTermMemorySource")
        if not isinstance(self.confidence, LongTermMemoryConfidence):
            raise LongTermMemoryValidationError("confidence must be LongTermMemoryConfidence")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise LongTermMemoryValidationError("created_at must contain text")
        if not isinstance(self.updated_at, str) or not self.updated_at:
            raise LongTermMemoryValidationError("updated_at must contain text")
        if self.last_accessed_at is not None and not isinstance(self.last_accessed_at, str):
            raise LongTermMemoryValidationError("last_accessed_at must be text or None")
        if not isinstance(self.access_count, int) or isinstance(self.access_count, bool) or self.access_count < 0:
            raise LongTermMemoryValidationError("access_count must be a non-negative integer")
        if not isinstance(self.status, LongTermMemoryStatus):
            raise LongTermMemoryValidationError("status must be LongTermMemoryStatus")
        if not isinstance(self.metadata, Mapping):
            raise LongTermMemoryValidationError("metadata must be a mapping")
        if not isinstance(self.conflict_with, tuple) or any(not isinstance(item, str) for item in self.conflict_with):
            raise LongTermMemoryValidationError("conflict_with must be a tuple of IDs")
        if self.schema_version != LONG_TERM_MEMORY_SCHEMA_VERSION:
            raise LongTermMemoryValidationError("unsupported Long-Term Memory schema version")
        object.__setattr__(self, "metadata", _freeze_value(_redact_value(dict(self.metadata))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "category": self.category.value,
            "source": self.source.value,
            "confidence": self.confidence.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "status": self.status.value,
            "metadata": _thaw_value(self.metadata),
            "conflict_with": list(self.conflict_with),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class LongTermMemorySnapshot:
    entries: tuple[LongTermMemoryEntry, ...]
    status: LongTermMemoryLoadStatus
    sequence: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(not isinstance(item, LongTermMemoryEntry) for item in self.entries):
            raise LongTermMemoryValidationError("entries must be a tuple of LongTermMemoryEntry")
        if not isinstance(self.status, LongTermMemoryLoadStatus):
            raise LongTermMemoryValidationError("status must be LongTermMemoryLoadStatus")
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise LongTermMemoryValidationError("sequence must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": LONG_TERM_MEMORY_FORMAT,
            "schema_version": LONG_TERM_MEMORY_SCHEMA_VERSION,
            "entries": [item.to_dict() for item in self.entries],
            "status": self.status.value,
            "sequence": self.sequence,
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class LongTermMemoryLoadResult:
    status: LongTermMemoryLoadStatus
    memory: "LongTermMemory | None"
    message: str | None = None


class LongTermMemory:
    """Explicit in-memory owner for global reusable knowledge."""

    def __init__(self, *, limits: LongTermMemoryLimits | None = None, clock: Callable[[], str] | None = None) -> None:
        self.limits = limits or LongTermMemoryLimits()
        self._clock = clock or _utc_now
        self._entries: dict[str, LongTermMemoryEntry] = {}
        self._sequence = 0
        self._warnings: list[str] = []
        self._closed = False
        self._load_status = LongTermMemoryLoadStatus.MEMORY_MISSING

    @classmethod
    def from_snapshot(cls, snapshot: LongTermMemorySnapshot, *, limits: LongTermMemoryLimits | None = None, clock: Callable[[], str] | None = None) -> "LongTermMemory":
        memory = cls(limits=limits, clock=clock)
        memory._entries = {item.entry_id: item for item in snapshot.entries}
        memory._sequence = snapshot.sequence
        memory._warnings = list(snapshot.warnings)
        memory._load_status = snapshot.status
        return memory

    @property
    def load_status(self) -> LongTermMemoryLoadStatus:
        return self._load_status

    def add(
        self,
        *,
        content: str,
        category: LongTermMemoryCategory | str,
        source: LongTermMemorySource | str,
        confidence: LongTermMemoryConfidence | str,
        metadata: Mapping[str, Any] | None = None,
        status: LongTermMemoryStatus | str = LongTermMemoryStatus.ACTIVE,
    ) -> LongTermMemoryEntry:
        self._ensure_writable()
        category = _coerce_enum(category, LongTermMemoryCategory, "category")
        source = _coerce_enum(source, LongTermMemorySource, "source")
        confidence = _coerce_enum(confidence, LongTermMemoryConfidence, "confidence")
        status = _coerce_enum(status, LongTermMemoryStatus, "status")
        safe_content = _safe_content(content, self.limits.max_content_length)
        safe_metadata = _safe_metadata(metadata or {}, self.limits)
        self._sequence += 1
        now = self._now()
        entry_id = "ltm-" + hashlib.sha256(f"{category.value}|{safe_content}|{self._sequence}|{now}".encode("utf-8")).hexdigest()[:24]
        conflict_ids = self._conflicts_for(category, safe_metadata, safe_content)
        entry_status = status
        if conflict_ids:
            entry_status = LongTermMemoryStatus.CONFLICTED
        entry = LongTermMemoryEntry(entry_id, safe_content, category, source, confidence, now, now, None, 0, entry_status, safe_metadata, tuple(sorted(conflict_ids)))
        if conflict_ids:
            for conflict_id in conflict_ids:
                existing = self._entries[conflict_id]
                self._entries[conflict_id] = _replace_entry(existing, status=LongTermMemoryStatus.CONFLICTED, updated_at=now, conflict_with=tuple(sorted(set(existing.conflict_with + (entry_id,)))))
        previous_entries = dict(self._entries)
        previous_sequence = self._sequence
        previous_warnings = list(self._warnings)
        self._entries[entry_id] = entry
        try:
            self._enforce_limits()
        except LongTermMemoryError:
            self._entries = previous_entries
            self._sequence = previous_sequence
            self._warnings = previous_warnings
            raise
        return entry

    def get(self, entry_id: str, *, track_access: bool = True) -> LongTermMemoryEntry | None:
        self._validate_id(entry_id)
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        if not track_access:
            return entry
        self._ensure_writable()
        updated = _replace_entry(entry, last_accessed_at=self._now(), access_count=entry.access_count + 1)
        self._entries[entry_id] = updated
        return updated

    def update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        category: LongTermMemoryCategory | str | None = None,
        source: LongTermMemorySource | str | None = None,
        confidence: LongTermMemoryConfidence | str | None = None,
        status: LongTermMemoryStatus | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LongTermMemoryEntry:
        self._ensure_writable()
        self._validate_id(entry_id)
        current = self._entries.get(entry_id)
        if current is None:
            raise LongTermMemoryValidationError("unknown entry_id")
        now = self._now()
        updated = _replace_entry(
            current,
            content=_safe_content(content, self.limits.max_content_length) if content is not None else current.content,
            category=_coerce_enum(category, LongTermMemoryCategory, "category") if category is not None else current.category,
            source=_coerce_enum(source, LongTermMemorySource, "source") if source is not None else current.source,
            confidence=_coerce_enum(confidence, LongTermMemoryConfidence, "confidence") if confidence is not None else current.confidence,
            status=_coerce_enum(status, LongTermMemoryStatus, "status") if status is not None else current.status,
            metadata=_safe_metadata(metadata, self.limits) if metadata is not None else current.metadata,
            updated_at=now,
        )
        previous_entries = dict(self._entries)
        previous_sequence = self._sequence
        previous_warnings = list(self._warnings)
        self._entries[entry_id] = updated
        try:
            self._enforce_limits()
        except LongTermMemoryError:
            self._entries = previous_entries
            self._sequence = previous_sequence
            self._warnings = previous_warnings
            raise
        return self._entries.get(entry_id, updated)

    def delete(self, entry_id: str) -> bool:
        self._ensure_writable()
        self._validate_id(entry_id)
        return self._entries.pop(entry_id, None) is not None

    def list(
        self,
        *,
        category: LongTermMemoryCategory | str | None = None,
        status: LongTermMemoryStatus | str | None = None,
    ) -> tuple[LongTermMemoryEntry, ...]:
        category_value = _coerce_enum(category, LongTermMemoryCategory, "category") if category is not None else None
        status_value = _coerce_enum(status, LongTermMemoryStatus, "status") if status is not None else None
        return tuple(item for item in sorted(self._entries.values(), key=lambda entry: (entry.created_at, entry.entry_id)) if (category_value is None or item.category is category_value) and (status_value is None or item.status is status_value))

    def search(
        self,
        query: str,
        *,
        category: LongTermMemoryCategory | str | None = None,
        limit: int = 5,
    ) -> tuple[LongTermMemoryEntry, ...]:
        self._ensure_writable()
        query_tokens = _tokens(query)
        if not query_tokens:
            raise LongTermMemoryValidationError("query must contain searchable text")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0 or limit > self.limits.max_memories:
            raise LongTermMemoryValidationError("limit is outside the configured bound")
        category_value = _coerce_enum(category, LongTermMemoryCategory, "category") if category is not None else None
        candidates = [item for item in self._entries.values() if item.status is LongTermMemoryStatus.ACTIVE and (category_value is None or item.category is category_value)]
        ranked = sorted(candidates, key=lambda entry: _ranking_key(entry, query_tokens), reverse=True)
        selected = ranked[:limit]
        now = self._now()
        results: list[LongTermMemoryEntry] = []
        for entry in selected:
            updated = _replace_entry(entry, last_accessed_at=now, access_count=entry.access_count + 1)
            self._entries[entry.entry_id] = updated
            results.append(updated)
        return tuple(results)

    def snapshot(self) -> LongTermMemorySnapshot:
        return LongTermMemorySnapshot(tuple(sorted(self._entries.values(), key=lambda entry: (entry.created_at, entry.entry_id))), self._load_status, self._sequence, tuple(self._warnings))

    def to_json(self) -> str:
        return self.snapshot().to_json()

    def close(self) -> LongTermMemorySnapshot:
        self._closed = True
        return self.snapshot()

    def _conflicts_for(self, category: LongTermMemoryCategory, metadata: Mapping[str, Any], content: str) -> tuple[str, ...]:
        topic = _conflict_topic(metadata)
        if not topic:
            return ()
        return tuple(entry.entry_id for entry in self._entries.values() if entry.category is category and entry.status in {LongTermMemoryStatus.ACTIVE, LongTermMemoryStatus.CONFLICTED} and _conflict_topic(entry.metadata) == topic and entry.content != content)

    def _enforce_limits(self) -> None:
        if len(self._entries) > self.limits.max_memories:
            raise LongTermMemoryValidationError("max_memories exceeded; explicit deletion is required")
        if self._serialized_size() > self.limits.max_total_memory_bytes:
            raise LongTermMemoryValidationError("max_total_memory_bytes exceeded; explicit deletion is required")

    def _serialized_size(self) -> int:
        return len(self.to_json().encode("utf-8"))

    def _now(self) -> str:
        value = self._clock()
        if not isinstance(value, str) or not value:
            raise LongTermMemoryValidationError("clock must return a non-empty timestamp")
        return value

    def _validate_id(self, entry_id: str) -> None:
        if not isinstance(entry_id, str) or not entry_id.startswith("ltm-"):
            raise LongTermMemoryValidationError("entry_id must be canonical")

    def _ensure_writable(self) -> None:
        if self._closed:
            raise LongTermMemoryClosedError("long-term memory is closed")


class LongTermMemoryStore:
    """Atomic global persistence independent from any project root."""

    def __init__(self, path: Path | str | None = None, *, limits: LongTermMemoryLimits | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_long_term_memory_path()
        self.limits = limits or LongTermMemoryLimits()
        self._loaded_digest: str | None = None

    @classmethod
    def default(cls, *, limits: LongTermMemoryLimits | None = None) -> "LongTermMemoryStore":
        return cls(None, limits=limits)

    def empty(self, *, clock: Callable[[], str] | None = None) -> LongTermMemory:
        return LongTermMemory(limits=self.limits, clock=clock)

    def load(self, *, clock: Callable[[], str] | None = None) -> LongTermMemoryLoadResult:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            self._loaded_digest = None
            return LongTermMemoryLoadResult(LongTermMemoryLoadStatus.MEMORY_MISSING, None, "long-term memory file does not exist")
        except OSError as exc:
            return LongTermMemoryLoadResult(LongTermMemoryLoadStatus.MEMORY_UNAVAILABLE, None, str(exc))
        if len(raw) > self.limits.max_total_memory_bytes:
            return LongTermMemoryLoadResult(LongTermMemoryLoadStatus.MEMORY_INVALID, None, "long-term memory exceeds configured byte limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
            snapshot = _snapshot_from_dict(payload, self.limits)
        except UnicodeDecodeError as exc:
            return LongTermMemoryLoadResult(LongTermMemoryLoadStatus.MEMORY_CORRUPTED, None, f"invalid UTF-8: {exc}")
        except json.JSONDecodeError as exc:
            return LongTermMemoryLoadResult(LongTermMemoryLoadStatus.MEMORY_CORRUPTED, None, str(exc))
        except (LongTermMemoryValidationError, TypeError, ValueError) as exc:
            return LongTermMemoryLoadResult(LongTermMemoryLoadStatus.MEMORY_INVALID, None, str(exc))
        self._loaded_digest = _digest(raw)
        return LongTermMemoryLoadResult(LongTermMemoryLoadStatus.LOADED, LongTermMemory.from_snapshot(snapshot, limits=self.limits, clock=clock), None)

    def save(self, memory: LongTermMemory) -> Path:
        if not isinstance(memory, LongTermMemory):
            raise LongTermMemoryValidationError("memory must be LongTermMemory")
        if memory._closed:
            raise LongTermMemoryClosedError("cannot persist closed long-term memory")
        if self.path.exists():
            current_digest = _digest(self.path.read_bytes())
            if self._loaded_digest is None or current_digest != self._loaded_digest:
                raise LongTermMemoryConflictError("long-term memory changed since this store loaded it")
        memory._load_status = LongTermMemoryLoadStatus.LOADED
        payload = memory.to_json().encode("utf-8")
        if len(payload) > self.limits.max_total_memory_bytes:
            raise LongTermMemoryValidationError("long-term memory exceeds configured byte limit")
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=directory, prefix=".long_term_memory.", suffix=".tmp", delete=False) as stream:
                temporary_path = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            try:
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
            self._loaded_digest = _digest(payload)
            return self.path
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass


def default_long_term_memory_path() -> Path:
    """Return the deliberate user-global storage location."""

    return Path.home() / _DEFAULT_DIRECTORY / _DEFAULT_FILENAME


def _snapshot_from_dict(payload: Any, limits: LongTermMemoryLimits) -> LongTermMemorySnapshot:
    if not isinstance(payload, Mapping) or payload.get("format") != LONG_TERM_MEMORY_FORMAT:
        raise LongTermMemoryValidationError("unsupported Long-Term Memory format")
    allowed = {"format", "schema_version", "entries", "status", "sequence", "warnings"}
    if set(payload) - allowed:
        raise LongTermMemoryValidationError("unknown Long-Term Memory fields are not accepted")
    if payload.get("schema_version") != LONG_TERM_MEMORY_SCHEMA_VERSION:
        raise LongTermMemoryValidationError("unsupported Long-Term Memory schema version")
    entries_payload = payload.get("entries", [])
    if not isinstance(entries_payload, list) or len(entries_payload) > limits.max_memories:
        raise LongTermMemoryValidationError("invalid or oversized entries array")
    entries = tuple(_entry_from_dict(item, limits) for item in entries_payload)
    sequence = payload.get("sequence", 0)
    if not isinstance(sequence, int) or sequence < 0:
        raise LongTermMemoryValidationError("invalid sequence")
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise LongTermMemoryValidationError("invalid warnings")
    status = _coerce_enum(payload.get("status", LongTermMemoryLoadStatus.LOADED.value), LongTermMemoryLoadStatus, "status")
    return LongTermMemorySnapshot(entries, status, sequence, tuple(warnings[:_MAX_COLLECTION_ITEMS]))


def _entry_from_dict(payload: Any, limits: LongTermMemoryLimits) -> LongTermMemoryEntry:
    if not isinstance(payload, Mapping):
        raise LongTermMemoryValidationError("entry must be an object")
    allowed = {"entry_id", "content", "category", "source", "confidence", "created_at", "updated_at", "last_accessed_at", "access_count", "status", "metadata", "conflict_with", "schema_version"}
    if set(payload) - allowed:
        raise LongTermMemoryValidationError("unknown Long-Term Memory entry fields are not accepted")
    metadata = _safe_metadata(payload.get("metadata", {}), limits)
    conflict_with = payload.get("conflict_with", [])
    if not isinstance(conflict_with, list) or len(conflict_with) > _MAX_COLLECTION_ITEMS or any(not isinstance(item, str) for item in conflict_with):
        raise LongTermMemoryValidationError("invalid conflict_with")
    return LongTermMemoryEntry(
        str(payload.get("entry_id", "")),
        _safe_content(payload.get("content", ""), limits.max_content_length),
        _coerce_enum(payload.get("category"), LongTermMemoryCategory, "category"),
        _coerce_enum(payload.get("source"), LongTermMemorySource, "source"),
        _coerce_enum(payload.get("confidence"), LongTermMemoryConfidence, "confidence"),
        str(payload.get("created_at", "")),
        str(payload.get("updated_at", "")),
        payload.get("last_accessed_at"),
        payload.get("access_count", 0),
        _coerce_enum(payload.get("status"), LongTermMemoryStatus, "status"),
        metadata,
        tuple(conflict_with),
        str(payload.get("schema_version", "")),
    )


def _coerce_enum(value: Any, enum_type: type[Enum], name: str):
    if isinstance(value, enum_type):
        return value
    try:
        if issubclass(enum_type, IntEnum) and isinstance(value, str):
            return enum_type[value]
        return enum_type(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise LongTermMemoryValidationError(f"invalid {name}: {value!r}") from exc


def _safe_content(value: Any, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongTermMemoryValidationError("content must contain text")
    safe = _redact_text(value.strip())
    if len(safe) > limit:
        raise LongTermMemoryValidationError("content exceeds configured length bound")
    return safe


def _safe_metadata(value: Mapping[str, Any], limits: LongTermMemoryLimits) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LongTermMemoryValidationError("metadata must be a mapping")
    safe = _redact_value(dict(value))
    encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > limits.max_metadata_size:
        raise LongTermMemoryValidationError("metadata exceeds configured size bound")
    return _freeze_value(safe)


def _replace_entry(entry: LongTermMemoryEntry, **changes: Any) -> LongTermMemoryEntry:
    fields = {
        "entry_id": entry.entry_id, "content": entry.content, "category": entry.category,
        "source": entry.source, "confidence": entry.confidence, "created_at": entry.created_at,
        "updated_at": entry.updated_at, "last_accessed_at": entry.last_accessed_at,
        "access_count": entry.access_count, "status": entry.status, "metadata": entry.metadata,
        "conflict_with": entry.conflict_with, "schema_version": entry.schema_version,
    }
    fields.update(changes)
    return LongTermMemoryEntry(**fields)


def _conflict_topic(metadata: Mapping[str, Any]) -> str | None:
    for key in ("topic", "key", "subject"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.casefold().strip()
    return None


def _tokens(value: str) -> frozenset[str]:
    if not isinstance(value, str):
        raise LongTermMemoryValidationError("query must contain text")
    return frozenset(token for token in _WORD_RE.findall(value.casefold()) if token)


def _ranking_key(entry: LongTermMemoryEntry, query_tokens: frozenset[str]) -> tuple[int, int, int, str, int, str]:
    searchable = " ".join((entry.content, entry.category.value, entry.source.value, _canonical(entry.metadata))).casefold()
    entry_tokens = set(_WORD_RE.findall(searchable))
    overlap = len(query_tokens & entry_tokens)
    phrase = int(" ".join(sorted(query_tokens)) in searchable) if len(query_tokens) > 1 else int(next(iter(query_tokens)) in entry_tokens)
    return (overlap, phrase, int(entry.confidence), entry.last_accessed_at or entry.updated_at, entry.access_count, entry.entry_id)


def _canonical(value: Any) -> str:
    return json.dumps(_thaw_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze_value(item) for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in list(value)[:_MAX_COLLECTION_ITEMS])
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _thaw_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_value(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "LONG_TERM_MEMORY_FORMAT",
    "LONG_TERM_MEMORY_SCHEMA_VERSION",
    "LongTermMemory",
    "LongTermMemoryCategory",
    "LongTermMemoryClosedError",
    "LongTermMemoryConfidence",
    "LongTermMemoryConflictError",
    "LongTermMemoryEntry",
    "LongTermMemoryError",
    "LongTermMemoryLimits",
    "LongTermMemoryLoadResult",
    "LongTermMemoryLoadStatus",
    "LongTermMemorySnapshot",
    "LongTermMemorySource",
    "LongTermMemoryStatus",
    "LongTermMemoryStore",
    "LongTermMemoryValidationError",
    "default_long_term_memory_path",
]
