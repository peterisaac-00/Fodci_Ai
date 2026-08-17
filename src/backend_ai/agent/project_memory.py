"""Bounded, deterministic, persistent memory for verified project facts.

Phase 9.2 deliberately stores reusable project knowledge, not task history.  It
has no retrieval, ranking, execution, network, or cross-project persistence API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any

from backend_ai.agent.short_term_memory import _redact_text, _redact_value


PROJECT_MEMORY_SCHEMA_VERSION = "9.2"
PROJECT_MEMORY_FORMAT = "fodci.project_memory"
_STORAGE_DIRECTORY = ".fodci"
_STORAGE_FILENAME = "project_memory.json"
_MAX_COLLECTION_ITEMS = 64


class ProjectMemoryError(RuntimeError):
    """Base error for controlled Project Memory operations."""


class ProjectMemoryValidationError(ProjectMemoryError, ValueError):
    """An invalid project memory input or persisted document."""


class ProjectMemoryConflictError(ProjectMemoryError):
    """The on-disk memory changed since this store loaded it."""


class ProjectMemoryClosedError(ProjectMemoryError):
    """A write was attempted after a project memory instance was closed."""


class ProjectMemoryLoadStatus(str, Enum):
    LOADED = "LOADED"
    MEMORY_MISSING = "MEMORY_MISSING"
    MEMORY_CORRUPTED = "MEMORY_CORRUPTED"
    MEMORY_INVALID = "MEMORY_INVALID"
    MEMORY_UNAVAILABLE = "MEMORY_UNAVAILABLE"


class FactCategory(str, Enum):
    PROJECT_TYPE = "PROJECT_TYPE"
    LANGUAGE = "LANGUAGE"
    FRAMEWORK = "FRAMEWORK"
    DATABASE = "DATABASE"
    TESTING = "TESTING"
    AUTHENTICATION = "AUTHENTICATION"
    PACKAGE_MANAGEMENT = "PACKAGE_MANAGEMENT"
    API = "API"
    ARCHITECTURE = "ARCHITECTURE"
    PROJECT_STRUCTURE = "PROJECT_STRUCTURE"
    BUILD = "BUILD"
    DEVELOPMENT_TOOLING = "DEVELOPMENT_TOOLING"
    DEPLOYMENT = "DEPLOYMENT"
    CONTAINERIZATION = "CONTAINERIZATION"
    CONFIGURATION = "CONFIGURATION"
    CONVENTION = "CONVENTION"
    CONSTRAINT = "CONSTRAINT"


class FactSource(str, Enum):
    USER_PROVIDED = "USER_PROVIDED"
    PROJECT_CONTEXT = "PROJECT_CONTEXT"
    CONFIGURATION = "CONFIGURATION"
    MANIFEST = "MANIFEST"
    STRUCTURED_TOOL_RESULT = "STRUCTURED_TOOL_RESULT"
    TEST_RESULT = "TEST_RESULT"
    VERIFICATION_RESULT = "VERIFICATION_RESULT"
    EXPLICIT_HOST_CONFIGURATION = "EXPLICIT_HOST_CONFIGURATION"


class FactConfidence(IntEnum):
    UNKNOWN = 0
    INFERRED = 1
    OBSERVED = 2
    VERIFIED = 3
    USER_CONFIRMED = 4


class FactStatus(str, Enum):
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    INVALID = "INVALID"
    CONFLICTED = "CONFLICTED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True, slots=True)
class ProjectMemoryLimits:
    """Host-controlled bounds for facts, evidence, metadata, and serialization."""

    max_facts: int = 64
    max_fact_value_length: int = 512
    max_evidence_per_fact: int = 8
    max_total_memory_bytes: int = 65_536
    max_metadata_size: int = 2_048
    max_conflict_records: int = 32
    max_key_length: int = 128
    max_evidence_text_length: int = 512

    def __post_init__(self) -> None:
        fields = (
            "max_facts", "max_fact_value_length", "max_evidence_per_fact",
            "max_total_memory_bytes", "max_metadata_size", "max_conflict_records",
            "max_key_length", "max_evidence_text_length",
        )
        for name in fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ProjectMemoryValidationError(f"{name} must be a positive integer")
        ceilings = {
            "max_facts": 512,
            "max_fact_value_length": 16_384,
            "max_evidence_per_fact": 64,
            "max_total_memory_bytes": 4 * 1024 * 1024,
            "max_metadata_size": 32_768,
            "max_conflict_records": 256,
            "max_key_length": 512,
            "max_evidence_text_length": 16_384,
        }
        for name, ceiling in ceilings.items():
            if getattr(self, name) > ceiling:
                raise ProjectMemoryValidationError(f"{name} exceeds the safety ceiling")

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in (
            "max_facts", "max_fact_value_length", "max_evidence_per_fact",
            "max_total_memory_bytes", "max_metadata_size", "max_conflict_records",
            "max_key_length", "max_evidence_text_length",
        )}


@dataclass(frozen=True, slots=True)
class ProjectIdentity:
    """Canonical identity for one normalized project root."""

    project_id: str
    project_root: str
    schema_version: str = PROJECT_MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, str) or not self.project_id.startswith("project-"):
            raise ProjectMemoryValidationError("project_id must be a canonical project-* identifier")
        if not isinstance(self.project_root, str) or not self.project_root:
            raise ProjectMemoryValidationError("project_root must contain text")
        if self.schema_version != PROJECT_MEMORY_SCHEMA_VERSION:
            raise ProjectMemoryValidationError(f"unsupported Project Memory schema: {self.schema_version}")

    @classmethod
    def for_root(cls, project_root: Path | str) -> "ProjectIdentity":
        root = str(Path(project_root).expanduser().resolve(strict=False))
        digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:24]
        return cls(f"project-{digest}", root)

    def to_dict(self) -> dict[str, str]:
        return {"project_id": self.project_id, "project_root": self.project_root, "schema_version": self.schema_version}


@dataclass(frozen=True, slots=True)
class FactEvidence:
    """Bounded provenance for one fact; raw logs and source content are not accepted."""

    source: FactSource
    reference: str
    summary: str
    verified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, FactSource):
            raise ProjectMemoryValidationError("evidence source must be FactSource")
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ProjectMemoryValidationError("evidence reference must contain text")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ProjectMemoryValidationError("evidence summary must contain text")
        if not isinstance(self.verified, bool):
            raise ProjectMemoryValidationError("evidence verified must be boolean")
        if not isinstance(self.metadata, Mapping):
            raise ProjectMemoryValidationError("evidence metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_value(_redact_value(dict(self.metadata))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "reference": self.reference,
            "summary": self.summary,
            "verified": self.verified,
            "metadata": _thaw_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ProjectFact:
    """Strongly typed reusable project fact with bounded evidence."""

    fact_id: str
    category: FactCategory
    key: str
    value: Any
    confidence: FactConfidence
    status: FactStatus
    evidence: tuple[FactEvidence, ...]
    source: FactSource
    sequence: int
    conflict_with: tuple[str, ...] = ()
    schema_version: str = PROJECT_MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.fact_id, str) or not self.fact_id.startswith("fact-"):
            raise ProjectMemoryValidationError("fact_id must be canonical")
        if not isinstance(self.category, FactCategory):
            raise ProjectMemoryValidationError("category must be FactCategory")
        if not isinstance(self.key, str) or not self.key.strip() or self.key.startswith("/") or ".." in self.key.split("/"):
            raise ProjectMemoryValidationError("fact key must be a safe non-empty relative key")
        if not isinstance(self.confidence, FactConfidence):
            raise ProjectMemoryValidationError("confidence must be FactConfidence")
        if not isinstance(self.status, FactStatus):
            raise ProjectMemoryValidationError("status must be FactStatus")
        if not isinstance(self.evidence, tuple) or any(not isinstance(item, FactEvidence) for item in self.evidence):
            raise ProjectMemoryValidationError("evidence must be a tuple of FactEvidence")
        if not isinstance(self.source, FactSource):
            raise ProjectMemoryValidationError("source must be FactSource")
        if not isinstance(self.sequence, int) or self.sequence <= 0:
            raise ProjectMemoryValidationError("fact sequence must be positive")
        if self.schema_version != PROJECT_MEMORY_SCHEMA_VERSION:
            raise ProjectMemoryValidationError("unsupported fact schema version")
        object.__setattr__(self, "value", _freeze_value(_redact_value(self.value)))
        object.__setattr__(self, "conflict_with", tuple(self.conflict_with))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "category": self.category.value,
            "key": self.key,
            "value": _thaw_value(self.value),
            "confidence": self.confidence.name,
            "status": self.status.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "source": self.source.value,
            "sequence": self.sequence,
            "conflict_with": list(self.conflict_with),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ProjectMemorySnapshot:
    """Read-only bounded view of persistent project facts."""

    identity: ProjectIdentity
    facts: tuple[ProjectFact, ...]
    conflicts: tuple[ProjectFact, ...]
    status: ProjectMemoryLoadStatus
    sequence: int
    evictions: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.facts, tuple) or not isinstance(self.conflicts, tuple):
            raise ProjectMemoryValidationError("snapshot facts and conflicts must be tuples")
        if not isinstance(self.status, ProjectMemoryLoadStatus):
            raise ProjectMemoryValidationError("snapshot status must be ProjectMemoryLoadStatus")

    @property
    def active_facts(self) -> tuple[ProjectFact, ...]:
        return tuple(item for item in self.facts if item.status is FactStatus.ACTIVE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PROJECT_MEMORY_FORMAT,
            "schema_version": PROJECT_MEMORY_SCHEMA_VERSION,
            "identity": self.identity.to_dict(),
            "facts": [item.to_dict() for item in self.facts],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "status": self.status.value,
            "sequence": self.sequence,
            "evictions": self.evictions,
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ProjectMemoryLoadResult:
    status: ProjectMemoryLoadStatus
    memory: "ProjectMemory | None"
    message: str | None = None


class ProjectMemory:
    """Single authoritative owner for facts belonging to one project."""

    def __init__(self, identity: ProjectIdentity, *, limits: ProjectMemoryLimits | None = None) -> None:
        self.identity = identity
        self.limits = limits or ProjectMemoryLimits()
        self._facts: dict[str, ProjectFact] = {}
        self._active_by_key: dict[tuple[FactCategory, str], str] = {}
        self._sequence = 0
        self._evictions = 0
        self._warnings: list[str] = []
        self._closed = False
        self._load_status = ProjectMemoryLoadStatus.MEMORY_MISSING

    @classmethod
    def for_project(cls, project_root: Path | str, *, limits: ProjectMemoryLimits | None = None) -> "ProjectMemory":
        return cls(ProjectIdentity.for_root(project_root), limits=limits)

    @classmethod
    def from_snapshot(cls, snapshot: ProjectMemorySnapshot, *, limits: ProjectMemoryLimits | None = None) -> "ProjectMemory":
        memory = cls(snapshot.identity, limits=limits)
        memory._facts = {item.fact_id: item for item in snapshot.facts}
        memory._facts.update({item.fact_id: item for item in snapshot.conflicts})
        memory._active_by_key = {
            (item.category, item.key): item.fact_id
            for item in memory._facts.values()
            if item.status is FactStatus.ACTIVE
        }
        memory._sequence = snapshot.sequence
        memory._evictions = snapshot.evictions
        memory._warnings = list(snapshot.warnings)
        memory._load_status = snapshot.status
        return memory

    @property
    def project_id(self) -> str:
        return self.identity.project_id

    @property
    def project_root(self) -> str:
        return self.identity.project_root

    @property
    def load_status(self) -> ProjectMemoryLoadStatus:
        return self._load_status

    def add_fact(
        self,
        *,
        category: FactCategory | str,
        key: str,
        value: Any,
        source: FactSource | str,
        confidence: FactConfidence | str,
        evidence: Sequence[FactEvidence],
        metadata: Mapping[str, Any] | None = None,
    ) -> ProjectMemorySnapshot:
        self._ensure_writable()
        category = _coerce_enum(category, FactCategory, "category")
        source = _coerce_enum(source, FactSource, "source")
        confidence = _coerce_enum(confidence, FactConfidence, "confidence")
        safe_key = _safe_key(key, self.limits.max_key_length)
        safe_value = _bounded_value(value, self.limits.max_fact_value_length, self.limits.max_metadata_size)
        safe_evidence = _bounded_evidence(evidence, self.limits)
        if not safe_evidence:
            raise ProjectMemoryValidationError("a project fact requires at least one evidence record")
        self._sequence += 1
        fact_id = _fact_id(category, safe_key, safe_value)
        existing_id = self._active_by_key.get((category, safe_key))
        existing = self._facts.get(existing_id) if existing_id else None
        if existing is not None and _canonical(existing.value) == _canonical(safe_value):
            merged = _merge_evidence(existing.evidence, safe_evidence, self.limits.max_evidence_per_fact)
            updated = ProjectFact(
                existing.fact_id, existing.category, existing.key, existing.value,
                max(existing.confidence, confidence), FactStatus.ACTIVE, merged,
                _stronger_source(existing.source, source), self._sequence, existing.conflict_with,
            )
            self._facts[updated.fact_id] = updated
            self._active_by_key[(category, safe_key)] = updated.fact_id
        else:
            fact = ProjectFact(fact_id, category, safe_key, safe_value, confidence, FactStatus.ACTIVE, safe_evidence, source, self._sequence)
            if existing is not None:
                if _authority_score(confidence, source) > _authority_score(existing.confidence, existing.source):
                    self._facts[existing.fact_id] = _replace_fact(existing, status=FactStatus.SUPERSEDED, sequence=self._sequence, conflict_with=(fact.fact_id,))
                    self._facts[fact.fact_id] = fact
                    self._active_by_key[(category, safe_key)] = fact.fact_id
                else:
                    rejected = _replace_fact(fact, status=FactStatus.REJECTED, sequence=self._sequence, conflict_with=(existing.fact_id,))
                    self._facts[rejected.fact_id] = rejected
                    self._facts[existing.fact_id] = _replace_fact(existing, status=FactStatus.ACTIVE, sequence=self._sequence, conflict_with=tuple(sorted(set(existing.conflict_with + (rejected.fact_id,)))))
                    self._active_by_key[(category, safe_key)] = existing.fact_id
            else:
                self._facts[fact.fact_id] = fact
                self._active_by_key[(category, safe_key)] = fact.fact_id
        self._compact()
        return self.snapshot()

    def add_project_context(self, context: object) -> ProjectMemorySnapshot:
        """Add only bounded structured facts from an existing ProjectContext."""

        self._ensure_writable()
        project_type = getattr(context, "project_type", None)
        confidence = FactConfidence.VERIFIED if getattr(context, "completeness", "partial") == "complete" and not getattr(context, "truncated", False) else FactConfidence.OBSERVED
        evidence = FactEvidence(FactSource.PROJECT_CONTEXT, "ProjectContext", "bounded structured project context", verified=confidence >= FactConfidence.VERIFIED)
        if isinstance(project_type, str) and project_type:
            self.add_fact(category=FactCategory.PROJECT_TYPE, key="project.type", value=project_type, source=FactSource.PROJECT_CONTEXT, confidence=confidence, evidence=(evidence,))
        mappings = (
            ("languages", FactCategory.LANGUAGE, "language.name"),
            ("frameworks", FactCategory.FRAMEWORK, "framework.name"),
            ("package_managers", FactCategory.PACKAGE_MANAGEMENT, "package_manager.name"),
            ("databases", FactCategory.DATABASE, "database.name"),
            ("test_frameworks", FactCategory.TESTING, "testing.framework"),
            ("infrastructure", FactCategory.CONTAINERIZATION, "infrastructure.name"),
        )
        for attribute, category, key in mappings:
            for item in tuple(getattr(context, attribute, ()) or ())[:_MAX_COLLECTION_ITEMS]:
                name = getattr(item, "name", item if isinstance(item, str) else None)
                if isinstance(name, str) and name:
                    self.add_fact(category=category, key=key, value=name, source=FactSource.PROJECT_CONTEXT, confidence=confidence, evidence=(evidence,))
        for directory in tuple(getattr(context, "source_directories", ()) or ())[:_MAX_COLLECTION_ITEMS]:
            self.add_fact(category=FactCategory.PROJECT_STRUCTURE, key="source.directory", value=directory, source=FactSource.PROJECT_CONTEXT, confidence=confidence, evidence=(evidence,))
        for directory in tuple(getattr(context, "test_directories", ()) or ())[:_MAX_COLLECTION_ITEMS]:
            self.add_fact(category=FactCategory.PROJECT_STRUCTURE, key="test.directory", value=directory, source=FactSource.PROJECT_CONTEXT, confidence=confidence, evidence=(evidence,))
        return self.snapshot()

    def confirm_fact(self, fact_id: str, *, evidence: Sequence[FactEvidence], source: FactSource | str = FactSource.USER_PROVIDED) -> ProjectMemorySnapshot:
        self._ensure_writable()
        current = self._facts.get(fact_id)
        if current is None:
            raise ProjectMemoryValidationError("unknown fact_id")
        source = _coerce_enum(source, FactSource, "source")
        evidence = _bounded_evidence(evidence, self.limits)
        if not evidence:
            raise ProjectMemoryValidationError("confirmation requires evidence")
        self._sequence += 1
        confirmed = _replace_fact(current, confidence=FactConfidence.USER_CONFIRMED, status=FactStatus.ACTIVE, source=source, evidence=_merge_evidence(current.evidence, evidence, self.limits.max_evidence_per_fact), sequence=self._sequence)
        old = self._active_by_key.get((current.category, current.key))
        if old and old != fact_id:
            self._facts[old] = _replace_fact(self._facts[old], status=FactStatus.SUPERSEDED, sequence=self._sequence, conflict_with=(fact_id,))
        self._facts[fact_id] = confirmed
        self._active_by_key[(current.category, current.key)] = fact_id
        return self.snapshot()

    def invalidate_fact(self, fact_id: str, *, reason: str, source: FactSource | str = FactSource.VERIFICATION_RESULT) -> ProjectMemorySnapshot:
        self._ensure_writable()
        current = self._facts.get(fact_id)
        if current is None:
            raise ProjectMemoryValidationError("unknown fact_id")
        self._sequence += 1
        self._facts[fact_id] = _replace_fact(current, status=FactStatus.INVALID, sequence=self._sequence, evidence=_merge_evidence(current.evidence, (FactEvidence(_coerce_enum(source, FactSource, "source"), "invalidation", _bounded_text(reason, self.limits.max_evidence_text_length), verified=True),), self.limits.max_evidence_per_fact))
        if self._active_by_key.get((current.category, current.key)) == fact_id:
            self._active_by_key.pop((current.category, current.key), None)
        return self.snapshot()

    def snapshot(self) -> ProjectMemorySnapshot:
        facts = tuple(sorted(self._facts.values(), key=lambda item: (item.category.value, item.key, item.status.value, item.fact_id)))
        conflicts = tuple(item for item in facts if item.status in {FactStatus.CONFLICTED, FactStatus.REJECTED, FactStatus.SUPERSEDED})
        return ProjectMemorySnapshot(self.identity, facts, conflicts[: self.limits.max_conflict_records], self._load_status, self._sequence, self._evictions, tuple(self._warnings))

    def to_json(self) -> str:
        return self.snapshot().to_json()

    def close(self) -> ProjectMemorySnapshot:
        self._closed = True
        return self.snapshot()

    def _ensure_writable(self) -> None:
        if self._closed:
            raise ProjectMemoryClosedError("project memory is closed")

    def _compact(self) -> None:
        while len(self._facts) > self.limits.max_facts:
            candidates = [item for item in self._facts.values() if item.status is not FactStatus.ACTIVE]
            if not candidates:
                candidates = list(self._facts.values())
            victim = min(candidates, key=lambda item: (item.confidence, item.sequence, item.fact_id))
            self._facts.pop(victim.fact_id, None)
            if self._active_by_key.get((victim.category, victim.key)) == victim.fact_id:
                self._active_by_key.pop((victim.category, victim.key), None)
            self._evictions += 1
        if self._serialized_size() > self.limits.max_total_memory_bytes:
            removable = sorted(self._facts.values(), key=lambda item: (item.confidence, item.sequence, item.fact_id))
            for victim in removable:
                if self._serialized_size() <= self.limits.max_total_memory_bytes:
                    break
                self._facts.pop(victim.fact_id, None)
                if self._active_by_key.get((victim.category, victim.key)) == victim.fact_id:
                    self._active_by_key.pop((victim.category, victim.key), None)
                self._evictions += 1
        if self._evictions and "bounded eviction applied" not in self._warnings:
            self._warnings.append("bounded eviction applied")

    def _serialized_size(self) -> int:
        return len(self.to_json().encode("utf-8"))


class ProjectMemoryStore:
    """Atomic project-local JSON persistence with stale-write detection."""

    def __init__(self, project_root: Path | str, *, limits: ProjectMemoryLimits | None = None) -> None:
        self.identity = ProjectIdentity.for_root(project_root)
        self.limits = limits or ProjectMemoryLimits()
        self.path = Path(self.identity.project_root) / _STORAGE_DIRECTORY / _STORAGE_FILENAME
        self._loaded_digest: str | None = None

    @classmethod
    def for_project(cls, project_root: Path | str, *, limits: ProjectMemoryLimits | None = None) -> "ProjectMemoryStore":
        return cls(project_root, limits=limits)

    def empty(self) -> ProjectMemory:
        return ProjectMemory(self.identity, limits=self.limits)

    def load(self) -> ProjectMemoryLoadResult:
        try:
            self._validate_storage_location()
        except ProjectMemoryValidationError as exc:
            root = Path(self.identity.project_root)
            status = ProjectMemoryLoadStatus.MEMORY_UNAVAILABLE if not root.is_dir() else ProjectMemoryLoadStatus.MEMORY_INVALID
            return ProjectMemoryLoadResult(status, None, str(exc))
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            self._loaded_digest = None
            return ProjectMemoryLoadResult(ProjectMemoryLoadStatus.MEMORY_MISSING, None, "project memory file does not exist")
        except OSError as exc:
            return ProjectMemoryLoadResult(ProjectMemoryLoadStatus.MEMORY_UNAVAILABLE, None, str(exc))
        if len(raw) > self.limits.max_total_memory_bytes:
            return ProjectMemoryLoadResult(ProjectMemoryLoadStatus.MEMORY_INVALID, None, "project memory exceeds configured byte limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
            snapshot = _snapshot_from_dict(payload, self.limits)
        except UnicodeDecodeError as exc:
            return ProjectMemoryLoadResult(ProjectMemoryLoadStatus.MEMORY_CORRUPTED, None, f"invalid UTF-8: {exc}")
        except json.JSONDecodeError as exc:
            return ProjectMemoryLoadResult(ProjectMemoryLoadStatus.MEMORY_CORRUPTED, None, str(exc))
        except (ProjectMemoryValidationError, TypeError, ValueError) as exc:
            return ProjectMemoryLoadResult(ProjectMemoryLoadStatus.MEMORY_INVALID, None, str(exc))
        if snapshot.identity != self.identity:
            return ProjectMemoryLoadResult(ProjectMemoryLoadStatus.MEMORY_INVALID, None, "persisted project identity does not match this root")
        self._loaded_digest = _digest(raw)
        return ProjectMemoryLoadResult(ProjectMemoryLoadStatus.LOADED, ProjectMemory.from_snapshot(snapshot, limits=self.limits), None)

    def _validate_storage_location(self) -> None:
        root = Path(self.identity.project_root)
        if not root.is_dir():
            raise ProjectMemoryValidationError("project root must be an existing directory")
        directory = root / _STORAGE_DIRECTORY
        if directory.is_symlink() or self.path.is_symlink():
            raise ProjectMemoryValidationError("project memory storage must not use symlinks")

    def save(self, memory: ProjectMemory) -> Path:
        self._validate_storage_location()
        if not isinstance(memory, ProjectMemory) or memory.identity != self.identity:
            raise ProjectMemoryValidationError("memory identity does not match this ProjectMemoryStore")
        if memory._closed:
            raise ProjectMemoryClosedError("cannot persist closed project memory")
        if self.path.exists():
            current_digest = _digest(self.path.read_bytes())
            if self._loaded_digest is None or current_digest != self._loaded_digest:
                raise ProjectMemoryConflictError("project memory changed since this store loaded it")
        memory._load_status = ProjectMemoryLoadStatus.LOADED
        payload = memory.to_json().encode("utf-8")
        if len(payload) > self.limits.max_total_memory_bytes:
            raise ProjectMemoryValidationError("project memory exceeds configured byte limit")
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=directory, prefix=".project_memory.", suffix=".tmp", delete=False) as stream:
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


def _snapshot_from_dict(payload: Any, limits: ProjectMemoryLimits) -> ProjectMemorySnapshot:
    if not isinstance(payload, Mapping) or payload.get("format") != PROJECT_MEMORY_FORMAT:
        raise ProjectMemoryValidationError("unsupported Project Memory format")
    allowed_keys = {"format", "schema_version", "identity", "facts", "conflicts", "status", "sequence", "evictions", "warnings"}
    if set(payload) - allowed_keys:
        raise ProjectMemoryValidationError("unknown Project Memory fields are not accepted")
    if payload.get("schema_version") != PROJECT_MEMORY_SCHEMA_VERSION:
        raise ProjectMemoryValidationError("unsupported Project Memory schema version")
    identity_payload = payload.get("identity")
    if not isinstance(identity_payload, Mapping):
        raise ProjectMemoryValidationError("missing project identity")
    identity = ProjectIdentity(str(identity_payload.get("project_id", "")), str(identity_payload.get("project_root", "")), str(identity_payload.get("schema_version", "")))
    facts_payload = payload.get("facts", [])
    conflicts_payload = payload.get("conflicts", [])
    if not isinstance(facts_payload, list) or not isinstance(conflicts_payload, list):
        raise ProjectMemoryValidationError("facts and conflicts must be arrays")
    if len(facts_payload) > limits.max_facts or len(conflicts_payload) > limits.max_conflict_records:
        raise ProjectMemoryValidationError("persisted fact count exceeds configured limits")
    facts = tuple(_fact_from_dict(item, limits) for item in facts_payload)
    conflicts = tuple(_fact_from_dict(item, limits) for item in conflicts_payload)
    sequence = payload.get("sequence", 0)
    evictions = payload.get("evictions", 0)
    if not isinstance(sequence, int) or sequence < 0 or not isinstance(evictions, int) or evictions < 0:
        raise ProjectMemoryValidationError("invalid memory counters")
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise ProjectMemoryValidationError("invalid memory warnings")
    status = _coerce_enum(payload.get("status", ProjectMemoryLoadStatus.LOADED.value), ProjectMemoryLoadStatus, "status")
    return ProjectMemorySnapshot(identity, facts, conflicts, status, sequence, evictions, tuple(warnings[:_MAX_COLLECTION_ITEMS]))


def _fact_from_dict(payload: Any, limits: ProjectMemoryLimits) -> ProjectFact:
    if not isinstance(payload, Mapping):
        raise ProjectMemoryValidationError("fact must be an object")
    allowed_keys = {"fact_id", "category", "key", "value", "confidence", "status", "evidence", "source", "sequence", "conflict_with", "schema_version"}
    if set(payload) - allowed_keys:
        raise ProjectMemoryValidationError("unknown fact fields are not accepted")
    evidence_payload = payload.get("evidence", [])
    if not isinstance(evidence_payload, list):
        raise ProjectMemoryValidationError("fact evidence must be an array")
    evidence = tuple(_evidence_from_dict(item, limits) for item in evidence_payload)
    category = _coerce_enum(payload.get("category"), FactCategory, "category")
    source = _coerce_enum(payload.get("source"), FactSource, "source")
    confidence = _coerce_enum(payload.get("confidence"), FactConfidence, "confidence")
    status = _coerce_enum(payload.get("status"), FactStatus, "status")
    return ProjectFact(str(payload.get("fact_id", "")), category, str(payload.get("key", "")), _bounded_value(payload.get("value"), limits.max_fact_value_length, limits.max_metadata_size), confidence, status, evidence, source, int(payload.get("sequence", 0)), tuple(str(item) for item in payload.get("conflict_with", [])), str(payload.get("schema_version", "")))


def _evidence_from_dict(payload: Any, limits: ProjectMemoryLimits) -> FactEvidence:
    if not isinstance(payload, Mapping):
        raise ProjectMemoryValidationError("evidence must be an object")
    allowed_keys = {"source", "reference", "summary", "verified", "metadata"}
    if set(payload) - allowed_keys:
        raise ProjectMemoryValidationError("unknown evidence fields are not accepted")
    if not isinstance(payload.get("verified", False), bool):
        raise ProjectMemoryValidationError("evidence verified must be boolean")
    if not isinstance(payload.get("metadata", {}), Mapping):
        raise ProjectMemoryValidationError("evidence metadata must be a mapping")
    source = _coerce_enum(payload.get("source"), FactSource, "evidence source")
    return FactEvidence(source, _bounded_text(str(payload.get("reference", "")), limits.max_evidence_text_length), _bounded_text(str(payload.get("summary", "")), limits.max_evidence_text_length), payload.get("verified", False), _bounded_value(payload.get("metadata", {}), limits.max_metadata_size, limits.max_metadata_size))


def _coerce_enum(value: Any, enum_type: type[Enum], name: str):
    if isinstance(value, enum_type):
        return value
    try:
        if issubclass(enum_type, IntEnum) and isinstance(value, str):
            return enum_type[value]
        return enum_type(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectMemoryValidationError(f"invalid {name}: {value!r}") from exc


def _safe_key(value: Any, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit or ".." in value.split("/") or value.startswith("/"):
        raise ProjectMemoryValidationError("fact key is empty, unsafe, or oversized")
    if any(not (char.isalnum() or char in "._-/" ) for char in value):
        raise ProjectMemoryValidationError("fact key contains unsupported characters")
    return value.strip()


def _bounded_text(value: str, limit: int) -> str:
    value = _redact_text(value)
    if len(value) <= limit:
        return value
    marker = f"\n[truncated: kept_first_{limit}_chars]"
    return value[: max(0, limit - len(marker))] + marker


def _bounded_value(value: Any, limit: int, metadata_limit: int) -> Any:
    safe = _redact_value(value)
    if isinstance(safe, str):
        safe = _redact_text(safe)
        if len(safe) > limit:
            raise ProjectMemoryValidationError("scalar fact value exceeds configured length bound")
        return safe
    if isinstance(safe, (int, float, bool)) or safe is None:
        return safe
    if isinstance(safe, Mapping):
        encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > metadata_limit:
            raise ProjectMemoryValidationError("structured fact value exceeds metadata bound")
        return _freeze_value(safe)
    if isinstance(safe, list):
        if len(safe) > _MAX_COLLECTION_ITEMS:
            raise ProjectMemoryValidationError("fact list value exceeds item bound")
        return _freeze_value(safe)
    raise ProjectMemoryValidationError("unsupported project fact value type")


def _bounded_evidence(evidence: Sequence[FactEvidence], limits: ProjectMemoryLimits) -> tuple[FactEvidence, ...]:
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise ProjectMemoryValidationError("evidence must be a sequence")
    if len(evidence) > limits.max_evidence_per_fact:
        raise ProjectMemoryValidationError("too many evidence records")
    result: list[FactEvidence] = []
    for item in evidence:
        if not isinstance(item, FactEvidence):
            raise ProjectMemoryValidationError("evidence must contain FactEvidence")
        result.append(FactEvidence(item.source, _bounded_text(item.reference, limits.max_evidence_text_length), _bounded_text(item.summary, limits.max_evidence_text_length), item.verified, _bounded_value(item.metadata, limits.max_metadata_size, limits.max_metadata_size)))
    return tuple(result)


def _merge_evidence(left: Sequence[FactEvidence], right: Sequence[FactEvidence], limit: int) -> tuple[FactEvidence, ...]:
    values: dict[tuple[str, str, str], FactEvidence] = {}
    for item in tuple(left) + tuple(right):
        values[(item.source.value, item.reference, item.summary)] = item
    return tuple(values[key] for key in sorted(values)[:limit])


def _fact_id(category: FactCategory, key: str, value: Any) -> str:
    encoded = _canonical({"category": category.value, "key": key, "value": value})
    return "fact-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _canonical(value: Any) -> str:
    return json.dumps(_thaw_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _authority_score(confidence: FactConfidence, source: FactSource) -> int:
    source_weight = {
        FactSource.USER_PROVIDED: 50,
        FactSource.EXPLICIT_HOST_CONFIGURATION: 50,
        FactSource.CONFIGURATION: 45,
        FactSource.MANIFEST: 45,
        FactSource.PROJECT_CONTEXT: 40,
        FactSource.VERIFICATION_RESULT: 40,
        FactSource.TEST_RESULT: 30,
        FactSource.STRUCTURED_TOOL_RESULT: 20,
    }
    return int(confidence) * 100 + source_weight[source]


def _stronger_source(left: FactSource, right: FactSource) -> FactSource:
    order = {FactSource.USER_PROVIDED: 6, FactSource.EXPLICIT_HOST_CONFIGURATION: 6, FactSource.CONFIGURATION: 5, FactSource.MANIFEST: 5, FactSource.PROJECT_CONTEXT: 4, FactSource.VERIFICATION_RESULT: 4, FactSource.TEST_RESULT: 3, FactSource.STRUCTURED_TOOL_RESULT: 2}
    return right if order[right] > order[left] else left


def _replace_fact(fact: ProjectFact, **changes: Any) -> ProjectFact:
    fields = {
        "fact_id": fact.fact_id, "category": fact.category, "key": fact.key, "value": fact.value,
        "confidence": fact.confidence, "status": fact.status, "evidence": fact.evidence,
        "source": fact.source, "sequence": fact.sequence, "conflict_with": fact.conflict_with,
        "schema_version": fact.schema_version,
    }
    fields.update(changes)
    return ProjectFact(**fields)


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


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "PROJECT_MEMORY_FORMAT",
    "PROJECT_MEMORY_SCHEMA_VERSION",
    "FactCategory",
    "FactConfidence",
    "FactEvidence",
    "FactSource",
    "FactStatus",
    "ProjectFact",
    "ProjectIdentity",
    "ProjectMemory",
    "ProjectMemoryClosedError",
    "ProjectMemoryConflictError",
    "ProjectMemoryError",
    "ProjectMemoryLimits",
    "ProjectMemoryLoadResult",
    "ProjectMemoryLoadStatus",
    "ProjectMemorySnapshot",
    "ProjectMemoryStore",
    "ProjectMemoryValidationError",
]
