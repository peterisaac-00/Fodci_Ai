"""Deterministic quality and governance for the existing memory architecture.

Phase 9.6 is a read-only decision layer by default.  It evaluates normalized
retrieval candidates without replacing any memory store, and exposes explicit
invalidation helpers that delegate to the existing owner APIs when available.
It does not perform semantic similarity, model judging, network access, or
background work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from backend_ai.agent.memory_retrieval import MemoryRetrievalItem


class MemoryGovernanceError(ValueError):
    """Invalid governance input or policy."""


class QualityStatus(str, Enum):
    TRUSTED = "trusted"
    ACCEPTABLE = "acceptable"
    UNCERTAIN = "uncertain"
    STALE = "stale"
    INVALID = "invalid"
    CONFLICTED = "conflicted"
    DUPLICATE = "duplicate"


class FreshnessStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"


class ProvenanceStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    INVALID = "invalid"


class ConflictStatus(str, Enum):
    NONE = "none"
    DETECTED = "detected"
    UNRESOLVED = "unresolved"


class DuplicateStatus(str, Enum):
    UNIQUE = "unique"
    DUPLICATE = "duplicate"


class EligibilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


class SecurityStatus(str, Enum):
    CLEAR = "clear"
    VIOLATION = "violation"


class RetentionAction(str, Enum):
    RETAIN_ACTIVE = "retain_active"
    RETAIN_AGING = "retain_aging"
    ARCHIVE_CANDIDATE = "archive_candidate"
    PRESERVE_INVALIDATED = "preserve_invalidated"
    PRESERVE_DUPLICATE = "preserve_duplicate"
    PRESERVE_CONFLICTED = "preserve_conflicted"


_SECRET_KEY_RE = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie|database_url)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie|database_url)\s*(?:=|:)\s*(?!\[REDACTED\])[^,\s}\]]+",
    re.IGNORECASE,
)
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class FreshnessPolicy:
    """Source-aware freshness windows measured in seconds."""

    long_term_fresh_seconds: int = 7 * 24 * 60 * 60
    long_term_aging_seconds: int = 30 * 24 * 60 * 60
    long_term_stale_seconds: int = 90 * 24 * 60 * 60
    experience_fresh_seconds: int = 30 * 24 * 60 * 60
    experience_aging_seconds: int = 90 * 24 * 60 * 60
    experience_stale_seconds: int = 365 * 24 * 60 * 60

    def __post_init__(self) -> None:
        names = (
            "long_term_fresh_seconds", "long_term_aging_seconds", "long_term_stale_seconds",
            "experience_fresh_seconds", "experience_aging_seconds", "experience_stale_seconds",
        )
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise MemoryGovernanceError(f"{name} must be a positive integer")
        if not (self.long_term_fresh_seconds < self.long_term_aging_seconds < self.long_term_stale_seconds):
            raise MemoryGovernanceError("Long-Term freshness windows must be increasing")
        if not (self.experience_fresh_seconds < self.experience_aging_seconds < self.experience_stale_seconds):
            raise MemoryGovernanceError("Experience freshness windows must be increasing")

    def evaluate(self, source: str, timestamp: str | None, *, as_of: str | None = None) -> FreshnessStatus:
        source_value = _source_value(source)
        if source_value in {"short_term_memory", "project_memory"}:
            return FreshnessStatus.NOT_APPLICABLE
        if timestamp is None or not str(timestamp).strip() or as_of is None:
            return FreshnessStatus.UNKNOWN
        age = (_parse_timestamp(as_of) - _parse_timestamp(timestamp)).total_seconds()
        if age < 0:
            return FreshnessStatus.FRESH
        if source_value == "long_term_memory":
            fresh, aging, stale = self.long_term_fresh_seconds, self.long_term_aging_seconds, self.long_term_stale_seconds
        elif source_value == "experience_records":
            fresh, aging, stale = self.experience_fresh_seconds, self.experience_aging_seconds, self.experience_stale_seconds
        else:
            return FreshnessStatus.UNKNOWN
        if age <= fresh:
            return FreshnessStatus.FRESH
        if age <= aging:
            return FreshnessStatus.AGING
        if age <= stale:
            return FreshnessStatus.STALE
        return FreshnessStatus.STALE


@dataclass(frozen=True, slots=True)
class GovernancePolicy:
    """Explainable eligibility policy; retrieval status filters are explicit."""

    minimum_confidence: int = 1
    exclude_stale: bool = True
    exclude_conflicted: bool = True
    exclude_duplicates: bool = True
    require_provenance: bool = True
    allow_explicit_archived: bool = True
    freshness: FreshnessPolicy = field(default_factory=FreshnessPolicy)
    max_candidates: int = 512

    def __post_init__(self) -> None:
        if not isinstance(self.minimum_confidence, int) or isinstance(self.minimum_confidence, bool) or not 0 <= self.minimum_confidence <= 4:
            raise MemoryGovernanceError("minimum_confidence must be between 0 and 4")
        for name in ("exclude_stale", "exclude_conflicted", "exclude_duplicates", "require_provenance", "allow_explicit_archived"):
            if not isinstance(getattr(self, name), bool):
                raise MemoryGovernanceError(f"{name} must be boolean")
        if not isinstance(self.freshness, FreshnessPolicy):
            raise MemoryGovernanceError("freshness must be FreshnessPolicy")
        if not isinstance(self.max_candidates, int) or isinstance(self.max_candidates, bool) or not 0 < self.max_candidates <= 4096:
            raise MemoryGovernanceError("max_candidates is outside its bound")


@dataclass(frozen=True, slots=True)
class MemoryQualityAssessment:
    """Complete, explainable governance decision for one normalized memory item."""

    source: str
    memory_id: str
    project_id: str | None
    quality_status: QualityStatus
    source_confidence: int | None
    verification_status: VerificationStatus
    freshness_status: FreshnessStatus
    provenance_status: ProvenanceStatus
    conflict_status: ConflictStatus
    duplicate_status: DuplicateStatus
    security_status: SecurityStatus
    eligibility_status: EligibilityStatus
    retention_action: RetentionAction
    reasons: tuple[str, ...]
    timestamp: str | None = None

    @property
    def eligible(self) -> bool:
        return self.eligibility_status is EligibilityStatus.ELIGIBLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "memory_id": self.memory_id,
            "project_id": self.project_id,
            "quality_status": self.quality_status.value,
            "source_confidence": self.source_confidence,
            "verification_status": self.verification_status.value,
            "freshness_status": self.freshness_status.value,
            "provenance_status": self.provenance_status.value,
            "conflict_status": self.conflict_status.value,
            "duplicate_status": self.duplicate_status.value,
            "security_status": self.security_status.value,
            "eligibility_status": self.eligibility_status.value,
            "retention_action": self.retention_action.value,
            "reasons": list(self.reasons),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class GovernanceAudit:
    """Read-only aggregate audit over a bounded candidate collection."""

    total_memories_inspected: int
    eligible_memories: int
    fresh_memories: int
    aging_memories: int
    stale_memories: int
    invalidated_memories: int
    duplicates: int
    conflicts: int
    missing_provenance: int
    security_violations: int
    malformed_entries: int
    findings: tuple[str, ...]
    assessments: tuple[MemoryQualityAssessment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_memories_inspected": self.total_memories_inspected,
            "eligible_memories": self.eligible_memories,
            "fresh_memories": self.fresh_memories,
            "aging_memories": self.aging_memories,
            "stale_memories": self.stale_memories,
            "invalidated_memories": self.invalidated_memories,
            "duplicates": self.duplicates,
            "conflicts": self.conflicts,
            "missing_provenance": self.missing_provenance,
            "security_violations": self.security_violations,
            "malformed_entries": self.malformed_entries,
            "findings": list(self.findings),
            "assessments": [item.to_dict() for item in self.assessments],
        }


@dataclass(frozen=True, slots=True)
class GovernanceEvaluation:
    """Candidate evaluation plus the eligible subset used by retrieval."""

    eligible_items: tuple[Any, ...]
    assessments: tuple[MemoryQualityAssessment, ...]
    audit: GovernanceAudit
    deduplicated_count: int


@dataclass(frozen=True, slots=True)
class InvalidationResult:
    source: str
    memory_id: str
    applied: bool
    reason: str
    previous_status: str | None
    new_status: str | None
    preserved: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "memory_id": self.memory_id,
            "applied": self.applied,
            "reason": self.reason,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "preserved": self.preserved,
            "message": self.message,
        }


class MemoryGovernance:
    """Deterministic governance service over existing normalized memory items."""

    def __init__(self, *, policy: GovernancePolicy | None = None) -> None:
        self.policy = policy or GovernancePolicy()

    def assess(
        self,
        memory: "MemoryRetrievalItem",
        *,
        as_of: str | None = None,
        explicit_status: str | None = None,
        duplicate: bool = False,
        conflict: bool = False,
    ) -> MemoryQualityAssessment:
        source = _source_value(getattr(memory, "source", None))
        memory_id = getattr(memory, "memory_id", None)
        content = getattr(memory, "content", None)
        project_id = getattr(memory, "project_id", None)
        timestamp = getattr(memory, "timestamp", None)
        confidence = getattr(memory, "confidence", None)
        status = _normalized_status(getattr(memory, "status", None))
        metadata = getattr(memory, "metadata", {})
        reasons: list[str] = []
        malformed = not isinstance(source, str) or not source or not isinstance(memory_id, str) or not memory_id.strip() or not isinstance(content, str) or not content.strip() or not isinstance(metadata, Mapping)
        if malformed:
            reasons.append("malformed memory item")
        if isinstance(confidence, bool) or (confidence is not None and (not isinstance(confidence, int) or not 0 <= confidence <= 4)):
            malformed = True
            reasons.append("invalid confidence")
            confidence = None
        security = _security_status(content if isinstance(content, str) else "", metadata if isinstance(metadata, Mapping) else {})
        if security is SecurityStatus.VIOLATION:
            reasons.append("security policy violation")
        provenance = _provenance_status(source, memory_id, content, timestamp)
        if provenance is not ProvenanceStatus.SUFFICIENT:
            reasons.append("missing provenance" if provenance is ProvenanceStatus.INSUFFICIENT else "invalid provenance")
        verification = _verification_status(source, status, confidence, metadata if isinstance(metadata, Mapping) else {})
        try:
            freshness = self.policy.freshness.evaluate(source, timestamp, as_of=as_of)
        except MemoryGovernanceError:
            freshness = FreshnessStatus.UNKNOWN
            malformed = True
            reasons.append("invalid timestamp")
        if freshness is FreshnessStatus.STALE:
            reasons.append("memory expired freshness policy")
        elif freshness is FreshnessStatus.AGING:
            reasons.append("memory is aging")
        elif freshness is FreshnessStatus.UNKNOWN and source in {"long_term_memory", "experience_records"}:
            reasons.append("freshness could not be established")
        conflict_status = ConflictStatus.DETECTED if conflict or status in {"conflicted", "rejected", "superseded"} or _has_conflict_metadata(metadata) else ConflictStatus.NONE
        if conflict_status is not ConflictStatus.NONE:
            reasons.append("conflict detected")
        duplicate_status = DuplicateStatus.DUPLICATE if duplicate else DuplicateStatus.UNIQUE
        if duplicate:
            reasons.append("duplicate normalized identity")
        invalidated = status in {"invalid", "invalidated"} or _is_invalidated(metadata)
        if invalidated:
            reasons.append("memory invalidated")
        if confidence is None:
            reasons.append("missing source confidence")
        elif confidence < self.policy.minimum_confidence and not (explicit_status == "archived" and self.policy.allow_explicit_archived):
            reasons.append("insufficient confidence")
        if status and not _status_allowed(source, status, explicit_status=explicit_status, allow_explicit_archived=self.policy.allow_explicit_archived):
            reasons.append("status is not eligible for this source")
        if verification is VerificationStatus.UNVERIFIED and source == "experience_records":
            reasons.append("historical evidence is unverified")
        ineligible = (
            malformed
            or security is SecurityStatus.VIOLATION
            or provenance is not ProvenanceStatus.SUFFICIENT and self.policy.require_provenance
            or invalidated
            or (conflict_status is not ConflictStatus.NONE and self.policy.exclude_conflicted)
            or (duplicate and self.policy.exclude_duplicates)
            or (freshness is FreshnessStatus.STALE and self.policy.exclude_stale)
            or confidence is None
            or (confidence < self.policy.minimum_confidence and not (explicit_status == "archived" and self.policy.allow_explicit_archived))
            or (status is not None and not _status_allowed(source, status, explicit_status=explicit_status, allow_explicit_archived=self.policy.allow_explicit_archived))
        )
        quality = QualityStatus.INVALID if malformed or security is SecurityStatus.VIOLATION or invalidated or provenance is ProvenanceStatus.INVALID else QualityStatus.CONFLICTED if conflict_status is not ConflictStatus.NONE and self.policy.exclude_conflicted else QualityStatus.DUPLICATE if duplicate else QualityStatus.STALE if freshness is FreshnessStatus.STALE else QualityStatus.TRUSTED if verification is VerificationStatus.VERIFIED and not ineligible else QualityStatus.ACCEPTABLE if not ineligible else QualityStatus.UNCERTAIN
        retention = _retention_action(quality, freshness, invalidated, duplicate, conflict_status)
        if not reasons:
            reasons.append("verified and active" if quality is QualityStatus.TRUSTED else "memory meets governance policy")
        return MemoryQualityAssessment(
            source=source or "unknown",
            memory_id=memory_id if isinstance(memory_id, str) and memory_id else "unknown",
            project_id=project_id if isinstance(project_id, str) else None,
            quality_status=quality,
            source_confidence=confidence,
            verification_status=verification,
            freshness_status=freshness,
            provenance_status=provenance,
            conflict_status=conflict_status,
            duplicate_status=duplicate_status,
            security_status=security,
            eligibility_status=EligibilityStatus.INELIGIBLE if ineligible else EligibilityStatus.ELIGIBLE,
            retention_action=retention,
            reasons=tuple(dict.fromkeys(reasons)),
            timestamp=timestamp if isinstance(timestamp, str) else None,
        )

    def is_eligible(self, memory: "MemoryRetrievalItem", *, as_of: str | None = None, explicit_status: str | None = None) -> bool:
        return self.assess(memory, as_of=as_of, explicit_status=explicit_status).eligible

    def evaluate_candidates(
        self,
        memories: Sequence["MemoryRetrievalItem"],
        *,
        as_of: str | None = None,
        explicit_status: str | None = None,
    ) -> GovernanceEvaluation:
        if not isinstance(memories, Sequence) or isinstance(memories, (str, bytes)):
            raise MemoryGovernanceError("memories must be a bounded sequence")
        candidates = tuple(memories[: self.policy.max_candidates])
        duplicate_keys: dict[str, int] = {}
        for item in candidates:
            key = _normalized_content(getattr(item, "content", ""))
            duplicate_keys[key] = duplicate_keys.get(key, 0) + 1
        duplicate_canonical: dict[str, int] = {}
        grouped_indices: dict[str, list[int]] = {}
        for index, item in enumerate(candidates):
            key = _normalized_content(getattr(item, "content", ""))
            if key:
                grouped_indices.setdefault(key, []).append(index)
        for key, indices in grouped_indices.items():
            duplicate_canonical[key] = min(indices, key=lambda index: _duplicate_priority(candidates[index], index))
        conflict_groups: dict[tuple[str, str, str], set[str]] = {}
        for item in candidates:
            group = _conflict_group(item)
            if group is not None:
                conflict_groups.setdefault(group, set()).add(_normalized_content(getattr(item, "content", "")))
        assessments: list[MemoryQualityAssessment] = []
        eligible: list[Any] = []
        deduplicated = 0
        for index, item in enumerate(candidates):
            key = _normalized_content(getattr(item, "content", ""))
            is_duplicate = bool(key) and duplicate_keys.get(key, 0) > 1 and duplicate_canonical.get(key) != index
            if is_duplicate:
                deduplicated += 1
            group = _conflict_group(item)
            has_conflict = group is not None and len(conflict_groups.get(group, set())) > 1
            assessment = self.assess(item, as_of=as_of, explicit_status=explicit_status, duplicate=is_duplicate, conflict=has_conflict)
            assessments.append(assessment)
            if assessment.eligible:
                eligible.append(item)
        audit = self._audit_from_assessments(tuple(assessments))
        return GovernanceEvaluation(tuple(eligible), tuple(assessments), audit, deduplicated)

    def audit(self, memories: Sequence["MemoryRetrievalItem"], *, as_of: str | None = None, explicit_status: str | None = None) -> GovernanceAudit:
        return self.evaluate_candidates(memories, as_of=as_of, explicit_status=explicit_status).audit

    def explain(self, memory: "MemoryRetrievalItem", *, as_of: str | None = None, explicit_status: str | None = None) -> tuple[str, ...]:
        return self.assess(memory, as_of=as_of, explicit_status=explicit_status).reasons

    def retention_evaluate(self, memory: "MemoryRetrievalItem", *, as_of: str | None = None, explicit_status: str | None = None) -> RetentionAction:
        return self.assess(memory, as_of=as_of, explicit_status=explicit_status).retention_action

    def invalidate(self, owner: object, memory_id: str, *, reason: str) -> InvalidationResult:
        """Explicitly invalidate through an existing owner API and preserve history."""

        if not isinstance(memory_id, str) or not memory_id.strip():
            raise MemoryGovernanceError("memory_id must contain text")
        safe_reason = _safe_reason(reason)
        from backend_ai.agent.experience_records import ExperienceRecords
        from backend_ai.agent.long_term_memory import LongTermMemory
        from backend_ai.agent.project_memory import ProjectMemory

        if isinstance(owner, ProjectMemory):
            current = next((item for item in owner.snapshot().facts if item.fact_id == memory_id), None)
            if current is None:
                return InvalidationResult("project_memory", memory_id, False, safe_reason, None, None, True, "project fact was not found")
            owner.invalidate_fact(memory_id, reason=safe_reason)
            return InvalidationResult("project_memory", memory_id, True, safe_reason, current.status.value, "INVALID", True, "project fact invalidated through ProjectMemory.invalidate_fact")
        if isinstance(owner, LongTermMemory):
            current = owner.get(memory_id, track_access=False)
            if current is None:
                return InvalidationResult("long_term_memory", memory_id, False, safe_reason, None, None, True, "long-term memory entry was not found")
            metadata = dict(current.metadata)
            metadata["governance_invalidation_reason"] = safe_reason
            metadata["governance_invalidated"] = True
            owner.update(memory_id, status="invalidated", metadata=metadata)
            return InvalidationResult("long_term_memory", memory_id, True, safe_reason, current.status.value, "invalidated", True, "long-term memory entry invalidated through LongTermMemory.update")
        if isinstance(owner, ExperienceRecords):
            current = owner.get(memory_id)
            if current is None:
                return InvalidationResult("experience_records", memory_id, False, safe_reason, None, None, True, "experience record was not found")
            updated = owner.invalidate(memory_id, reason=safe_reason)
            return InvalidationResult("experience_records", memory_id, True, safe_reason, current.status.value, updated.status.value, True, "experience record invalidated through ExperienceRecords.invalidate")
        return InvalidationResult("unknown", memory_id, False, safe_reason, None, None, True, "owner does not expose an approved invalidation API")

    @staticmethod
    def _audit_from_assessments(assessments: tuple[MemoryQualityAssessment, ...]) -> GovernanceAudit:
        findings: list[str] = []
        for item in assessments:
            for reason in item.reasons:
                findings.append(f"{item.source}:{item.memory_id}: {reason}")
        return GovernanceAudit(
            total_memories_inspected=len(assessments),
            eligible_memories=sum(item.eligible for item in assessments),
            fresh_memories=sum(item.freshness_status is FreshnessStatus.FRESH for item in assessments),
            aging_memories=sum(item.freshness_status is FreshnessStatus.AGING for item in assessments),
            stale_memories=sum(item.freshness_status is FreshnessStatus.STALE for item in assessments),
            invalidated_memories=sum("memory invalidated" in item.reasons for item in assessments),
            duplicates=sum(item.duplicate_status is DuplicateStatus.DUPLICATE for item in assessments),
            conflicts=sum(item.conflict_status is not ConflictStatus.NONE for item in assessments),
            missing_provenance=sum(item.provenance_status is not ProvenanceStatus.SUFFICIENT for item in assessments),
            security_violations=sum(item.security_status is SecurityStatus.VIOLATION for item in assessments),
            malformed_entries=sum("malformed memory item" in item.reasons or "invalid confidence" in item.reasons for item in assessments),
            findings=tuple(dict.fromkeys(findings)),
            assessments=assessments,
        )


def _source_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw) if raw is not None else ""


def _normalized_status(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw).casefold()


def _status_allowed(source: str, status: str, *, explicit_status: str | None, allow_explicit_archived: bool) -> bool:
    if status in {"invalid", "invalidated", "rejected", "superseded", "conflicted", "memory_invalid"}:
        return False
    if source == "project_memory":
        return status == "active"
    if source == "long_term_memory":
        return status == "active" or (status == "archived" and explicit_status == "archived" and allow_explicit_archived)
    if source == "experience_records":
        return status in {"started", "running", "completed", "failed", "cancelled"}
    if source == "short_term_memory":
        return status in {"active", "updated", "closed", "memory_limit_reached", "success", "failed", "pass"}
    return False


def _verification_status(source: str, status: str | None, confidence: int | None, metadata: Mapping[str, Any]) -> VerificationStatus:
    if source == "short_term_memory":
        return VerificationStatus.VERIFIED if metadata.get("information_kind") == "AUTHORITATIVE" else VerificationStatus.PARTIAL
    if source == "project_memory":
        return VerificationStatus.VERIFIED if metadata.get("evidence_count", 0) and (confidence or 0) >= 3 else VerificationStatus.PARTIAL
    if source == "long_term_memory":
        return VerificationStatus.VERIFIED if (confidence or 0) >= 3 else VerificationStatus.UNVERIFIED
    if source == "experience_records":
        return VerificationStatus.VERIFIED if bool(metadata.get("verified")) else VerificationStatus.UNVERIFIED
    return VerificationStatus.UNVERIFIED


def _provenance_status(source: str, memory_id: Any, content: Any, timestamp: Any) -> ProvenanceStatus:
    if not isinstance(memory_id, str) or not memory_id.strip() or not isinstance(content, str) or not content.strip():
        return ProvenanceStatus.INVALID
    if source not in {"short_term_memory", "project_memory", "long_term_memory", "experience_records"}:
        return ProvenanceStatus.INVALID
    if source in {"long_term_memory", "experience_records"} and (not isinstance(timestamp, str) or not timestamp.strip()):
        return ProvenanceStatus.INSUFFICIENT
    return ProvenanceStatus.SUFFICIENT


def _security_status(content: str, metadata: Mapping[str, Any]) -> SecurityStatus:
    if _SECRET_ASSIGNMENT_RE.search(content) or _PRIVATE_KEY_RE.search(content):
        return SecurityStatus.VIOLATION
    def walk(value: Any, key: str = "") -> bool:
        if key and _SECRET_KEY_RE.search(key):
            return not (isinstance(value, str) and value == "[REDACTED]")
        if isinstance(value, Mapping):
            return any(walk(item, str(name)) for name, item in value.items())
        if isinstance(value, (list, tuple)):
            return any(walk(item, key) for item in value)
        if isinstance(value, str):
            return bool(_SECRET_ASSIGNMENT_RE.search(value) or _PRIVATE_KEY_RE.search(value))
        return False
    return SecurityStatus.VIOLATION if walk(metadata) else SecurityStatus.CLEAR


def _has_conflict_metadata(metadata: Any) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    value = metadata.get("conflict_with")
    return bool(value)


def _is_invalidated(metadata: Any) -> bool:
    return isinstance(metadata, Mapping) and metadata.get("governance_invalidated") is True


def _conflict_group(item: Any) -> tuple[str, str, str] | None:
    source = _source_value(getattr(item, "source", None))
    metadata = getattr(item, "metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    if source == "project_memory" and metadata.get("key"):
        return (source, str(getattr(item, "project_id", None)), str(metadata["key"]).casefold())
    if source == "long_term_memory" and metadata.get("category") and metadata.get("topic"):
        return (source, str(metadata["category"]).casefold(), str(metadata["topic"]).casefold())
    if metadata.get("governance_key"):
        return (source, str(getattr(item, "project_id", None)), str(metadata["governance_key"]).casefold())
    return None


def _normalized_content(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(_WORD_RE.findall(value.casefold()))


def _duplicate_priority(item: Any, index: int) -> tuple[int, int, int, str, str, int]:
    source = _source_value(getattr(item, "source", None))
    source_priority = {"project_memory": 4, "long_term_memory": 3, "experience_records": 2, "short_term_memory": 1}.get(source, 0)
    confidence = getattr(item, "confidence", None)
    confidence_value = int(confidence) if isinstance(confidence, int) and not isinstance(confidence, bool) else -1
    metadata = getattr(item, "metadata", {})
    verified = int(isinstance(metadata, Mapping) and metadata.get("verified") is True)
    timestamp = str(getattr(item, "timestamp", None) or "")
    memory_id = str(getattr(item, "memory_id", ""))
    return (-confidence_value, -verified, -source_priority, timestamp, memory_id, index)


def _parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise MemoryGovernanceError("timestamp must contain text")
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _retention_action(quality: QualityStatus, freshness: FreshnessStatus, invalidated: bool, duplicate: bool, conflict: ConflictStatus) -> RetentionAction:
    if invalidated or quality is QualityStatus.INVALID:
        return RetentionAction.PRESERVE_INVALIDATED
    if duplicate:
        return RetentionAction.PRESERVE_DUPLICATE
    if conflict is not ConflictStatus.NONE or quality is QualityStatus.CONFLICTED:
        return RetentionAction.PRESERVE_CONFLICTED
    if freshness is FreshnessStatus.STALE or quality is QualityStatus.STALE:
        return RetentionAction.ARCHIVE_CANDIDATE
    if freshness is FreshnessStatus.AGING:
        return RetentionAction.RETAIN_AGING
    return RetentionAction.RETAIN_ACTIVE


def _safe_reason(reason: Any) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise MemoryGovernanceError("invalidation reason must contain text")
    if len(reason.strip()) > 512:
        raise MemoryGovernanceError("invalidation reason exceeds 512 characters")
    if _SECRET_ASSIGNMENT_RE.search(reason) or _PRIVATE_KEY_RE.search(reason):
        raise MemoryGovernanceError("invalidation reason contains prohibited secret material")
    return reason.strip()


__all__ = [
    "ConflictStatus",
    "DuplicateStatus",
    "EligibilityStatus",
    "FreshnessPolicy",
    "FreshnessStatus",
    "GovernanceAudit",
    "GovernanceEvaluation",
    "GovernancePolicy",
    "InvalidationResult",
    "MemoryGovernance",
    "MemoryGovernanceError",
    "MemoryQualityAssessment",
    "ProvenanceStatus",
    "QualityStatus",
    "RetentionAction",
    "SecurityStatus",
    "VerificationStatus",
]
