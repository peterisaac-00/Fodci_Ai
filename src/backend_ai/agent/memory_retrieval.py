"""Unified deterministic retrieval over the existing memory subsystems."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import json
import re
from types import MappingProxyType
from typing import Any

from backend_ai.agent.experience_records import ExperienceRecord, ExperienceRecords
from backend_ai.agent.long_term_memory import LongTermMemory, LongTermMemoryCategory, LongTermMemoryConfidence, LongTermMemoryStatus
from backend_ai.agent.project_memory import FactConfidence, FactStatus, ProjectMemorySnapshot
from backend_ai.agent.short_term_memory import MemoryImportance, MemorySnapshot, _redact_text


class MemoryRetrievalError(ValueError):
    """Invalid retrieval request or configured retrieval bound."""


class RetrievalSource(str, Enum):
    SHORT_TERM_MEMORY = "short_term_memory"
    PROJECT_MEMORY = "project_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    EXPERIENCE_RECORDS = "experience_records"


@dataclass(frozen=True, slots=True)
class MemoryRetrievalLimits:
    max_results: int = 32
    max_results_per_source: int = 16
    max_total_characters: int = 12_288
    max_query_length: int = 512
    max_reason_length: int = 512
    max_metadata_size: int = 2_048

    def __post_init__(self) -> None:
        ceilings = {
            "max_results": 256,
            "max_results_per_source": 128,
            "max_total_characters": 1_048_576,
            "max_query_length": 16_384,
            "max_reason_length": 4_096,
            "max_metadata_size": 65_536,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > ceiling:
                raise MemoryRetrievalError(f"{name} is outside its configured bound")


@dataclass(frozen=True, slots=True)
class MemoryRetrievalRequest:
    query: str
    sources: tuple[RetrievalSource, ...]
    project_id: str | None = None
    project_root: str | None = None
    short_term_memory: MemorySnapshot | None = None
    project_memory: ProjectMemorySnapshot | None = None
    long_term_memory: LongTermMemory | None = None
    experience_records: ExperienceRecords | None = None
    category: str | None = None
    status: str | None = None
    confidence_threshold: int | None = None
    max_results: int = 32
    max_results_per_source: int = 16
    max_total_characters: int = 12_288

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise MemoryRetrievalError("query must contain searchable text")
        if not isinstance(self.sources, tuple) or not self.sources or any(not isinstance(item, RetrievalSource) for item in self.sources):
            raise MemoryRetrievalError("sources must be a non-empty tuple of RetrievalSource")
        if len(set(self.sources)) != len(self.sources):
            raise MemoryRetrievalError("sources must not contain duplicates")
        if self.project_id is not None and (not isinstance(self.project_id, str) or not self.project_id.strip()):
            raise MemoryRetrievalError("project_id must contain text when provided")
        if self.project_root is not None and (not isinstance(self.project_root, str) or not self.project_root.strip()):
            raise MemoryRetrievalError("project_root must contain text when provided")
        if self.confidence_threshold is not None and (not isinstance(self.confidence_threshold, int) or isinstance(self.confidence_threshold, bool) or not 0 <= self.confidence_threshold <= 4):
            raise MemoryRetrievalError("confidence_threshold must be between 0 and 4")
        MemoryRetrievalLimits(self.max_results, self.max_results_per_source, self.max_total_characters, max(1, len(self.query)))
        if self.category is not None and (not isinstance(self.category, str) or not self.category.strip()):
            raise MemoryRetrievalError("category must contain text when provided")
        if self.status is not None and (not isinstance(self.status, str) or not self.status.strip()):
            raise MemoryRetrievalError("status must contain text when provided")


@dataclass(frozen=True, slots=True)
class MemoryRetrievalItem:
    source: RetrievalSource
    memory_id: str
    content: str
    relevance_score: float
    confidence: int | None
    status: str | None
    timestamp: str | None
    metadata: Mapping[str, Any]
    retrieval_reason: str
    project_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, RetrievalSource):
            raise MemoryRetrievalError("result source must be RetrievalSource")
        for value, name in ((self.memory_id, "memory_id"), (self.content, "content"), (self.retrieval_reason, "retrieval_reason")):
            if not isinstance(value, str) or not value.strip():
                raise MemoryRetrievalError(f"{name} must contain text")
        if not isinstance(self.relevance_score, (int, float)) or isinstance(self.relevance_score, bool) or not 0.0 <= float(self.relevance_score) <= 1.0:
            raise MemoryRetrievalError("relevance_score must be between 0 and 1")
        if self.confidence is not None and (not isinstance(self.confidence, int) or isinstance(self.confidence, bool) or not 0 <= self.confidence <= 4):
            raise MemoryRetrievalError("confidence must be between 0 and 4")
        object.__setattr__(self, "metadata", _freeze_value(_redact_value(dict(self.metadata))))

    @property
    def normalized_content(self) -> str:
        return _normalize(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source.value, "memory_id": self.memory_id, "content": _redact_text(self.content), "relevance_score": round(float(self.relevance_score), 8), "confidence": self.confidence, "status": self.status, "timestamp": self.timestamp, "metadata": _thaw_value(self.metadata), "retrieval_reason": self.retrieval_reason, "project_id": self.project_id}


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostic:
    source: RetrievalSource
    status: str
    message: str | None = None
    candidate_count: int = 0
    returned_count: int = 0
    filtered_count: int = 0
    deduplicated_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source.value, "status": self.status, "message": self.message, "candidate_count": self.candidate_count, "returned_count": self.returned_count, "filtered_count": self.filtered_count, "deduplicated_count": self.deduplicated_count}


@dataclass(frozen=True, slots=True)
class MemoryRetrievalResult:
    query: str
    items: tuple[MemoryRetrievalItem, ...]
    context: str
    diagnostics: tuple[RetrievalDiagnostic, ...]
    queried_sources: tuple[RetrievalSource, ...]
    context_characters: int
    deduplicated_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "items": [item.to_dict() for item in self.items], "context": self.context, "diagnostics": [item.to_dict() for item in self.diagnostics], "queried_sources": [item.value for item in self.queried_sources], "context_characters": self.context_characters, "deduplicated_count": self.deduplicated_count}


class MemoryRetrieval:
    """Explicit orchestration layer; it never reads memory storage files."""

    def retrieve(self, request: MemoryRetrievalRequest) -> MemoryRetrievalResult:
        if not isinstance(request, MemoryRetrievalRequest):
            raise MemoryRetrievalError("request must be MemoryRetrievalRequest")
        all_items: list[MemoryRetrievalItem] = []
        diagnostics: list[RetrievalDiagnostic] = []
        for source in request.sources:
            try:
                candidates = self._source_items(source, request)
                filtered = [item for item in candidates if self._matches(item, request)]
                ranked = sorted(filtered, key=lambda item: self._ranking_key(item, request.query), reverse=True)[:request.max_results_per_source]
                all_items.extend(ranked)
                diagnostics.append(RetrievalDiagnostic(source, "AVAILABLE", candidate_count=len(candidates), filtered_count=len(candidates) - len(filtered), returned_count=len(ranked)))
            except Exception as exc:  # source failure is isolated and observable
                diagnostics.append(RetrievalDiagnostic(source, "FAILED", _safe_message(exc)))
        deduped: list[MemoryRetrievalItem] = []
        seen: set[str] = set()
        for item in sorted(all_items, key=lambda candidate: self._ranking_key(candidate, request.query), reverse=True):
            key = item.normalized_content
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        deduplicated_count = len(all_items) - len(deduped)
        selected: list[MemoryRetrievalItem] = []
        per_source: dict[RetrievalSource, int] = {}
        total_chars = 0
        for item in deduped:
            if len(selected) >= request.max_results:
                break
            if per_source.get(item.source, 0) >= request.max_results_per_source:
                continue
            candidate_items = tuple(selected + [item])
            try:
                candidate_context = self.render_context(candidate_items, max_total_characters=request.max_total_characters)
            except MemoryRetrievalError:
                if not selected:
                    raise
                break
            selected.append(item)
            per_source[item.source] = per_source.get(item.source, 0) + 1
            total_chars = len(candidate_context)
        context = self.render_context(tuple(selected), max_total_characters=request.max_total_characters)
        diagnostics = tuple(replace_diagnostic(item, returned_count=sum(1 for result in selected if result.source is item.source), deduplicated_count=deduplicated_count if item.status == "AVAILABLE" else 0) for item in diagnostics)
        return MemoryRetrievalResult(request.query, tuple(selected), context, diagnostics, request.sources, len(context), deduplicated_count)

    @staticmethod
    def render_context(items: Sequence[MemoryRetrievalItem], *, max_total_characters: int = 12_288) -> str:
        if not isinstance(max_total_characters, int) or max_total_characters <= 0:
            raise MemoryRetrievalError("max_total_characters must be positive")
        groups: dict[RetrievalSource, list[str]] = {source: [] for source in RetrievalSource}
        for item in items:
            groups[item.source].append(f"- id={item.memory_id}; confidence={item.confidence}; status={item.status}; content={_redact_text(item.content)}")
        sections = []
        for source in RetrievalSource:
            if groups[source]:
                sections.append(f"[{source.value.upper()}]\n" + "\n".join(groups[source]))
        context = "\n\n".join(sections)
        if len(context) > max_total_characters:
            raise MemoryRetrievalError("selected retrieval context exceeds max_total_characters")
        return context

    def _source_items(self, source: RetrievalSource, request: MemoryRetrievalRequest) -> list[MemoryRetrievalItem]:
        if source is RetrievalSource.SHORT_TERM_MEMORY:
            return self._short_term(request.short_term_memory, request)
        if source is RetrievalSource.PROJECT_MEMORY:
            return self._project(request.project_memory, request)
        if source is RetrievalSource.LONG_TERM_MEMORY:
            return self._long_term(request.long_term_memory, request)
        return self._experiences(request.experience_records, request)

    def _short_term(self, snapshot: MemorySnapshot | None, request: MemoryRetrievalRequest) -> list[MemoryRetrievalItem]:
        if snapshot is None:
            raise MemoryRetrievalError("short-term memory snapshot was not supplied")
        if request.project_id is not None and snapshot.project_id not in {None, request.project_id}:
            return []
        records: list[tuple[str, Any]] = [("objective", snapshot.objective)]
        records.extend(("requirement", value) for value in snapshot.requirements)
        records.extend(("constraint", value) for value in snapshot.constraints)
        for category in (snapshot.observations, snapshot.tool_records, snapshot.test_records, snapshot.failure_records, snapshot.fix_records, snapshot.verification_records):
            records.extend((record.category, record) for record in category)
        items: list[MemoryRetrievalItem] = []
        for index, (kind, value) in enumerate(records):
            if isinstance(value, str):
                content = value
                metadata = {"kind": kind, "lifecycle": snapshot.lifecycle.value}
                status = snapshot.status.value
            else:
                content = value.summary
                metadata = {"kind": kind, "category": value.category, "operation": value.operation, "information_kind": value.information_kind.value}
                status = value.status or snapshot.status.value
            items.append(MemoryRetrievalItem(RetrievalSource.SHORT_TERM_MEMORY, f"{snapshot.task_id}:st-{index:04d}", _redact_text(content), 0.0, _importance_confidence(value), status, None, metadata, "current task/session context", snapshot.project_id))
        return items

    def _project(self, snapshot: ProjectMemorySnapshot | None, request: MemoryRetrievalRequest) -> list[MemoryRetrievalItem]:
        if snapshot is None:
            raise MemoryRetrievalError("project memory snapshot was not supplied")
        if request.project_id is not None and snapshot.identity.project_id != request.project_id:
            raise MemoryRetrievalError("project identity does not match request")
        if request.project_root is not None and snapshot.identity.project_root != request.project_root:
            raise MemoryRetrievalError("project root does not match request")
        facts = snapshot.active_facts if request.status is None else tuple(snapshot.facts) + tuple(snapshot.conflicts)
        return [MemoryRetrievalItem(RetrievalSource.PROJECT_MEMORY, fact.fact_id, _fact_content(fact), 0.0, int(fact.confidence), fact.status.value, None, {"key": fact.key, "category": fact.category.value, "source": fact.source.value, "evidence_count": len(fact.evidence)}, "verified project fact", snapshot.identity.project_id) for fact in facts]

    def _long_term(self, memory: LongTermMemory | None, request: MemoryRetrievalRequest) -> list[MemoryRetrievalItem]:
        if memory is None:
            raise MemoryRetrievalError("long-term memory owner was not supplied")
        category = request.category
        if request.status is None or request.status == LongTermMemoryStatus.ACTIVE.value:
            entries = memory.search(request.query, category=category, limit=request.max_results_per_source)
        else:
            entries = memory.list(category=category, status=request.status)
        return [MemoryRetrievalItem(RetrievalSource.LONG_TERM_MEMORY, entry.entry_id, entry.content, 0.0, int(entry.confidence), entry.status.value, entry.updated_at, {"category": entry.category.value, "source": entry.source.value, "access_count": entry.access_count, "conflict_with": list(entry.conflict_with)}, "global reusable knowledge", None) for entry in entries]

    def _experiences(self, records: ExperienceRecords | None, request: MemoryRetrievalRequest) -> list[MemoryRetrievalItem]:
        if records is None:
            raise MemoryRetrievalError("experience records owner was not supplied")
        experiences = records.list(project_id=request.project_id, status=request.status)
        items: list[MemoryRetrievalItem] = []
        for record in experiences:
            content = _experience_content(record)
            items.append(MemoryRetrievalItem(RetrievalSource.EXPERIENCE_RECORDS, record.experience_id, content, 0.0, _experience_confidence(record), record.status.value, record.completed_at or record.started_at, {"outcome": record.outcome.value if record.outcome else None, "attempts": len(record.attempts), "verified": record.verification is not None}, "historical execution evidence; not general knowledge", record.project_identity.project_id if record.project_identity else None))
        return items

    def _matches(self, item: MemoryRetrievalItem, request: MemoryRetrievalRequest) -> bool:
        if request.category is not None and str(item.metadata.get("category", "")) != request.category:
            return False
        if request.status is not None and item.status != request.status:
            return False
        if request.confidence_threshold is not None and (item.confidence is None or item.confidence < request.confidence_threshold):
            return False
        return bool(_tokens(request.query) & _tokens(item.content))

    @staticmethod
    def _ranking_key(item: MemoryRetrievalItem, query: str) -> tuple[float, int, int, str, str]:
        tokens = _tokens(query)
        content_tokens = _tokens(item.content)
        overlap = len(tokens & content_tokens) / max(1, len(tokens))
        exact = 0.15 if _normalize(query) in _normalize(item.content) else 0.0
        source_prior = {RetrievalSource.PROJECT_MEMORY: 0.12, RetrievalSource.LONG_TERM_MEMORY: 0.10, RetrievalSource.EXPERIENCE_RECORDS: 0.08, RetrievalSource.SHORT_TERM_MEMORY: 0.06}[item.source]
        status_bonus = 0.05 if str(item.status).lower() in {"active", "completed", "success", "verified", "pass"} else 0.0
        score = min(1.0, overlap * 0.68 + exact + source_prior + status_bonus)
        return (score, item.confidence or 0, 1 if status_bonus else 0, item.timestamp or "", item.memory_id)


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"\w+", value.casefold(), flags=re.UNICODE) if len(token) > 1}


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _fact_content(fact: Any) -> str:
    value = json.dumps(_thaw_value(fact.value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) if not isinstance(fact.value, str) else fact.value
    evidence = "; ".join(item.summary for item in fact.evidence[:3])
    return _redact_text(f"{fact.key}: {value}. Evidence: {evidence}")


def _experience_content(record: ExperienceRecord) -> str:
    parts = [record.task]
    if record.final_solution:
        parts.append("Solution: " + record.final_solution)
    if record.final_summary:
        parts.append("Summary: " + record.final_summary)
    if record.verification:
        parts.append("Verification: " + record.verification.summary)
    for attempt in record.attempts:
        parts.extend(error.summary for error in attempt.errors[:3])
        parts.extend(correction.summary for correction in attempt.corrections[:3])
    return _redact_text(". ".join(parts))


def _importance_confidence(value: Any) -> int | None:
    if isinstance(value, str):
        return None
    importance = getattr(value, "importance", None)
    if not isinstance(importance, MemoryImportance):
        return None
    return min(4, int(importance) + 1)


def _experience_confidence(record: ExperienceRecord) -> int | None:
    if record.outcome is not None and record.outcome.value == "success" and record.verification is not None:
        return 4
    if record.verification is not None:
        return 3
    return 1


def _safe_message(exc: Exception) -> str:
    return _redact_text(f"{type(exc).__name__}: {exc}")[:512]


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_value(item) for item in value]
    return value


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def replace_diagnostic(diagnostic: RetrievalDiagnostic, **changes: Any) -> RetrievalDiagnostic:
    return RetrievalDiagnostic(diagnostic.source, diagnostic.status, diagnostic.message, diagnostic.candidate_count, diagnostic.returned_count, diagnostic.filtered_count, changes.get("deduplicated_count", diagnostic.deduplicated_count))


__all__ = [
    "MemoryRetrieval",
    "MemoryRetrievalError",
    "MemoryRetrievalItem",
    "MemoryRetrievalLimits",
    "MemoryRetrievalRequest",
    "MemoryRetrievalResult",
    "RetrievalDiagnostic",
    "RetrievalSource",
]
