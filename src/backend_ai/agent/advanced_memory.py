from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any

from backend_ai.agent.recovery import NormalizedError, ErrorCategory


class MemoryScope(str, Enum):
    GLOBAL = "GLOBAL"
    PROJECT = "PROJECT"
    TASK = "TASK"
    SESSION = "SESSION"


class MemoryType(str, Enum):
    PROJECT_MEMORY = "PROJECT_MEMORY"
    TECHNICAL_MEMORY = "TECHNICAL_MEMORY"
    ERROR_MEMORY = "ERROR_MEMORY"
    SOLUTION_MEMORY = "SOLUTION_MEMORY"
    PREFERENCE_MEMORY = "PREFERENCE_MEMORY"
    TASK_MEMORY = "TASK_MEMORY"


class MemoryStatus(str, Enum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    REINFORCED = "REINFORCED"
    STALE = "STALE"
    SUPERSEDED = "SUPERSEDED"
    INVALIDATED = "INVALIDATED"
    ARCHIVED = "ARCHIVED"


class MemoryProvenance(str, Enum):
    USER = "USER"
    AGENT_OBSERVATION = "AGENT_OBSERVATION"
    TOOL_RESULT = "TOOL_RESULT"
    TEST_RESULT = "TEST_RESULT"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    SUCCESSFUL_TASK = "SUCCESSFUL_TASK"
    PROJECT_ANALYSIS = "PROJECT_ANALYSIS"
    IMPORTED_KNOWLEDGE = "IMPORTED_KNOWLEDGE"


class MemoryConfidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True, slots=True)
class AdvancedMemoryRecord:
    id: str
    memory_type: MemoryType
    scope: MemoryScope
    content: str
    project_id: str | None = None
    task_id: str | None = None
    source: MemoryProvenance = MemoryProvenance.AGENT_OBSERVATION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence: MemoryConfidence = MemoryConfidence.MEDIUM
    importance: float = 0.5
    relevance: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_accessed: str | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "scope": self.scope.value,
            "content": self.content,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "source": self.source.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confidence": self.confidence.value,
            "importance": self.importance,
            "relevance": self.relevance,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_accessed": self.last_accessed,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AdvancedMemoryStore:
    records: tuple[AdvancedMemoryRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"records": [r.to_dict() for r in self.records]}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    @classmethod
    def load(cls, path: Path) -> AdvancedMemoryStore:
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = []
            for item in data.get("records", []):
                rec = AdvancedMemoryRecord(
                    id=item["id"],
                    memory_type=MemoryType(item["memory_type"]),
                    scope=MemoryScope(item["scope"]),
                    content=item["content"],
                    project_id=item.get("project_id"),
                    task_id=item.get("task_id"),
                    source=MemoryProvenance(item.get("source", "AGENT_OBSERVATION")),
                    created_at=item.get("created_at", datetime.now(timezone.utc).isoformat()),
                    updated_at=item.get("updated_at", datetime.now(timezone.utc).isoformat()),
                    confidence=MemoryConfidence(item.get("confidence", "MEDIUM")),
                    importance=float(item.get("importance", 0.5)),
                    relevance=float(item.get("relevance", 0.0)),
                    usage_count=int(item.get("usage_count", 0)),
                    success_count=int(item.get("success_count", 0)),
                    failure_count=int(item.get("failure_count", 0)),
                    last_accessed=item.get("last_accessed"),
                    status=MemoryStatus(item.get("status", "ACTIVE")),
                    metadata=item.get("metadata", {}),
                )
                records.append(rec)
            return cls(tuple(records))
        except Exception:
            return cls()


class AdvancedMemorySystem:
    def __init__(self, store_path: Path | None = None) -> None:
        self.store_path = store_path
        self.store = AdvancedMemoryStore.load(store_path) if store_path else AdvancedMemoryStore()

    def add(self, record: AdvancedMemoryRecord) -> None:
        # Check duplication or contradiction
        updated = list(self.store.records)
        # Deduplication check by content similarity or ID
        exists = False
        for i, r in enumerate(updated):
            if r.id == record.id or r.content.strip().lower() == record.content.strip().lower():
                # Reinforce instead of duplicate
                updated[i] = replace_record(r, usage_count=r.usage_count + 1, updated_at=datetime.now(timezone.utc).isoformat(), status=MemoryStatus.REINFORCED)
                exists = True
                break
        if not exists:
            updated.append(record)
        self.store = AdvancedMemoryStore(tuple(updated))
        if self.store_path:
            self.store.save(self.store_path)

    def retrieve(self, query: str, *, project_id: str | None = None, scope: MemoryScope | None = None, max_results: int = 5) -> tuple[AdvancedMemoryRecord, ...]:
        query_terms = set(query.lower().split())
        scored = []
        for r in self.store.records:
            if r.status not in {MemoryStatus.ACTIVE, MemoryStatus.REINFORCED}:
                continue
            if scope and r.scope != scope and r.scope != MemoryScope.GLOBAL:
                continue
            if r.scope == MemoryScope.PROJECT and project_id and r.project_id and r.project_id != project_id:
                continue
            
            content_terms = set(r.content.lower().split())
            intersection = query_terms.intersection(content_terms)
            relevance = len(intersection) / max(len(query_terms), 1)
            if relevance == 0.0 and len(query_terms) > 0:
                # check substring
                if any(term in r.content.lower() for term in query_terms):
                    relevance = 0.3
            
            conf_weight = 1.0 if r.confidence == MemoryConfidence.HIGH else 0.7 if r.confidence == MemoryConfidence.MEDIUM else 0.4
            score = relevance * 0.5 + r.importance * 0.3 + conf_weight * 0.2
            if score > 0.1:
                scored.append((score, r))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(r for _, r in scored[:max_results])


def replace_record(rec: AdvancedMemoryRecord, **kwargs: Any) -> AdvancedMemoryRecord:
    data = rec.to_dict()
    data.update(kwargs)
    return AdvancedMemoryRecord(
        id=data["id"],
        memory_type=MemoryType(data["memory_type"]),
        scope=MemoryScope(data["scope"]),
        content=data["content"],
        project_id=data.get("project_id"),
        task_id=data.get("task_id"),
        source=MemoryProvenance(data["source"]),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        confidence=MemoryConfidence(data["confidence"]),
        importance=float(data["importance"]),
        relevance=float(data["relevance"]),
        usage_count=int(data["usage_count"]),
        success_count=int(data["success_count"]),
        failure_count=int(data["failure_count"]),
        last_accessed=data.get("last_accessed"),
        status=MemoryStatus(data["status"]),
        metadata=data["metadata"],
    )
