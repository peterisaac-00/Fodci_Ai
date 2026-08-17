from __future__ import annotations

from backend_ai.agent.experience_records import ExperienceProjectIdentity, ExperienceRecords, ExperienceVerification
from backend_ai.agent.long_term_memory import LongTermMemory
from backend_ai.agent.memory_governance import (
    ConflictStatus,
    DuplicateStatus,
    EligibilityStatus,
    FreshnessPolicy,
    FreshnessStatus,
    GovernancePolicy,
    MemoryGovernance,
    QualityStatus,
    RetentionAction,
    SecurityStatus,
    VerificationStatus,
)
from backend_ai.agent.memory_retrieval import MemoryRetrieval, MemoryRetrievalItem, MemoryRetrievalRequest, RetrievalSource
from backend_ai.agent.project_memory import FactCategory, FactConfidence, FactEvidence, FactSource, ProjectMemory


AS_OF = "2026-08-17T00:00:00Z"


def _item(
    content: str = "Django uses PostgreSQL",
    *,
    source: RetrievalSource = RetrievalSource.LONG_TERM_MEMORY,
    memory_id: str = "ltm-test",
    confidence: int | None = 3,
    status: str | None = "active",
    timestamp: str | None = "2026-08-16T00:00:00Z",
    metadata: dict[str, object] | None = None,
    project_id: str | None = None,
) -> MemoryRetrievalItem:
    return MemoryRetrievalItem(source, memory_id, content, 0.5, confidence, status, timestamp, metadata or {"category": "solution"}, "test provenance", project_id)


def _project(root: str = "/project-a") -> ProjectMemory:
    memory = ProjectMemory.for_project(root)
    memory.add_fact(
        category=FactCategory.DATABASE,
        key="database.engine",
        value="PostgreSQL",
        source=FactSource.PROJECT_CONTEXT,
        confidence=FactConfidence.VERIFIED,
        evidence=(FactEvidence(FactSource.PROJECT_CONTEXT, "project-context", "verified database", verified=True),),
    )
    return memory


def _experience(project_id: str = "project-a") -> ExperienceRecords:
    records = ExperienceRecords()
    session = records.start_experience("database verification", project_identity=ExperienceProjectIdentity(project_id, "/project-a"))
    session.start_attempt()
    session.record_verification(ExperienceVerification(1, 1, 0, "PASS", "database passed", AS_OF))
    session.finalize(status="completed", outcome="success", final_summary="database verified")
    return records


def test_quality_confidence_verification_and_provenance_are_explainable() -> None:
    governance = MemoryGovernance()
    trusted = governance.assess(_item(), as_of=AS_OF)
    assert trusted.quality_status is QualityStatus.TRUSTED
    assert trusted.verification_status is VerificationStatus.VERIFIED
    assert trusted.provenance_status.value == "sufficient"
    assert trusted.eligibility_status is EligibilityStatus.ELIGIBLE
    assert "verified and active" in trusted.reasons

    low = governance.assess(_item(confidence=0), as_of=AS_OF)
    assert low.quality_status is QualityStatus.UNCERTAIN
    assert not low.eligible
    assert "insufficient confidence" in low.reasons

    missing = governance.assess(_item(timestamp=None), as_of=AS_OF)
    assert not missing.eligible
    assert "missing provenance" in missing.reasons


def test_freshness_boundaries_are_source_aware_and_deterministic() -> None:
    policy = FreshnessPolicy(long_term_fresh_seconds=10, long_term_aging_seconds=20, long_term_stale_seconds=30, experience_fresh_seconds=10, experience_aging_seconds=20, experience_stale_seconds=30)
    governance = MemoryGovernance(policy=GovernancePolicy(freshness=policy))
    for seconds, expected in ((10, FreshnessStatus.FRESH), (20, FreshnessStatus.AGING), (30, FreshnessStatus.STALE), (31, FreshnessStatus.STALE)):
        timestamp = f"2026-08-16T23:59:{60 - seconds:02d}Z" if seconds < 60 else "2026-08-16T23:59:00Z"
        if seconds == 10:
            timestamp = "2026-08-16T23:59:50Z"
        elif seconds == 20:
            timestamp = "2026-08-16T23:59:40Z"
        elif seconds == 30:
            timestamp = "2026-08-16T23:59:30Z"
        else:
            timestamp = "2026-08-16T23:59:29Z"
        assessment = governance.assess(_item(timestamp=timestamp), as_of="2026-08-17T00:00:00Z")
        assert assessment.freshness_status is expected
    project = governance.assess(_item(source=RetrievalSource.PROJECT_MEMORY, status="ACTIVE", timestamp=None, metadata={"key": "database.engine", "evidence_count": 1}, confidence=3), as_of=AS_OF)
    assert project.freshness_status is FreshnessStatus.NOT_APPLICABLE


def test_invalidated_project_long_term_and_experience_memories_are_preserved_but_excluded() -> None:
    governance = MemoryGovernance()

    project = _project()
    fact_id = project.snapshot().active_facts[0].fact_id
    project_result = governance.invalidate(project, fact_id, reason="verification disproved fact")
    assert project_result.applied and project_result.preserved and project_result.new_status == "INVALID"
    project_retrieval = MemoryRetrieval().retrieve(MemoryRetrievalRequest("PostgreSQL", (RetrievalSource.PROJECT_MEMORY,), project_id=project.project_id, project_memory=project.snapshot()))
    assert not project_retrieval.items
    assert project_retrieval.governance_audit.invalidated_memories == 0  # invalid facts are not normal retrieval candidates

    long_term = LongTermMemory()
    entry = long_term.add(content="Django uses PostgreSQL", category="solution", source="VERIFIED_TASK", confidence="VERIFIED", metadata={"topic": "database"})
    ltm_result = governance.invalidate(long_term, entry.entry_id, reason="obsolete solution")
    assert ltm_result.applied and ltm_result.preserved and ltm_result.new_status == "invalidated"
    assert long_term.get(entry.entry_id, track_access=False).status.value == "invalidated"

    experiences = _experience()
    experience_id = experiences.list()[0].experience_id
    exp_result = governance.invalidate(experiences, experience_id, reason="historical evidence rejected")
    assert exp_result.applied and exp_result.preserved and exp_result.new_status == "completed"
    assert experiences.get(experience_id).metadata["governance_invalidated"] is True
    experience_retrieval = MemoryRetrieval().retrieve(MemoryRetrievalRequest("database verified", (RetrievalSource.EXPERIENCE_RECORDS,), project_id="project-a", experience_records=experiences))
    assert not experience_retrieval.items


def test_duplicate_detection_preserves_first_provenance_and_distinct_similar_text() -> None:
    governance = MemoryGovernance()
    first = _item(memory_id="first")
    duplicate = _item("Django uses PostgreSQL.", memory_id="second")
    similar = _item("Django uses MySQL", memory_id="third")
    evaluation = governance.evaluate_candidates((first, duplicate, similar), as_of=AS_OF)
    assert evaluation.deduplicated_count == 1
    assert evaluation.assessments[1].duplicate_status is DuplicateStatus.DUPLICATE
    assert evaluation.assessments[1].eligibility_status is EligibilityStatus.INELIGIBLE
    assert evaluation.assessments[2].duplicate_status is DuplicateStatus.UNIQUE
    assert evaluation.eligible_items == (first, similar)


def test_structured_conflict_is_visible_and_neither_side_is_eligible() -> None:
    governance = MemoryGovernance()
    first = _item("Project uses PostgreSQL", memory_id="fact-a", source=RetrievalSource.PROJECT_MEMORY, status="ACTIVE", project_id="project-a", metadata={"key": "database.engine", "evidence_count": 1})
    second = _item("Project uses MySQL", memory_id="fact-b", source=RetrievalSource.PROJECT_MEMORY, status="ACTIVE", project_id="project-a", metadata={"key": "database.engine", "evidence_count": 1})
    evaluation = governance.evaluate_candidates((first, second), as_of=AS_OF)
    assert evaluation.audit.conflicts == 2
    assert all(item.conflict_status is ConflictStatus.DETECTED for item in evaluation.assessments)
    assert evaluation.eligible_items == ()


def test_security_violation_and_retention_are_explicit() -> None:
    governance = MemoryGovernance()
    secret = governance.assess(_item("password=hunter2", memory_id="secret"), as_of=AS_OF)
    assert secret.security_status is SecurityStatus.VIOLATION
    assert not secret.eligible
    assert secret.retention_action is RetentionAction.PRESERVE_INVALIDATED

    stale = governance.assess(_item(timestamp="2025-01-01T00:00:00Z"), as_of=AS_OF)
    assert stale.quality_status is QualityStatus.STALE
    assert stale.retention_action is RetentionAction.ARCHIVE_CANDIDATE


def test_audit_is_read_only_and_counts_findings_deterministically() -> None:
    governance = MemoryGovernance()
    items = (_item(memory_id="one"), _item("Django low confidence", memory_id="two", confidence=0), _item(memory_id="three", content="Django uses PostgreSQL."))
    before = tuple(item.to_dict() for item in items)
    audit = governance.audit(items, as_of=AS_OF)
    assert audit.total_memories_inspected == 3
    assert audit.eligible_memories == 1
    assert audit.duplicates == 1
    assert audit.findings == governance.audit(items, as_of=AS_OF).findings
    assert tuple(item.to_dict() for item in items) == before


def test_explicit_archived_low_confidence_is_allowed_only_when_requested() -> None:
    governance = MemoryGovernance()
    archived = _item(confidence=0, status="archived")
    assert not governance.is_eligible(archived, as_of=AS_OF)
    assert governance.is_eligible(archived, as_of=AS_OF, explicit_status="archived")
