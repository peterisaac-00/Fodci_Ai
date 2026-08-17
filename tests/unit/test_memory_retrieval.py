from __future__ import annotations

from backend_ai.agent.experience_records import ExperienceProjectIdentity, ExperienceRecords, ExperienceVerification
from backend_ai.agent.long_term_memory import LongTermMemory
from backend_ai.agent.memory_retrieval import (
    MemoryRetrieval,
    MemoryRetrievalError,
    MemoryRetrievalRequest,
    RetrievalSource,
)
from backend_ai.agent.project_memory import FactCategory, FactConfidence, FactEvidence, FactSource, ProjectMemory
from backend_ai.agent.short_term_memory import MemoryImportance, ShortTermMemory


def _project(root: str, framework: str = "Django") -> ProjectMemory:
    memory = ProjectMemory.for_project(root)
    memory.add_fact(
        category=FactCategory.FRAMEWORK,
        key="framework.name",
        value=framework,
        source=FactSource.PROJECT_CONTEXT,
        confidence=FactConfidence.VERIFIED,
        evidence=(FactEvidence(FactSource.PROJECT_CONTEXT, "project-context", f"verified framework {framework}", verified=True),),
    )
    return memory


def _long_term(content: str = "Django authentication requires explicit token validation") -> LongTermMemory:
    memory = LongTermMemory()
    memory.add(content=content, category="solution", source="VERIFIED_TASK", confidence="VERIFIED", metadata={"topic": "authentication"})
    return memory


def _experience(task: str = "Django authentication task", project_id: str = "project-a", project_root: str = "/project-a") -> ExperienceRecords:
    records = ExperienceRecords()
    session = records.start_experience(task, project_identity=ExperienceProjectIdentity(project_id, project_root))
    session.start_attempt()
    session.record_observation("Django authentication tests passed", source="run_tests")
    session.record_verification(ExperienceVerification(1, 1, 0, "PASS", "authentication passed", "2026-08-17T00:00:01Z"))
    session.finalize(status="completed", outcome="success", final_solution="Validate JWT explicitly", final_summary="Django authentication passed")
    return records


def test_explicit_source_selection_does_not_query_unrequested_sources() -> None:
    result = MemoryRetrieval().retrieve(MemoryRetrievalRequest("Django", (RetrievalSource.PROJECT_MEMORY,), project_memory=_project("/project-a"), project_id="project-ignored"))
    assert result.queried_sources == (RetrievalSource.PROJECT_MEMORY,)
    assert [diagnostic.source for diagnostic in result.diagnostics] == [RetrievalSource.PROJECT_MEMORY]
    assert result.items == ()  # wrong project identity is diagnosed, not leaked
    assert result.diagnostics[0].status == "FAILED"


def test_unified_sources_have_provenance_and_project_isolation() -> None:
    project = _project("/project-a")
    project_id = project.project_id
    short = ShortTermMemory.for_task("Django authentication task", "/project-a", session_id="session-a")
    short.record_observation("Django authentication observation", source="test", importance=MemoryImportance.IMPORTANT)
    request = MemoryRetrievalRequest(
        "Django authentication",
        (RetrievalSource.SHORT_TERM_MEMORY, RetrievalSource.PROJECT_MEMORY, RetrievalSource.LONG_TERM_MEMORY, RetrievalSource.EXPERIENCE_RECORDS),
        project_id=project_id,
        short_term_memory=short.snapshot(),
        project_memory=project.snapshot(),
        long_term_memory=_long_term(),
        experience_records=_experience(project_id=project_id, project_root=project.project_root),
    )
    result = MemoryRetrieval().retrieve(request)
    assert {item.source for item in result.items} == {RetrievalSource.SHORT_TERM_MEMORY, RetrievalSource.PROJECT_MEMORY, RetrievalSource.LONG_TERM_MEMORY, RetrievalSource.EXPERIENCE_RECORDS}
    assert all(item.memory_id and item.retrieval_reason for item in result.items)
    assert all(item.project_id in {None, project_id} for item in result.items)
    assert "[PROJECT_MEMORY]" in result.context
    assert "[LONG_TERM_MEMORY]" in result.context
    assert "[EXPERIENCE_RECORDS]" in result.context
    project_b = _project("/project-b", framework="Flask")
    isolated = MemoryRetrieval().retrieve(MemoryRetrievalRequest("Django", (RetrievalSource.PROJECT_MEMORY,), project_id=project_id, project_memory=project_b.snapshot()))
    assert not isolated.items
    assert isolated.diagnostics[0].status == "FAILED"


def test_deterministic_ranking_and_stable_tie_breaking() -> None:
    retrieval = MemoryRetrieval()
    request = MemoryRetrievalRequest("Django authentication", (RetrievalSource.LONG_TERM_MEMORY,), long_term_memory=_long_term())
    first = retrieval.retrieve(request)
    second = retrieval.retrieve(request)
    assert [item.memory_id for item in first.items] == [item.memory_id for item in second.items]
    assert first.items[0].relevance_score == second.items[0].relevance_score


def test_deduplication_removes_only_exact_normalized_content() -> None:
    short = ShortTermMemory.for_task("Django authentication", "/project-a")
    short.record_observation("Django authentication requires explicit token validation", source="test")
    ltm = _long_term("Django authentication requires explicit token validation")
    result = MemoryRetrieval().retrieve(MemoryRetrievalRequest("Django authentication", (RetrievalSource.SHORT_TERM_MEMORY, RetrievalSource.LONG_TERM_MEMORY), short_term_memory=short.snapshot(), long_term_memory=ltm))
    assert len(result.items) == 2
    assert result.deduplicated_count == 1


def test_confidence_category_and_status_filters_are_respected() -> None:
    ltm = LongTermMemory()
    ltm.add(content="Django verified solution", category="solution", source="VERIFIED_TASK", confidence="VERIFIED", metadata={})
    ltm.add(content="Django warning", category="warning", source="USER_PROVIDED", confidence="UNKNOWN", metadata={})
    result = MemoryRetrieval().retrieve(MemoryRetrievalRequest("Django", (RetrievalSource.LONG_TERM_MEMORY,), long_term_memory=ltm, category="solution", confidence_threshold=3))
    assert len(result.items) == 1
    assert result.items[0].metadata["category"] == "solution"
    archived = ltm.list(category="warning")[0]
    ltm.update(archived.entry_id, status="archived")
    archived_result = MemoryRetrieval().retrieve(MemoryRetrievalRequest("Django", (RetrievalSource.LONG_TERM_MEMORY,), long_term_memory=ltm, status="archived"))
    assert archived_result.items[0].status == "archived"


def test_context_budget_selects_ranked_results_without_semantic_truncation() -> None:
    ltm = LongTermMemory()
    ltm.add(content="Django authentication short", category="solution", source="VERIFIED_TASK", confidence="VERIFIED", metadata={})
    ltm.add(content="Django authentication another short", category="solution", source="VERIFIED_TASK", confidence="VERIFIED", metadata={})
    result = MemoryRetrieval().retrieve(MemoryRetrievalRequest("Django authentication", (RetrievalSource.LONG_TERM_MEMORY,), long_term_memory=ltm, max_results=2, max_results_per_source=2, max_total_characters=1000))
    assert len(result.context) <= 1000
    assert all("[truncated" not in item.content for item in result.items)


def test_source_failure_is_diagnostic_and_partial_results_survive() -> None:
    result = MemoryRetrieval().retrieve(MemoryRetrievalRequest("Django", (RetrievalSource.PROJECT_MEMORY, RetrievalSource.LONG_TERM_MEMORY), project_memory=None, long_term_memory=_long_term()))
    assert any(item.source is RetrievalSource.PROJECT_MEMORY and item.status == "FAILED" for item in result.diagnostics)
    assert any(item.source is RetrievalSource.LONG_TERM_MEMORY and item.status == "AVAILABLE" for item in result.diagnostics)
    assert result.items


def test_secret_content_does_not_reach_final_context() -> None:
    ltm = LongTermMemory()
    ltm.add(content="Use API_KEY=super-secret only as a redacted warning", category="warning", source="USER_PROVIDED", confidence="USER_CONFIRMED", metadata={"password": "hidden"})
    result = MemoryRetrieval().retrieve(MemoryRetrievalRequest("redacted warning", (RetrievalSource.LONG_TERM_MEMORY,), long_term_memory=ltm))
    assert result.items
    assert "super-secret" not in result.context
    assert "hidden" not in result.context
    assert "[REDACTED]" in result.context


def test_invalid_requests_fail_deterministically() -> None:
    try:
        MemoryRetrievalRequest("", (RetrievalSource.LONG_TERM_MEMORY,))
        raise AssertionError("empty query was accepted")
    except MemoryRetrievalError:
        pass
    try:
        MemoryRetrievalRequest("Django", ())
        raise AssertionError("empty source set was accepted")
    except MemoryRetrievalError:
        pass
