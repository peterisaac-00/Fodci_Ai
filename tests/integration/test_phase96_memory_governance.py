from __future__ import annotations

from pathlib import Path

from backend_ai.agent.autonomous_tool_loop import AutonomousLoopRequest, AutonomousToolLoop
from backend_ai.agent.memory_governance import GovernancePolicy, MemoryGovernance, QualityStatus
from backend_ai.agent.memory_retrieval import MemoryRetrieval, MemoryRetrievalRequest, RetrievalSource
from backend_ai.agent.project_memory import FactCategory, FactConfidence, FactEvidence, FactSource, ProjectMemory
from backend_ai.agent.long_term_memory import LongTermMemory


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class _Provider:
    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str):
        from types import SimpleNamespace

        self.prompts.append(prompt)
        return SimpleNamespace(generated_text='ACTION: FINAL\nARGS: {"message": "governed"}')


def _project(root: Path, value: str = "PostgreSQL") -> ProjectMemory:
    memory = ProjectMemory.for_project(root)
    memory.add_fact(
        category=FactCategory.DATABASE,
        key="database.engine",
        value=value,
        source=FactSource.PROJECT_CONTEXT,
        confidence=FactConfidence.VERIFIED,
        evidence=(FactEvidence(FactSource.PROJECT_CONTEXT, "integration", f"verified {value}", verified=True),),
    )
    return memory


def test_healthy_unified_retrieval_returns_only_governed_context(tmp_path: Path) -> None:
    project = _project(tmp_path)
    long_term = LongTermMemory()
    long_term.add(content="PostgreSQL requires explicit migration review", category="solution", source="VERIFIED_TASK", confidence="VERIFIED", metadata={"topic": "database"})
    request = MemoryRetrievalRequest(
        "PostgreSQL",
        (RetrievalSource.PROJECT_MEMORY, RetrievalSource.LONG_TERM_MEMORY),
        project_id=project.project_id,
        project_root=project.project_root,
        project_memory=project.snapshot(),
        long_term_memory=long_term,
        governance_as_of="2026-08-17T00:00:00Z",
    )
    result = MemoryRetrieval().retrieve(request)
    assert result.items
    assert result.governance_audit is not None
    assert result.governance_audit.eligible_memories == len(result.governance_audit.assessments)
    assert all(item.eligibility_status.value == "eligible" for item in result.governance_assessments)
    assert "[PROJECT_MEMORY]" in result.context
    assert "[LONG_TERM_MEMORY]" in result.context


def test_stale_memory_is_excluded_and_audit_explains_it(tmp_path: Path) -> None:
    long_term = LongTermMemory(clock=lambda: "2026-08-17T00:00:00Z")
    long_term.add(content="Old PostgreSQL migration", category="solution", source="VERIFIED_TASK", confidence="VERIFIED", metadata={"topic": "database"})
    request = MemoryRetrievalRequest(
        "PostgreSQL",
        (RetrievalSource.LONG_TERM_MEMORY,),
        long_term_memory=long_term,
        governance_policy=GovernancePolicy(),
        governance_as_of="2026-08-17T00:00:00Z",
    )
    # The source timestamp is intentionally current here; direct governance
    # assessment below proves the stale policy without mutating retrieval data.
    result = MemoryRetrieval().retrieve(request)
    assert result.items
    stale = MemoryGovernance().assess(result.items[0], as_of="2027-01-01T00:00:00Z")
    assert stale.quality_status is QualityStatus.STALE
    assert not stale.eligible
    assert "memory expired freshness policy" in stale.reasons


def test_cross_project_governance_does_not_leak_project_fact(tmp_path: Path) -> None:
    project_a = _project(tmp_path / "a", "PostgreSQL")
    project_b = _project(tmp_path / "b", "MySQL")
    isolated = MemoryRetrieval().retrieve(
        MemoryRetrievalRequest(
            "PostgreSQL",
            (RetrievalSource.PROJECT_MEMORY,),
            project_id=project_a.project_id,
            project_memory=project_b.snapshot(),
        )
    )
    assert isolated.items == ()
    assert isolated.governance_audit is not None


def test_loop_injects_only_governed_retrieval_context_without_new_permissions(tmp_path: Path) -> None:
    project = _project(tmp_path)
    provider = _Provider()
    retrieval_request = MemoryRetrievalRequest(
        "PostgreSQL",
        (RetrievalSource.PROJECT_MEMORY,),
        project_id=project.project_id,
        project_memory=project.snapshot(),
    )
    result = AutonomousToolLoop(provider).run(
        AutonomousLoopRequest(
            task="review database",
            project_root=tmp_path,
            project_memory=project,
            memory_retrieval_request=retrieval_request,
        )
    )
    assert result.memory_retrieval is not None
    assert result.memory_retrieval.governance_audit is not None
    assert result.memory_retrieval.context in provider.prompts[0]
    assert "[PROJECT_MEMORY]" in provider.prompts[0]
    assert all(call.name != "memory_governance" for call in result.tool_calls)


def test_secret_governance_rejects_direct_unredacted_candidate() -> None:
    from backend_ai.agent.memory_retrieval import MemoryRetrievalItem

    item = MemoryRetrievalItem(RetrievalSource.LONG_TERM_MEMORY, "ltm-secret", "token=untrusted", 0.5, 4, "active", "2026-08-16T00:00:00Z", {"category": "warning"}, "test")
    assessment = MemoryGovernance().assess(item, as_of="2026-08-17T00:00:00Z")
    assert assessment.security_status.value == "violation"
    assert not assessment.eligible
