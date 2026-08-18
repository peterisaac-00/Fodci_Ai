from __future__ import annotations

import pytest
from pathlib import Path
from backend_ai.agent.advanced_memory import (
    AdvancedMemoryRecord,
    AdvancedMemorySystem,
    MemoryScope,
    MemoryType,
    MemoryStatus,
    MemoryProvenance,
    MemoryConfidence,
)


def test_advanced_memory_creation_and_persistence(tmp_path: Path) -> None:
    store_file = tmp_path / "memory.json"
    system = AdvancedMemorySystem(store_file)
    
    rec = AdvancedMemoryRecord(
        id="mem-1",
        memory_type=MemoryType.PROJECT_MEMORY,
        scope=MemoryScope.PROJECT,
        content="Project uses pytest for testing and PostgreSQL for database.",
        project_id="fodci",
        importance=0.9,
        confidence=MemoryConfidence.HIGH,
    )
    system.add(rec)
    
    # Reload from disk
    system2 = AdvancedMemorySystem(store_file)
    retrieved = system2.retrieve("pytest", project_id="fodci")
    assert len(retrieved) == 1
    assert "pytest" in retrieved[0].content


def test_advanced_memory_isolation(tmp_path: Path) -> None:
    store_file = tmp_path / "memory.json"
    system = AdvancedMemorySystem(store_file)
    
    rec1 = AdvancedMemoryRecord(
        id="mem-1",
        memory_type=MemoryType.PROJECT_MEMORY,
        scope=MemoryScope.PROJECT,
        content="Database is PostgreSQL",
        project_id="proj-a",
    )
    rec2 = AdvancedMemoryRecord(
        id="mem-2",
        memory_type=MemoryType.PROJECT_MEMORY,
        scope=MemoryScope.PROJECT,
        content="Database is MySQL",
        project_id="proj-b",
    )
    system.add(rec1)
    system.add(rec2)
    
    retrieved = system.retrieve("Database", project_id="proj-a")
    assert len(retrieved) == 1
    assert "PostgreSQL" in retrieved[0].content
