from __future__ import annotations

import pytest
from pathlib import Path
from backend_ai.agent.advanced_memory import (
    AdvancedMemoryRecord,
    AdvancedMemorySystem,
    MemoryScope,
    MemoryType,
    MemoryConfidence,
)


def test_end_to_end_memory_reuse_scenario(tmp_path: Path) -> None:
    store_file = tmp_path / "memory.json"
    memory_system = AdvancedMemorySystem(store_file)
    
    # Task 1: Store successful recovery knowledge
    error_knowledge = AdvancedMemoryRecord(
        id="err-jwt-1",
        memory_type=MemoryType.ERROR_MEMORY,
        scope=MemoryScope.PROJECT,
        content="Error DEPENDENCY_ERROR: ModuleNotFoundError for jwt. Resolution: install PyJWT and add to requirements.",
        project_id="fodci-app",
        importance=0.95,
        confidence=MemoryConfidence.HIGH,
    )
    memory_system.add(error_knowledge)
    
    # Task 2: Similar error occurs, retrieve historical memory
    retrieved = memory_system.retrieve("DEPENDENCY_ERROR jwt", project_id="fodci-app")
    assert len(retrieved) == 1
    assert "PyJWT" in retrieved[0].content
