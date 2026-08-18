from __future__ import annotations

import pytest
from pathlib import Path
from backend_ai.agent.multi_agent import (
    AgentRegistry,
    AgentRole,
    AgentOrchestrator,
    SubTask,
    SubTaskStatus,
    TaskState,
)
from backend_ai.agent.advanced_memory import AdvancedMemorySystem, MemoryScope


def test_agent_registry_retrieval() -> None:
    registry = AgentRegistry()
    assert registry.get(AgentRole.PLANNER) is not None
    assert registry.get(AgentRole.CODER) is not None
    assert registry.get(AgentRole.TESTER) is not None
    assert registry.get(AgentRole.DEBUGGER) is not None
    assert registry.get(AgentRole.REVIEWER) is not None
    assert registry.get(AgentRole.VERIFIER) is not None


def test_orchestrator_successful_workflow(tmp_path: Path) -> None:
    memory_system = AdvancedMemorySystem(tmp_path / "mem.json")
    orchestrator = AgentOrchestrator(memory_system=memory_system)
    
    subtasks = [
        SubTask(id="s1", description="Plan architecture", role=AgentRole.PLANNER),
        SubTask(id="s2", description="Write code", role=AgentRole.CODER, dependencies=("s1",)),
        SubTask(id="s3", description="Run tests", role=AgentRole.TESTER, dependencies=("s2",)),
        SubTask(id="s4", description="Verify final state", role=AgentRole.VERIFIER, dependencies=("s3",)),
    ]
    
    state = orchestrator.execute_task("task-1", "Implement feature", tmp_path, subtasks)
    assert state.status == "COMPLETED"
    assert len(state.completed_steps) == 4
    
    # Verify memory was written
    memories = memory_system.retrieve("Plan architecture", scope=MemoryScope.PROJECT)
    assert len(memories) >= 1
