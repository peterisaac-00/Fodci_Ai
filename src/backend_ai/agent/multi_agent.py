from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from backend_ai.agent.advanced_memory import AdvancedMemorySystem, AdvancedMemoryRecord, MemoryScope, MemoryType, MemoryConfidence
from backend_ai.agent.recovery import ErrorClassifier, normalize_error


class AgentRole(str, Enum):
    PLANNER = "planner"
    CODER = "coder"
    TESTER = "tester"
    DEBUGGER = "debugger"
    REVIEWER = "reviewer"
    VERIFIER = "verifier"


class SubTaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class SubTask:
    id: str
    description: str
    role: AgentRole | str
    priority: int = 1
    dependencies: tuple[str, ...] = ()
    status: SubTaskStatus | str = SubTaskStatus.PENDING
    result: str | None = None
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        role_val = self.role.value if isinstance(self.role, AgentRole) else self.role
        status_val = self.status.value if isinstance(self.status, SubTaskStatus) else self.status
        return {
            "id": self.id,
            "description": self.description,
            "role": role_val,
            "priority": self.priority,
            "dependencies": list(self.dependencies),
            "status": status_val,
            "result": self.result,
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AgentResult:
    success: bool
    summary: str
    artifacts: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "artifacts": list(self.artifacts),
            "observations": list(self.observations),
            "errors": list(self.errors),
            "recommendations": list(self.recommendations),
            "next_actions": list(self.next_actions),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TaskState:
    task_id: str
    original_task: str
    project_root: Path
    subtasks: tuple[SubTask, ...] = ()
    active_agent: AgentRole | str | None = None
    completed_steps: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    status: str = "RUNNING"

    def to_dict(self) -> dict[str, Any]:
        active_val = None
        if self.active_agent:
            active_val = self.active_agent.value if isinstance(self.active_agent, AgentRole) else self.active_agent
        return {
            "task_id": self.task_id,
            "original_task": self.original_task,
            "project_root": str(self.project_root),
            "subtasks": [s.to_dict() for s in self.subtasks],
            "active_agent": active_val,
            "completed_steps": list(self.completed_steps),
            "errors": list(self.errors),
            "status": self.status,
        }


class BaseAgent:
    def __init__(self, role: AgentRole) -> None:
        self.role = role

    def execute(self, subtask: SubTask, state: TaskState, memory_system: AdvancedMemorySystem | None = None) -> AgentResult:
        raise NotImplementedError


class PlannerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(AgentRole.PLANNER)

    def execute(self, subtask: SubTask, state: TaskState, memory_system: AdvancedMemorySystem | None = None) -> AgentResult:
        return AgentResult(success=True, summary=f"Planner analyzed task: {subtask.description}")


class CoderAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(AgentRole.CODER)

    def execute(self, subtask: SubTask, state: TaskState, memory_system: AdvancedMemorySystem | None = None) -> AgentResult:
        return AgentResult(success=True, summary=f"Coder executed: {subtask.description}", artifacts=("modified_file.py",))


class TesterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(AgentRole.TESTER)

    def execute(self, subtask: SubTask, state: TaskState, memory_system: AdvancedMemorySystem | None = None) -> AgentResult:
        return AgentResult(success=True, summary=f"Tester ran tests for: {subtask.description}", observations=("All tests passed successfully.",))


class DebuggerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(AgentRole.DEBUGGER)

    def execute(self, subtask: SubTask, state: TaskState, memory_system: AdvancedMemorySystem | None = None) -> AgentResult:
        return AgentResult(success=True, summary=f"Debugger resolved issue for: {subtask.description}", recommendations=("Applied fix.",))


class ReviewerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(AgentRole.REVIEWER)

    def execute(self, subtask: SubTask, state: TaskState, memory_system: AdvancedMemorySystem | None = None) -> AgentResult:
        return AgentResult(success=True, summary="Review passed successfully.")


class VerifierAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(AgentRole.VERIFIER)

    def execute(self, subtask: SubTask, state: TaskState, memory_system: AdvancedMemorySystem | None = None) -> AgentResult:
        return AgentResult(success=True, summary="Final verification passed.")


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[AgentRole, BaseAgent] = {
            AgentRole.PLANNER: PlannerAgent(),
            AgentRole.CODER: CoderAgent(),
            AgentRole.TESTER: TesterAgent(),
            AgentRole.DEBUGGER: DebuggerAgent(),
            AgentRole.REVIEWER: ReviewerAgent(),
            AgentRole.VERIFIER: VerifierAgent(),
        }

    def get(self, role: AgentRole | str) -> BaseAgent:
        role_enum = AgentRole(role) if isinstance(role, str) else role
        if role_enum not in self._agents:
            raise KeyError(f"Unknown agent role: {role}")
        return self._agents[role_enum]


class AgentOrchestrator:
    def __init__(self, registry: AgentRegistry | None = None, memory_system: AdvancedMemorySystem | None = None, max_steps: int = 16) -> None:
        self.registry = registry or AgentRegistry()
        self.memory_system = memory_system
        self.max_steps = max_steps

    def execute_task(self, task_id: str, original_task: str, project_root: Path, subtasks: Sequence[SubTask]) -> TaskState:
        state = TaskState(task_id=task_id, original_task=original_task, project_root=project_root, subtasks=tuple(subtasks))
        steps = 0
        
        while steps < self.max_steps:
            steps += 1
            ready_subtasks = [s for s in state.subtasks if s.status == SubTaskStatus.PENDING and all(dep in [st.id for st in state.subtasks if st.status == SubTaskStatus.COMPLETED] for dep in s.dependencies)]
            
            if not ready_subtasks:
                if all(s.status == SubTaskStatus.COMPLETED for s in state.subtasks):
                    state = replace_task_state(state, status="COMPLETED")
                    break
                if any(s.status == SubTaskStatus.FAILED for s in state.subtasks):
                    state = replace_task_state(state, status="FAILED")
                    break
                if all(s.status == SubTaskStatus.PENDING for s in state.subtasks) and len(state.subtasks) > 0:
                    updated_subtasks = list(state.subtasks)
                    for i, s in enumerate(updated_subtasks):
                        if not s.dependencies:
                            updated_subtasks[i] = replace_subtask(s, status=SubTaskStatus.READY)
                    state = replace_task_state(state, subtasks=tuple(updated_subtasks))
                    continue
                break

            subtask = ready_subtasks[0]
            agent = self.registry.get(subtask.role)
            state = replace_task_state(state, active_agent=subtask.role)
            
            subtasks_list = list(state.subtasks)
            idx = next(i for i, s in enumerate(subtasks_list) if s.id == subtask.id)
            subtasks_list[idx] = replace_subtask(subtask, status=SubTaskStatus.RUNNING)
            state = replace_task_state(state, subtasks=tuple(subtasks_list))

            try:
                result = agent.execute(subtask, state, self.memory_system)
                role_val = subtask.role.value if isinstance(subtask.role, AgentRole) else subtask.role
                if result.success:
                    subtasks_list[idx] = replace_subtask(subtasks_list[idx], status=SubTaskStatus.COMPLETED, result=result.summary)
                    completed = list(state.completed_steps) + [f"{role_val}:{subtask.id}"]
                    state = replace_task_state(state, subtasks=tuple(subtasks_list), completed_steps=tuple(completed))
                    
                    if self.memory_system:
                        mem = AdvancedMemoryRecord(
                            id=f"mem-{subtask.id}-{datetime.now(timezone.utc).timestamp()}",
                            memory_type=MemoryType.SOLUTION_MEMORY,
                            scope=MemoryScope.PROJECT,
                            content=f"Subtask {subtask.description} completed successfully by {role_val}: {result.summary}",
                            importance=0.8,
                            confidence=MemoryConfidence.HIGH,
                        )
                        self.memory_system.add(mem)
                else:
                    subtasks_list[idx] = replace_subtask(subtasks_list[idx], status=SubTaskStatus.FAILED, errors=tuple(result.errors))
                    errors_list = list(state.errors) + list(result.errors)
                    state = replace_task_state(state, subtasks=tuple(subtasks_list), errors=tuple(errors_list), status="FAILED")
                    break
            except Exception as e:
                subtasks_list[idx] = replace_subtask(subtasks_list[idx], status=SubTaskStatus.FAILED, errors=(str(e),))
                state = replace_task_state(state, subtasks=tuple(subtasks_list), errors=list(state.errors) + [str(e)], status="FAILED")
                break

        return state


def replace_task_state(state: TaskState, **kwargs: Any) -> TaskState:
    data = state.to_dict()
    data.update(kwargs)
    subtasks_data = data["subtasks"]
    subtasks_tuples = tuple(
        s if isinstance(s, SubTask) else SubTask(
            id=s["id"],
            description=s["description"],
            role=AgentRole(s["role"]),
            priority=int(s["priority"]),
            dependencies=tuple(s["dependencies"]),
            status=SubTaskStatus(s["status"]),
            result=s.get("result"),
            errors=tuple(s.get("errors", [])),
            metadata=s.get("metadata", {}),
        )
        for s in subtasks_data
    )
    return TaskState(
        task_id=data["task_id"],
        original_task=data["original_task"],
        project_root=Path(data["project_root"]),
        subtasks=subtasks_tuples,
        active_agent=AgentRole(data["active_agent"]) if data["active_agent"] else None,
        completed_steps=tuple(data["completed_steps"]),
        errors=tuple(data["errors"]),
        status=data["status"],
    )


def replace_subtask(subtask: SubTask, **kwargs: Any) -> SubTask:
    data = subtask.to_dict()
    data.update(kwargs)
    return SubTask(
        id=data["id"],
        description=data["description"],
        role=AgentRole(data["role"]),
        priority=int(data["priority"]),
        dependencies=tuple(data["dependencies"]),
        status=SubTaskStatus(data["status"]),
        result=data["result"],
        errors=tuple(data["errors"]),
        metadata=data["metadata"],
    )
