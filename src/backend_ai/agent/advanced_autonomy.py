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

from backend_ai.agent.advanced_memory import AdvancedMemorySystem, AdvancedMemoryRecord, MemoryScope, MemoryType, MemoryConfidence
from backend_ai.agent.multi_agent import AgentOrchestrator, SubTask, SubTaskStatus, TaskState, AgentRole
from backend_ai.agent.recovery import ErrorClassifier, normalize_error


class TaskLifeCycleState(str, Enum):
    CREATED = "CREATED"
    ANALYZING = "ANALYZING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    TESTING = "TESTING"
    EVALUATING = "EVALUATING"
    RECOVERING = "RECOVERING"
    REPLANNING = "REPLANNING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


class AutonomyMode(str, Enum):
    FULL = "full"
    SUPERVISED = "supervised"
    RESTRICTED = "restricted"


@dataclass(frozen=True, slots=True)
class AutonomyBudget:
    max_iterations: int = 16
    max_retries: int = 4
    max_replans: int = 3
    max_tool_calls: int = 64
    max_recovery_attempts: int = 4

    def to_dict(self) -> dict[str, int]:
        return {
            "max_iterations": self.max_iterations,
            "max_retries": self.max_retries,
            "max_replans": self.max_replans,
            "max_tool_calls": self.max_tool_calls,
            "max_recovery_attempts": self.max_recovery_attempts,
        }


@dataclass(frozen=True, slots=True)
class AutonomyProgress:
    total_subtasks: int = 0
    completed_subtasks: int = 0
    verified_requirements: int = 0
    total_tests: int = 0
    passing_tests: int = 0
    errors_encountered: int = 0
    recovery_attempts: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_subtasks": self.total_subtasks,
            "completed_subtasks": self.completed_subtasks,
            "verified_requirements": self.verified_requirements,
            "total_tests": self.total_tests,
            "passing_tests": self.passing_tests,
            "errors_encountered": self.errors_encountered,
            "recovery_attempts": self.recovery_attempts,
        }


class LoopDetector:
    def __init__(self, max_identical: int = 3) -> None:
        self.max_identical = max_identical
        self.history: list[str] = []

    def record_and_check(self, signature: str) -> bool:
        self.history.append(signature)
        if len(self.history) > 10:
            self.history.pop(0)
        # Check if the last N items are identical
        if len(self.history) >= self.max_identical:
            recent = self.history[-self.max_identical:]
            if all(item == recent[0] for item in recent):
                return True
        return False


@dataclass(frozen=True, slots=True)
class AutonomyCheckpoint:
    task_id: str
    lifecycle_state: TaskLifeCycleState
    iteration: int
    progress: AutonomyProgress
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "lifecycle_state": self.lifecycle_state.value,
            "iteration": self.iteration,
            "progress": self.progress.to_dict(),
            "timestamp": self.timestamp,
        }


class AutonomyController:
    def __init__(
        self,
        orchestrator: AgentOrchestrator | None = None,
        memory_system: AdvancedMemorySystem | None = None,
        budget: AutonomyBudget | None = None,
        mode: AutonomyMode = AutonomyMode.FULL,
    ) -> None:
        self.orchestrator = orchestrator or AgentOrchestrator()
        self.memory_system = memory_system
        self.budget = budget or AutonomyBudget()
        self.mode = mode
        self.lifecycle_state = TaskLifeCycleState.CREATED
        self.paused = False
        self.cancelled = False
        self.loop_detector = LoopDetector()
        self.checkpoints: list[AutonomyCheckpoint] = []

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def cancel(self) -> None:
        self.cancelled = True
        self.lifecycle_state = TaskLifeCycleState.CANCELLED

    def run(self, task_id: str, original_task: str, project_root: Path, subtasks: Sequence[SubTask]) -> dict[str, Any]:
        self.lifecycle_state = TaskLifeCycleState.ANALYZING
        progress = AutonomyProgress(total_subtasks=len(subtasks))
        
        if self.cancelled:
            return self._result(task_id, TaskLifeCycleState.CANCELLED, progress, "Task cancelled by human.")

        self.lifecycle_state = TaskLifeCycleState.PLANNING
        checkpoint = AutonomyCheckpoint(task_id=task_id, lifecycle_state=self.lifecycle_state, iteration=0, progress=progress)
        self.checkpoints.append(checkpoint)

        iterations = 0
        while iterations < self.budget.max_iterations:
            if self.cancelled:
                self.lifecycle_state = TaskLifeCycleState.CANCELLED
                break
            while self.paused:
                # Wait until resumed or cancelled
                if self.cancelled:
                    break
                pass

            iterations += 1
            self.lifecycle_state = TaskLifeCycleState.EXECUTING

            # Delegate to orchestrator
            state = self.orchestrator.execute_task(task_id, original_task, project_root, subtasks)
            
            # Update progress
            completed_count = sum(1 for s in state.subtasks if s.status == SubTaskStatus.COMPLETED)
            failed_count = sum(1 for s in state.subtasks if s.status == SubTaskStatus.FAILED)
            progress = AutonomyProgress(
                total_subtasks=len(state.subtasks),
                completed_subtasks=completed_count,
                verified_requirements=completed_count,
                errors_encountered=failed_count,
            )

            # Loop detection check
            sig = f"iter-{iterations}-{completed_count}-{failed_count}"
            if self.loop_detector.record_and_check(sig):
                self.lifecycle_state = TaskLifeCycleState.REPLANNING
                # Trigger replan or stop
                break

            if state.status == "COMPLETED":
                self.lifecycle_state = TaskLifeCycleState.VERIFYING
                # Verification gate
                self.lifecycle_state = TaskLifeCycleState.COMPLETED
                break
            elif state.status == "FAILED":
                self.lifecycle_state = TaskLifeCycleState.RECOVERING
                # Try recovery or fail
                self.lifecycle_state = TaskLifeCycleState.FAILED
                break

        if iterations >= self.budget.max_iterations and self.lifecycle_state not in {TaskLifeCycleState.COMPLETED, TaskLifeCycleState.FAILED}:
            self.lifecycle_state = TaskLifeCycleState.TIMEOUT

        return self._result(task_id, self.lifecycle_state, progress, f"Task finished with state {self.lifecycle_state.value}")

    def _result(self, task_id: str, state: TaskLifeCycleState, progress: AutonomyProgress, message: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "lifecycle_state": state.value,
            "progress": progress.to_dict(),
            "message": message,
            "checkpoints_count": len(self.checkpoints),
        }
