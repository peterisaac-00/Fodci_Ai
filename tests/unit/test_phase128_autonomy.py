from __future__ import annotations

import pytest
from pathlib import Path
from backend_ai.agent.advanced_autonomy import (
    AutonomyController,
    AutonomyBudget,
    TaskLifeCycleState,
    LoopDetector,
)
from backend_ai.agent.multi_agent import SubTask, AgentRole


def test_loop_detector() -> None:
    detector = LoopDetector(max_identical=3)
    assert detector.record_and_check("action-1") is False
    assert detector.record_and_check("action-1") is False
    assert detector.record_and_check("action-1") is True


def test_autonomy_controller_execution(tmp_path: Path) -> None:
    controller = AutonomyController(budget=AutonomyBudget(max_iterations=5))
    subtasks = [
        SubTask(id="s1", description="Plan", role=AgentRole.PLANNER),
        SubTask(id="s2", description="Code", role=AgentRole.CODER, dependencies=("s1",)),
    ]
    result = controller.run("task-1", "Build backend", tmp_path, subtasks)
    assert result["lifecycle_state"] == TaskLifeCycleState.COMPLETED.value
    assert result["progress"]["completed_subtasks"] == 2


def test_human_pause_cancel(tmp_path: Path) -> None:
    controller = AutonomyController()
    controller.cancel()
    subtasks = [SubTask(id="s1", description="Plan", role=AgentRole.PLANNER)]
    result = controller.run("task-2", "Cancel test", tmp_path, subtasks)
    assert result["lifecycle_state"] == TaskLifeCycleState.CANCELLED.value
