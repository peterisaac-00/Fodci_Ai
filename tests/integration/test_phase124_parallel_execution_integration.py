from __future__ import annotations

from pathlib import Path
from backend_ai.agent import (
    AutonomousLoopConfig,
    AutonomousLoopRequest,
    AutonomousToolLoop,
    LoopStatus,
    ToolRegistry,
)
from backend_ai.tools.base import ToolMetadata, ToolError
from backend_ai.core.contracts import Tool
from backend_ai.agent.planner import ExecutionPlan, PlanStep, PlanStepStatus, PlannerTaskType, PlannerConfidence, PlanCompleteness
from backend_ai.agent.planner import PlanRiskLevel
from backend_ai.agent.tool_selection import ToolSelectionDecision
from backend_ai.tools.project_context import project_context


class SlowRecordingTool(Tool):
    def __init__(self, name: str, return_value: Any) -> None:
        self._name = name
        self._return_value = return_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Test tool {self._name}"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name=self._name, description=self.description, input_schema={})

    def run(self, arguments: Mapping[str, Any]) -> Any:
        return self._return_value


class FailingTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Failing tool {self._name}"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name=self._name, description=self.description, input_schema={})

    def run(self, arguments: Mapping[str, Any]) -> Any:
        raise ToolError("Simulated tool failure")


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class Engine:
    tokenizer = FakeTokenizer()

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str):
        from types import SimpleNamespace
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else 'ACTION: FINAL\nARGS: {"message": "done"}'
        return SimpleNamespace(generated_text=text)


class StaticPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan_value = plan

    def plan(self, request: Any) -> Any:
        from types import SimpleNamespace
        return SimpleNamespace(plan=self.plan_value, warnings=(), errors=(), status="CREATED")


from backend_ai.agent.tool_selection import ToolSelector


def _plan() -> ExecutionPlan:
    step = PlanStep(
        step_id="s1",
        title="Inspect discovered source file",
        objective="Read file content in parallel",
        rationale="parallel test",
        risk_level=PlanRiskLevel.LOW,
        dependencies=(),
        expected_result="files observed",
        status=PlanStepStatus.IN_PROGRESS,
    )
    return ExecutionPlan(
        "inspect in parallel",
        "inspect in parallel",
        "Inspect files",
        PlannerTaskType.INVESTIGATION,
        (step,),
        (),
        (),
        (),
        (),
        (),
        PlannerConfidence.HIGH,
        (),
        PlanCompleteness.COMPLETE,
    )


def test_parallel_tool_batch_execution(tmp_path: Path) -> None:
    # Create test files
    (tmp_path / "a.py").write_text("print('a')")
    (tmp_path / "b.py").write_text("print('b')")
    registry = ToolRegistry.default()
    
    # Model returns batch action with multiple calls
    action_text = 'ACTION: TOOL\nARGS: {"calls": [{"tool": "read_file", "arguments": {"path": "a.py"}}, {"tool": "read_file", "arguments": {"path": "b.py"}}]}'
    engine = Engine([action_text, 'ACTION: FINAL\nARGS: {"message": "inspected"}'])
    
    loop = AutonomousToolLoop(
        engine,
        registry=registry,
        planner=StaticPlanner(_plan()),
        selector=ToolSelector(),
        config=AutonomousLoopConfig(parallel_execution_enabled=True, max_parallel_tools=2),
    )
    
    result = loop.run(AutonomousLoopRequest("inspect files in parallel", tmp_path, project_context(tmp_path)))
    assert result.status in {LoopStatus.COMPLETED, LoopStatus.CONTINUE}
    assert len(result.tool_calls) == 2
    assert len(result.tool_results) == 2
    assert result.parallel_metrics is not None
    assert result.parallel_metrics.parallel_tool_calls == 2
    assert result.parallel_metrics.parallel_batches >= 1


def test_disabled_parallel_execution_forces_sequential(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('a')")
    (tmp_path / "b.py").write_text("print('b')")
    registry = ToolRegistry.default()
    
    action_text = 'ACTION: TOOL\nARGS: {"calls": [{"tool": "read_file", "arguments": {"path": "a.py"}}, {"tool": "read_file", "arguments": {"path": "b.py"}}]}'
    engine = Engine([action_text, 'ACTION: FINAL\nARGS: {"message": "inspected"}'])
    
    loop = AutonomousToolLoop(
        engine,
        registry=registry,
        planner=StaticPlanner(_plan()),
        selector=ToolSelector(),
        config=AutonomousLoopConfig(parallel_execution_enabled=False),
    )
    
    result = loop.run(AutonomousLoopRequest("inspect files sequentially", tmp_path, project_context(tmp_path)))
    assert result.status in {LoopStatus.COMPLETED, LoopStatus.CONTINUE}
    assert len(result.tool_calls) == 2
    assert result.parallel_metrics is not None
    assert result.parallel_metrics.parallel_tool_calls == 0
    assert result.parallel_metrics.sequential_tool_calls == 2
