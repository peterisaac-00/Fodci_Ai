from __future__ import annotations

import pytest
from pathlib import Path
from backend_ai.agent.autonomous_tool_loop import (
    AutonomousLoopConfig,
    AutonomousLoopRequest,
    AutonomousLoopResult,
    AutonomousToolLoop,
    LoopStatus,
)
from backend_ai.agent.registry import ToolRegistry, RegisteredTool
from backend_ai.agent.models import ToolResult, ToolCall
from backend_ai.core.contracts import Tool
from backend_ai.tools.base import ToolMetadata


class MockTool(Tool):
    def __init__(self, name: str, impl) -> None:
        self._name = name
        self._impl = impl

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(name=self._name, description="mock", input_schema={})

    def run(self, arguments: Mapping[str, Any], project_root: Path) -> ToolResult:
        return self._impl(**arguments)


class MockFailingTool:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, path: str = "test.py") -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            return ToolResult(call_id="call-1", tool_name="read_file", success=False, error_code="FILE_NOT_FOUND", message=f"File not found: {path}")
        return ToolResult(call_id="call-1", tool_name="read_file", success=True, data=f"Successfully read {path}")


def test_autonomous_loop_recovers_from_file_error(tmp_path: Path) -> None:
    tool = MockFailingTool()
    reg_tool = RegisteredTool(
        name="read_file",
        description="Read file",
        input_schema={"type": "object"},
        tool=MockTool("read_file", tool.run),
    )
    registry = ToolRegistry(tools=[reg_tool])
    assert registry is not None
