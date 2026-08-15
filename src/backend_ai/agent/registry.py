"""Deterministic discovery and dispatch for existing read-only tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from backend_ai.core.contracts import Tool
from backend_ai.tools import (
    ListFilesTool,
    ProjectContextTool,
    ProjectStructureTool,
    ReadFileTool,
    SearchCodeTool,
    ToolMetadata,
    WriteFileTool,
    EditFileTool,
    DeleteFileTool,
    GitDiffTool,
    GitStatusTool,
    RunCommandTool,
    PolicyRunCommandTool,
    CommandPolicy,
    RunApplicationTool,
    RunTestsTool,
)


class ToolRegistryError(RuntimeError):
    """Failure in registry construction or lookup."""


class UnknownToolError(ToolRegistryError):
    """Raised when a requested tool is not registered."""


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """A registered tool and its descriptive metadata."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    tool: Tool

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )


class ToolRegistry:
    """Own tool discovery and dispatch, not tool implementations or validation."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        registered: dict[str, RegisteredTool] = {}
        for tool in tools:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", None)
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Registered tools require a non-empty name.")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Tool {name!r} requires a non-empty description.")
            metadata = getattr(tool, "metadata", None)
            input_schema = getattr(metadata, "input_schema", {})
            if not isinstance(input_schema, Mapping):
                raise ValueError(f"Tool {name!r} metadata input_schema must be a mapping.")
            if name in registered:
                raise ValueError(f"Duplicate tool name: {name}.")
            registered[name] = RegisteredTool(
                name=name,
                description=description,
                input_schema=dict(input_schema),
                tool=tool,
            )
        self._tools = dict(sorted(registered.items(), key=lambda item: item[0]))

    @classmethod
    def default(cls) -> "ToolRegistry":
        """Create the Phase 3 read-only tool set without executing anything."""

        return cls(
            (
                ListFilesTool(),
                ReadFileTool(),
                SearchCodeTool(),
                ProjectStructureTool(),
                ProjectContextTool(),
            )
        )

    @classmethod
    def with_write_file(cls) -> "ToolRegistry":
        """Create an explicit Phase 4.1 registry without changing AgentLoop defaults."""

        return cls((*cls.default()._tool_instances(), WriteFileTool()))

    @classmethod
    def with_file_modification(cls) -> "ToolRegistry":
        """Create an explicit Phase 4.2 registry with create and edit tools."""

        return cls((*cls.with_write_file()._tool_instances(), EditFileTool(), DeleteFileTool()))

    @classmethod
    def with_git_inspection(cls) -> "ToolRegistry":
        """Create an explicit read-only registry with the Git diff tool."""

        return cls((*cls.default()._tool_instances(), GitDiffTool(), GitStatusTool()))

    @classmethod
    def with_command_execution(cls) -> "ToolRegistry":
        """Create an explicit Phase 5.1 registry without changing AgentLoop defaults."""

        return cls((*cls.default()._tool_instances(), RunCommandTool()))

    @classmethod
    def with_command_policy(cls, policy: CommandPolicy | None = None) -> "ToolRegistry":
        """Create an explicit Phase 5.2 policy-wrapped execution registry."""

        return cls((*cls.default()._tool_instances(), PolicyRunCommandTool(policy)))

    @classmethod
    def with_application_execution(cls) -> "ToolRegistry":
        """Create an explicit Phase 5.4 application execution registry."""

        return cls((*cls.default()._tool_instances(), RunApplicationTool()))

    @classmethod
    def with_test_execution(cls) -> "ToolRegistry":
        """Create an explicit Phase 5.5 test execution registry."""

        return cls((*cls.default()._tool_instances(), RunTestsTool()))

    def _tool_instances(self) -> tuple[Tool, ...]:
        """Return registered concrete tools for controlled registry composition."""

        return tuple(item.tool for item in self._tools.values())

    def list(self) -> tuple[ToolMetadata, ...]:
        """Return metadata in deterministic name order."""

        return tuple(item.metadata for item in self._tools.values())

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""

        return tuple(self._tools)

    def get(self, name: str) -> Tool:
        """Return a registered tool or raise a structured registry error."""

        if not isinstance(name, str) or not name.strip():
            raise UnknownToolError("Tool name must be non-empty text.")
        try:
            return self._tools[name].tool
        except KeyError as exc:
            raise UnknownToolError(f"Unknown tool: {name}") from exc

    def metadata_for(self, name: str) -> ToolMetadata:
        """Return metadata for one registered name."""

        if name not in self._tools:
            raise UnknownToolError(f"Unknown tool: {name}")
        return self._tools[name].metadata

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """Dispatch through the Tool protocol without bypassing tool validation."""

        return self.get(name).run(arguments)


__all__ = [
    "RegisteredTool",
    "ToolRegistry",
    "ToolRegistryError",
    "UnknownToolError",
]
