"""Immutable structured state models for the first bounded Fodci Agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from backend_ai.tools.project_context import ProjectContext


class AgentMessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentStatus(str, Enum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    MAX_TOOL_CALLS = "max_tool_calls"
    CONTEXT_LIMIT = "context_limit"
    TOOL_ERROR = "tool_error"
    INFERENCE_ERROR = "inference_error"
    INVALID_ACTION = "invalid_action"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """One explicitly typed message in Agent execution state."""

    role: AgentMessageRole
    content: str
    name: str | None = None
    call_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise ValueError("AgentMessage content must be text.")
        if self.name is not None and not self.name.strip():
            raise ValueError("AgentMessage name must not be blank.")
        if self.call_id is not None and not self.call_id.strip():
            raise ValueError("AgentMessage call_id must not be blank.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "name": self.name,
            "call_id": self.call_id,
        }


@dataclass(frozen=True, slots=True)
class AgentTask:
    """Validated user task and explicit project boundary."""

    task: str
    project_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("AgentTask task must not be empty.")
        if not isinstance(self.project_root, Path):
            raise ValueError("AgentTask project_root must be a pathlib.Path.")

    def to_dict(self) -> dict[str, str]:
        return {"task": self.task, "project_root": str(self.project_root)}


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A parsed and validated request to dispatch one registered tool."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.name.strip():
            raise ValueError("ToolCall call_id and name must not be blank.")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured success or failure returned by a registered tool."""

    call_id: str
    tool_name: str
    success: bool
    data: Any = None
    error_code: str | None = None
    message: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.tool_name.strip():
            raise ValueError("ToolResult call_id and tool_name must not be blank.")
        if self.success and self.error_code is not None:
            raise ValueError("Successful ToolResult cannot contain error_code.")
        if not self.success and not self.error_code:
            raise ValueError("Failed ToolResult requires error_code.")

    def to_dict(self) -> dict[str, Any]:
        data = self.data.to_dict() if hasattr(self.data, "to_dict") else self.data
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "data": data,
            "error_code": self.error_code,
            "message": self.message,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One bounded model-decision and optional tool execution step."""

    index: int
    prompt_token_count: int
    model_output: str
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    context_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "prompt_token_count": self.prompt_token_count,
            "model_output": self.model_output,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "context_truncated": self.context_truncated,
        }


@dataclass(frozen=True, slots=True)
class AgentUsage:
    """Bounded execution counters useful for debugging and evaluation."""

    steps: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    tool_result_chars: int = 0
    context_truncations: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "tool_result_chars": self.tool_result_chars,
            "context_truncations": self.context_truncations,
        }


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Complete structured outcome of one bounded Agent run."""

    final_answer: str
    status: AgentStatus
    steps: tuple[AgentStep, ...]
    tool_calls: tuple[ToolCall, ...]
    tool_results: tuple[ToolResult, ...]
    project_context: ProjectContext | None
    stop_reason: str
    usage: AgentUsage
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_results": [result.to_dict() for result in self.tool_results],
            "project_context": self.project_context.to_dict() if self.project_context else None,
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Explicit bounds for the first local Agent loop."""

    max_steps: int = 8
    max_tool_calls: int = 8
    max_context_tokens: int = 256
    reserve_response_tokens: int = 32
    max_tool_result_chars: int = 4_000
    max_history_items: int = 8
    system_prompt: str = "Fodci. Output FINAL: text or ACTION: name ARGS: JSON."

    def __post_init__(self) -> None:
        for field_name in (
            "max_steps",
            "max_tool_calls",
            "max_context_tokens",
            "reserve_response_tokens",
            "max_tool_result_chars",
            "max_history_items",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
        if self.max_steps == 0:
            raise ValueError("max_steps must be positive.")
        if self.max_tool_calls == 0:
            raise ValueError("max_tool_calls must be positive because project_context is initialised first.")
        if self.max_context_tokens <= self.reserve_response_tokens:
            raise ValueError("max_context_tokens must exceed reserve_response_tokens.")
        if self.max_tool_result_chars == 0 or self.max_history_items == 0:
            raise ValueError("max_tool_result_chars and max_history_items must be positive.")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("system_prompt must contain text.")


__all__ = [
    "AgentConfig",
    "AgentMessage",
    "AgentMessageRole",
    "AgentResult",
    "AgentStatus",
    "AgentStep",
    "AgentTask",
    "AgentUsage",
    "ToolCall",
    "ToolResult",
]
