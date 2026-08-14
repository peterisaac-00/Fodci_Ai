"""Strict, deterministic model-output protocol for first Agent tool calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from backend_ai.agent.models import ToolCall


class ActionParseError(ValueError):
    """A model output did not follow the bounded ACTION protocol."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ParsedFinalAnswer:
    """Parsed final answer branch."""

    text: str


@dataclass(frozen=True, slots=True)
class ParsedToolAction:
    """Parsed but not yet registry-validated tool action."""

    name: str
    arguments: dict[str, Any]


ParsedAgentOutput = ParsedFinalAnswer | ParsedToolAction


def parse_agent_output(text: str) -> ParsedAgentOutput:
    """Parse final output or exactly one ACTION/ARGS request.

    The protocol is intentionally small:

    ``FINAL: answer``
    ``ACTION: tool_name``
    ``ARGS: {"key": "value"}``

    A response without an ACTION line is treated as a final answer. No free-form
    JSON or natural-language tool call is executed.
    """

    if not isinstance(text, str):
        raise ActionParseError("INVALID_ACTION", "Model output must be text.")
    stripped = text.strip()
    if not stripped:
        return ParsedFinalAnswer(text="")
    lines = text.splitlines()
    first_nonempty = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_nonempty is None:
        return ParsedFinalAnswer(text="")
    first = lines[first_nonempty].strip()
    if first.upper().startswith("FINAL:"):
        answer_lines = lines[first_nonempty:]
        answer_lines[0] = answer_lines[0][len("FINAL:"):].lstrip()
        return ParsedFinalAnswer(text="\n".join(answer_lines).strip())
    if not first.upper().startswith("ACTION:"):
        return ParsedFinalAnswer(text=text.strip())
    name = first[len("ACTION:"):].strip()
    if not name or any(character.isspace() for character in name) or not _valid_tool_name(name):
        raise ActionParseError("MALFORMED_ACTION", "ACTION must contain one valid tool name.")
    remaining = [line for line in lines[first_nonempty + 1:] if line.strip()]
    if not remaining or not remaining[0].strip().upper().startswith("ARGS:"):
        raise ActionParseError("MALFORMED_ACTION", "ACTION must be followed by an ARGS JSON object.")
    raw_arguments = remaining[0].strip()[len("ARGS:"):].strip()
    if not raw_arguments:
        raise ActionParseError("INVALID_ARGUMENT", "ARGS must contain a JSON object.")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ActionParseError("INVALID_ARGUMENT", f"ARGS is not valid JSON: {exc.msg}.") from exc
    if not isinstance(arguments, dict):
        raise ActionParseError("INVALID_ARGUMENT", "ARGS must decode to a JSON object.")
    if len(remaining) > 1 and any(line.strip() for line in remaining[1:]):
        raise ActionParseError("MALFORMED_ACTION", "A tool action may contain only ACTION and ARGS lines.")
    return ParsedToolAction(name=name, arguments=arguments)


def tool_call_from_action(action: ParsedToolAction, call_id: str) -> ToolCall:
    """Create the immutable ToolCall after protocol parsing."""

    return ToolCall(call_id=call_id, name=action.name, arguments=action.arguments)


def _valid_tool_name(name: str) -> bool:
    return all(character.isalnum() or character == "_" for character in name) and name[0].isalpha()


__all__ = [
    "ActionParseError",
    "ParsedAgentOutput",
    "ParsedFinalAnswer",
    "ParsedToolAction",
    "parse_agent_output",
    "tool_call_from_action",
]
