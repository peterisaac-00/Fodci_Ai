"""Command registration and dispatch boundaries for Phase 1.5."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from backend_ai.commands.parser import Command, ParsedInput

CommandHandler = Callable[[Command], str | None]
DispatchKind = Literal["normal_input", "handled", "unknown_command"]


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Structured outcome from classifying and dispatching one input."""

    kind: DispatchKind
    text: str
    command: Command | None = None
    response: str | None = None

    @property
    def is_command(self) -> bool:
        """Return whether the result came from command syntax."""

        return self.command is not None


class CommandDispatcher:
    """Dispatch recognized commands to explicitly registered handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}

    def register(self, name: str, handler: CommandHandler) -> None:
        """Register a case-insensitive command handler."""

        normalized_name = name.strip().casefold()
        if not normalized_name or any(character.isspace() for character in normalized_name):
            raise ValueError("Command names must contain non-whitespace text.")
        self._handlers[normalized_name] = handler

    def dispatch(self, parsed: ParsedInput) -> CommandResult:
        """Pass normal input through or dispatch a recognized command."""

        if parsed.command is None:
            return CommandResult(kind="normal_input", text=parsed.text)

        handler = self._handlers.get(parsed.command.name)
        if handler is None:
            return CommandResult(
                kind="unknown_command",
                text=parsed.text,
                command=parsed.command,
                response=f"Unknown command: {parsed.command.raw}",
            )

        return CommandResult(
            kind="handled",
            text=parsed.text,
            command=parsed.command,
            response=handler(parsed.command),
        )
