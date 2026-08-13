"""Command registration and dispatch boundaries for Phase 1.5."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from backend_ai.commands.parser import Command, ParsedInput


@dataclass(frozen=True, slots=True)
class CommandResponse:
    """Handler response that can request clean session termination."""

    text: str | None = None
    exit_requested: bool = False


CommandHandler = Callable[[Command], str | CommandResponse | None]
DispatchKind = Literal["normal_input", "handled", "unknown_command"]


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Structured outcome from classifying and dispatching one input."""

    kind: DispatchKind
    text: str
    command: Command | None = None
    response: str | None = None
    exit_requested: bool = False

    @property
    def is_command(self) -> bool:
        """Return whether the result came from command syntax."""

        return self.command is not None


class CommandDispatcher:
    """Dispatch recognized commands to explicitly registered handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}
        self._descriptions: dict[str, str] = {}

    def register(
        self,
        name: str,
        handler: CommandHandler,
        *,
        description: str = "",
    ) -> None:
        """Register a case-insensitive command handler and its help text."""

        normalized_name = name.strip().casefold()
        if not normalized_name or any(character.isspace() for character in normalized_name):
            raise ValueError("Command names must contain non-whitespace text.")
        self._handlers[normalized_name] = handler
        self._descriptions[normalized_name] = description

    def has(self, name: str) -> bool:
        """Return whether a command name is registered."""

        return name.strip().casefold() in self._handlers

    def help_text(self) -> str:
        """Render help from the registered command metadata."""

        lines = ["Available commands:"]
        for name in sorted(self._handlers):
            description = self._descriptions.get(name, "")
            lines.append(f"  /{name:<8} {description}".rstrip())
        return "\n".join(lines)

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

        handled = handler(parsed.command)
        if isinstance(handled, CommandResponse):
            response = handled
        else:
            response = CommandResponse(text=handled)

        return CommandResult(
            kind="handled",
            text=parsed.text,
            command=parsed.command,
            response=response.text,
            exit_requested=response.exit_requested,
        )
