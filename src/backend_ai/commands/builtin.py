"""Deterministic built-in CLI commands for Phase 1.6."""

from __future__ import annotations

from backend_ai.commands.dispatcher import CommandDispatcher, CommandResponse
from backend_ai.commands.parser import Command


def register_builtin_commands(dispatcher: CommandDispatcher) -> None:
    """Register local help and exit behavior once on a dispatcher."""

    if not dispatcher.has("help"):
        dispatcher.register(
            "help",
            lambda command: _help_response(dispatcher, command),
            description="Show available commands",
        )
    if not dispatcher.has("exit"):
        dispatcher.register(
            "exit",
            _exit_response,
            description="Exit the application",
        )


def _help_response(dispatcher: CommandDispatcher, command: Command) -> str:
    """Return dynamic help; arguments are intentionally unsupported."""

    if command.arguments:
        return "Usage: /help"
    return dispatcher.help_text()


def _exit_response(command: Command) -> CommandResponse:
    """Request a clean stop only for the argument-free exit command."""

    if command.arguments:
        return CommandResponse(text="Usage: /exit")
    return CommandResponse(text="Goodbye.", exit_requested=True)
