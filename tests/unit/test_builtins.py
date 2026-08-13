from __future__ import annotations

from io import StringIO

from backend_ai.commands import CommandDispatcher, CommandParser, register_builtin_commands
from backend_ai.terminal import InteractiveSession


def test_help_uses_dynamic_registry_and_keeps_session_running() -> None:
    dispatcher = CommandDispatcher()
    register_builtin_commands(dispatcher)

    result = dispatcher.dispatch(CommandParser().parse("/help"))

    assert result.kind == "handled"
    assert result.response is not None
    assert "/help" in result.response
    assert "/exit" in result.response
    assert not result.exit_requested


def test_exit_returns_structured_stop_request_without_process_exit() -> None:
    dispatcher = CommandDispatcher()
    register_builtin_commands(dispatcher)

    result = dispatcher.dispatch(CommandParser().parse("/EXIT"))

    assert result.kind == "handled"
    assert result.response == "Goodbye."
    assert result.exit_requested


def test_help_and_exit_arguments_are_rejected_without_side_effects() -> None:
    dispatcher = CommandDispatcher()
    register_builtin_commands(dispatcher)

    help_result = dispatcher.dispatch(CommandParser().parse("/help now"))
    exit_result = dispatcher.dispatch(CommandParser().parse("/exit now"))

    assert help_result.response == "Usage: /help"
    assert not help_result.exit_requested
    assert exit_result.response == "Usage: /exit"
    assert not exit_result.exit_requested


def test_session_help_continues_and_exit_stops_cleanly() -> None:
    class SequenceProvider:
        def __init__(self) -> None:
            self.values = iter(["/help\n", "/exit\n"])

        def read(self) -> str | None:
            return next(self.values)

    output = StringIO()
    session = InteractiveSession(output=output, input_provider=SequenceProvider())

    session.run()

    text = output.getvalue()
    assert "Available commands:" in text
    assert "/help" in text
    assert "/exit" in text
    assert "Goodbye." in text
    assert len(session.dispatch_results) == 2
    assert not session.is_active
