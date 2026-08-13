from __future__ import annotations

import pytest

from backend_ai.commands import CommandDispatcher, CommandParser


def test_parser_recognizes_command_and_preserves_arguments() -> None:
    parsed = CommandParser().parse("/Test hello   world")

    assert parsed.is_command
    assert parsed.command is not None
    assert parsed.command.name == "test"
    assert parsed.command.arguments == "hello   world"
    assert parsed.command.raw == "/Test hello   world"


def test_parser_keeps_normal_text_and_embedded_slashes_unchanged() -> None:
    parser = CommandParser()

    for text in ("hello", "Build /api/users", "   /help", ""):
        parsed = parser.parse(text)
        assert not parsed.is_command
        assert parsed.text == text


def test_parser_treats_slash_without_name_as_normal_input() -> None:
    parsed = CommandParser().parse("/")

    assert not parsed.is_command
    assert parsed.text == "/"


def test_dispatcher_routes_registered_command_case_insensitively() -> None:
    dispatcher = CommandDispatcher()
    dispatcher.register("help", lambda command: f"handled {command.arguments}")

    result = dispatcher.dispatch(CommandParser().parse("/HELP details"))

    assert result.kind == "handled"
    assert result.command is not None
    assert result.command.name == "help"
    assert result.command.arguments == "details"
    assert result.response == "handled details"


def test_dispatcher_reports_unknown_command_without_crashing() -> None:
    result = CommandDispatcher().dispatch(CommandParser().parse("/status"))

    assert result.kind == "unknown_command"
    assert result.response == "Unknown command: /status"


def test_dispatcher_passes_normal_input_unchanged() -> None:
    text = " Fix /api/users without changing this text "
    result = CommandDispatcher().dispatch(CommandParser().parse(text))

    assert result.kind == "normal_input"
    assert result.text == text
    assert result.command is None


def test_dispatcher_rejects_empty_or_whitespace_command_names() -> None:
    dispatcher = CommandDispatcher()

    with pytest.raises(ValueError):
        dispatcher.register(" ", lambda command: None)

    with pytest.raises(ValueError):
        dispatcher.register("two words", lambda command: None)
