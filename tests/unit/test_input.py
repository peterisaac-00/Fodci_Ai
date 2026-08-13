from __future__ import annotations

from io import StringIO
from typing import NoReturn

from backend_ai.terminal import InputProvider, InteractiveSession, StdinInputProvider


class SequenceInputProvider:
    """Deterministic input double for session tests."""

    def __init__(self, values: list[str | None]) -> None:
        self._values = iter(values)

    def read(self) -> str | None:
        return next(self._values)


def test_input_provider_protocol_and_stdin_provider_are_importable() -> None:
    provider = StdinInputProvider(stream=StringIO("hello\n"))

    assert isinstance(provider, InputProvider)
    assert provider.read() == "hello\n"


def test_stdin_provider_preserves_text_and_handles_eof() -> None:
    provider = StdinInputProvider(stream=StringIO("Build a REST API\n"))

    assert provider.read() == "Build a REST API\n"
    assert provider.read() is None


def test_session_receives_multiple_inputs_and_routes_commands() -> None:
    output = StringIO()
    session = InteractiveSession(
        output=output,
        input_provider=SequenceInputProvider(["hello\n", "/exit\n", None]),
    )

    session.run()

    assert session.received_inputs == ["hello", "/exit"]
    assert "Received: hello" in output.getvalue()
    assert "Unknown command: /exit" in output.getvalue()
    assert session.dispatch_results[1].kind == "unknown_command"
    assert not session.is_active


def test_empty_input_is_preserved_and_does_not_stop_session() -> None:
    session = InteractiveSession(
        input_provider=SequenceInputProvider(["\n", "after empty\n", None]),
    )

    session.run()

    assert session.received_inputs == ["", "after empty"]
    assert not session.is_active


def test_keyboard_interrupt_stops_session_cleanly() -> None:
    class InterruptingProvider:
        def read(self) -> NoReturn:
            raise KeyboardInterrupt

    session = InteractiveSession(input_provider=InterruptingProvider())

    session.run()

    assert not session.is_active
