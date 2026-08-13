"""Interactive terminal session with optional local provider integration."""

from __future__ import annotations

from threading import Event
from typing import TYPE_CHECKING, Any, TextIO

from backend_ai.commands import (
    CommandDispatcher,
    CommandParser,
    CommandResult,
    register_builtin_commands,
)
from backend_ai.core import ProjectContext
from backend_ai.terminal.input import InputProvider

if TYPE_CHECKING:
    from backend_ai.llm import LLMProvider, Message


class InteractiveSession:
    """Keep the application alive and route normal text to an injected provider."""

    def __init__(
        self,
        *,
        output: TextIO | None = None,
        input_provider: InputProvider | None = None,
        command_parser: CommandParser | None = None,
        command_dispatcher: CommandDispatcher | None = None,
        provider: Any | None = None,
        system_prompt: str | None = None,
        max_history_messages: int = 12,
    ) -> None:
        if max_history_messages < 2 or max_history_messages % 2:
            raise ValueError("max_history_messages must be a positive even number >= 2.")
        self._output = output
        self._input_provider = input_provider
        self._provider = provider
        self._system_prompt = system_prompt or getattr(provider, "system_prompt", None)
        self._max_history_messages = max_history_messages
        self.command_parser = command_parser or CommandParser()
        self.command_dispatcher = command_dispatcher or CommandDispatcher()
        register_builtin_commands(self.command_dispatcher)
        self._stop_event = Event()
        self._active = False
        self.received_inputs: list[str] = []
        self.dispatch_results: list[CommandResult] = []
        self.conversation_history: list[Any] = []
        self.project_context: ProjectContext | None = None

    @property
    def is_active(self) -> bool:
        """Return whether the session is currently running."""

        return self._active and not self._stop_event.is_set()

    @property
    def provider(self) -> Any | None:
        """Return the provider injected for this active session, if any."""

        return self._provider

    def set_provider(
        self,
        provider: Any,
        *,
        system_prompt: str | None = None,
    ) -> None:
        """Inject one provider before the session starts."""

        if self._active:
            raise RuntimeError("Cannot replace the provider during an active session.")
        self._provider = provider
        self._system_prompt = system_prompt or getattr(provider, "system_prompt", None)

    def run(self) -> None:
        """Receive normal text until a command, EOF, Ctrl+C, or stop request."""

        if self._active:
            raise RuntimeError("Interactive session is already running.")

        self._stop_event.clear()
        self._active = True
        if self._output is not None:
            print("Interactive session started.", file=self._output, flush=True)
            print(file=self._output, flush=True)

        try:
            while not self._stop_event.is_set():
                if self._input_provider is None:
                    self._stop_event.wait()
                    continue

                if self._output is not None:
                    print("You > ", end="", file=self._output, flush=True)
                value = self._input_provider.read()
                if value is None:
                    self.stop()
                    break

                value = _without_line_ending(value)
                self.received_inputs.append(value)
                result = self.command_dispatcher.dispatch(
                    self.command_parser.parse(value),
                )
                self.dispatch_results.append(result)
                if value and self._output is not None:
                    message = self._display_result(result, value)
                    if message:
                        print(file=self._output)
                        print(message, file=self._output, flush=True)
                        if not result.exit_requested:
                            print(file=self._output, flush=True)
                if result.exit_requested:
                    self.stop()
        except KeyboardInterrupt:
            self.stop()
        finally:
            self._active = False

    def stop(self) -> None:
        """Request that a running session stop and return from ``run``."""

        self._stop_event.set()

    def _display_result(self, result: CommandResult, value: str) -> str | None:
        if result.kind != "normal_input":
            return result.response
        if self._provider is None:
            return f"Received: {value}"
        try:
            response = self._generate_response(value)
        except Exception as exc:
            return f"Fodci error: {exc}"
        return f"Fodci > {response}"

    def _generate_response(self, value: str) -> str:
        from backend_ai.llm import LLMRequest, Message

        messages: list[Message] = []
        if self._system_prompt is not None:
            messages.append(Message(role="system", content=self._system_prompt))
        messages.extend(self.conversation_history)
        user_message = Message(role="user", content=value)
        messages.append(user_message)
        response = self._provider.generate(LLMRequest(messages=tuple(messages)))
        if not isinstance(response.text, str):
            raise ValueError("Provider returned a response with invalid text.")
        assistant_message = Message(role="assistant", content=response.text)
        self.conversation_history.extend((user_message, assistant_message))
        del self.conversation_history[: len(self.conversation_history) - self._max_history_messages]
        return response.text


def _without_line_ending(value: str) -> str:
    """Remove only the line ending added by a stream read."""

    if value.endswith("\n"):
        value = value[:-1]
        if value.endswith("\r"):
            value = value[:-1]
    return value
