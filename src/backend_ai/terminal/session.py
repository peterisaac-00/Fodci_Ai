"""Persistent interactive-session lifecycle for Phase 1.3 and 1.4.

Phase 1.4 adds normal text input only. This module deliberately does not
interpret commands or dispatch work.
"""

from __future__ import annotations

from threading import Event
from typing import TextIO

from backend_ai.terminal.input import InputProvider


class InteractiveSession:
    """Keep the application alive and receive unprocessed user text."""

    def __init__(
        self,
        *,
        output: TextIO | None = None,
        input_provider: InputProvider | None = None,
    ) -> None:
        self._output = output
        self._input_provider = input_provider
        self._stop_event = Event()
        self._active = False
        self.received_inputs: list[str] = []

    @property
    def is_active(self) -> bool:
        """Return whether the session is currently running."""

        return self._active and not self._stop_event.is_set()

    def run(self) -> None:
        """Enter the session and receive normal text until EOF or stop.

        When no provider is injected, the session retains the Phase 1.3
        lifecycle-only behavior and waits for ``stop``. Production application
        startup injects ``StdinInputProvider`` for normal terminal input.
        """

        if self._active:
            raise RuntimeError("Interactive session is already running.")

        self._stop_event.clear()
        self._active = True
        if self._output is not None:
            print("Interactive session started.", file=self._output, flush=True)

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
                if value and self._output is not None:
                    print(f"Received: {value}", file=self._output, flush=True)
        except KeyboardInterrupt:
            self.stop()
        finally:
            self._active = False

    def stop(self) -> None:
        """Request that a running session stop and return from ``run``."""

        self._stop_event.set()


def _without_line_ending(value: str) -> str:
    """Remove only the line ending added by a stream read."""

    if value.endswith("\n"):
        value = value[:-1]
        if value.endswith("\r"):
            value = value[:-1]
    return value
