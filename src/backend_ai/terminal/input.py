"""Input boundaries for the Phase 1.4 terminal session."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Protocol, TextIO, runtime_checkable


@runtime_checkable
class InputProvider(Protocol):
    """Provide one unprocessed line of normal user text."""

    def read(self) -> str | None:
        """Return the next line, or ``None`` when input reaches EOF."""


class StdinInputProvider:
    """Read normal text from a terminal stream without interpreting it."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        input_function: Callable[[str], str] | None = None,
    ) -> None:
        self._stream = stream
        self._input_function = input_function

    def read(self) -> str | None:
        """Read one line from stdin, preserving content except its line ending."""

        try:
            if self._input_function is not None:
                return self._input_function("")
            if self._stream is not None:
                value = self._stream.readline()
                return None if value == "" else value
            return input()
        except EOFError:
            return None
