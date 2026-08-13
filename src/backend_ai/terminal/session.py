"""Persistent interactive-session lifecycle for Phase 1.3.

This module intentionally does not read input, parse commands, or dispatch work.
It only provides a controllable lifecycle that later phases can build on.
"""

from __future__ import annotations

from threading import Event
from typing import TextIO


class InteractiveSession:
    """Keep the application alive until the session is explicitly stopped."""

    def __init__(self, *, output: TextIO | None = None) -> None:
        self._output = output
        self._stop_event = Event()
        self._active = False

    @property
    def is_active(self) -> bool:
        """Return whether the session is currently running."""

        return self._active and not self._stop_event.is_set()

    def run(self) -> None:
        """Enter the session lifecycle and wait for a clean stop signal.

        No user input is consumed in Phase 1.3. ``stop`` is the lifecycle
        control that tests and future application code can use to terminate it.
        A keyboard interrupt also exits the wait cleanly.
        """

        if self._active:
            raise RuntimeError("Interactive session is already running.")

        self._stop_event.clear()
        self._active = True
        if self._output is not None:
            print("Interactive session started.", file=self._output, flush=True)

        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            self.stop()
        finally:
            self._active = False

    def stop(self) -> None:
        """Request that a running session stop and return from ``run``."""

        self._stop_event.set()
