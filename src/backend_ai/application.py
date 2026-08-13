"""Application startup boundary for the Backend Engineering Agent."""

from __future__ import annotations

import sys
from typing import TextIO

from backend_ai.config import Settings
from backend_ai.core import ProjectContext, bootstrap, resolve_project_context
from backend_ai.terminal import InteractiveSession, StdinInputProvider


class Application:
    """Compose currently available application initialization steps.

    Future agent orchestration can be added behind this boundary without
    requiring the console entry point to know how startup is composed.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session: InteractiveSession | None = None,
    ) -> None:
        self._settings = settings
        self.session = session or InteractiveSession(input_provider=StdinInputProvider())
        self.settings: Settings | None = None
        self.project_context: ProjectContext | None = None

    def start(self) -> Settings:
        """Initialize configuration and logging, then return ready settings."""

        self.settings = bootstrap(self._settings)
        self.project_context = resolve_project_context(self.settings)
        self.session.project_context = self.project_context
        return self.settings

    def run(self) -> None:
        """Enter the interactive session after successful application startup."""

        if self.settings is None:
            self.start()
        self.session.run()

    def stop(self) -> None:
        """Request a clean stop for the active interactive session."""

        self.session.stop()


def start_application(settings: Settings | None = None) -> Settings:
    """Start the application through the application-level boundary."""

    return Application(settings).start()


def run_application(
    settings: Settings | None = None,
    *,
    output: TextIO | None = None,
) -> None:
    """Start the application and enter its persistent session lifecycle."""

    stream = output or sys.stdout
    application = Application(
        settings,
        session=InteractiveSession(
            output=stream,
            input_provider=StdinInputProvider(),
        ),
    )
    print("Starting application...", file=stream)
    application.start()
    print("Application started successfully.", file=stream)
    print(file=stream)
    application.run()
