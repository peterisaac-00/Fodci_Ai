"""Application startup boundary for the Backend Engineering Agent."""

from __future__ import annotations

from backend_ai.config import Settings
from backend_ai.core import bootstrap


class Application:
    """Compose currently available application initialization steps.

    Future agent orchestration can be added behind this boundary without
    requiring the console entry point to know how startup is composed.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self.settings: Settings | None = None

    def start(self) -> Settings:
        """Initialize configuration and logging, then return ready settings."""

        self.settings = bootstrap(self._settings)
        return self.settings


def start_application(settings: Settings | None = None) -> Settings:
    """Start the application through the application-level boundary."""

    return Application(settings).start()
