"""Application startup and composition boundary for the Backend Engineering Agent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, TextIO

from backend_ai.config import Settings
from backend_ai.core import ProjectContext, bootstrap, resolve_project_context
from backend_ai.terminal import InteractiveSession, StdinInputProvider

ProviderFactory = Callable[[Path], Any]
DEFAULT_CHECKPOINT_RELATIVE_PATH = Path("artifacts") / "checkpoints" / "fodci-testing-qa-v1.pt"
LEGACY_CHECKPOINT_RELATIVE_PATH = Path("artifacts") / "checkpoints" / "fodci-tiny-v1.pt"


def resolve_checkpoint_path(project_root: Path) -> Path:
    """Prefer the stable specialist checkpoint and fall back to Tiny v1."""

    candidates = (
        project_root / DEFAULT_CHECKPOINT_RELATIVE_PATH,
        project_root / LEGACY_CHECKPOINT_RELATIVE_PATH,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


class Application:
    """Compose startup, project context, provider, and terminal session."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session: InteractiveSession | None = None,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self._settings = settings
        self._provider_factory = provider_factory
        self.session = session or InteractiveSession(input_provider=StdinInputProvider())
        self.settings: Settings | None = None
        self.project_context: ProjectContext | None = None
        self.provider: Any | None = self.session.provider

    def start(self) -> Settings:
        """Initialize settings/project context and load one optional provider."""

        self.settings = bootstrap(self._settings)
        self.project_context = resolve_project_context(self.settings)
        self.session.project_context = self.project_context
        if self.provider is None and self._provider_factory is not None:
            checkpoint_path = resolve_checkpoint_path(self.project_context.root)
            self.provider = self._provider_factory(checkpoint_path)
            self.session.set_provider(self.provider)
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
    """Start the application through the foundation-only boundary."""

    return Application(settings).start()


def run_application(
    settings: Settings | None = None,
    *,
    output: TextIO | None = None,
) -> None:
    """Start the local provider and enter the persistent terminal session."""

    stream = output or sys.stdout
    from backend_ai.llm import FodciLocalProvider

    application = Application(
        settings,
        provider_factory=FodciLocalProvider.from_checkpoint,
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
