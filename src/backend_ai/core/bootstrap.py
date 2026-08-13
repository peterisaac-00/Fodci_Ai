"""Minimal application startup composition for the foundation."""

from __future__ import annotations

from backend_ai.config import Settings, configure_logging, load_settings


def bootstrap(settings: Settings | None = None) -> Settings:
    """Initialize foundation configuration and project-scoped logging only."""

    resolved_settings = settings or load_settings()
    configure_logging(resolved_settings.log_level)
    return resolved_settings
