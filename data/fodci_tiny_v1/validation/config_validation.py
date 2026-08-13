"""Validation-only configuration and health-check example."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    environment: str
    database_url: str
    debug: bool = False

    @classmethod
    def from_environment(cls) -> "Settings":
        environment = os.getenv("APP_ENV", "production")
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is required")
        if environment == "production" and os.getenv("DEBUG", "0") == "1":
            raise RuntimeError("debug mode is disabled in production")
        return cls(
            environment=environment,
            database_url=database_url,
            debug=os.getenv("DEBUG", "0") == "1",
        )


def health_payload(database_ok: bool, queue_ok: bool) -> dict[str, object]:
    checks = {"database": database_ok, "queue": queue_ok}
    healthy = all(checks.values())
    return {
        "status": "ok" if healthy else "degraded",
        "checks": checks,
    }
