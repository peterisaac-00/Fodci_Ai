"""Typed application settings with explicit environment parsing."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    database_url: str
    log_level: str = "INFO"
    request_timeout_seconds: float = 5.0

    @classmethod
    def from_environment(cls) -> "Settings":
        database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            raise RuntimeError("DATABASE_URL is required")
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("LOG_LEVEL is not supported")
        timeout = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "5"))
        if timeout <= 0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be positive")
        return cls(database_url=database_url, log_level=log_level, request_timeout_seconds=timeout)


def readiness_check(settings: Settings, database: object) -> bool:
    """A health endpoint should report dependency readiness, not only process liveness."""

    try:
        database.execute("SELECT 1")
    except Exception:
        return False
    return bool(settings.database_url)
