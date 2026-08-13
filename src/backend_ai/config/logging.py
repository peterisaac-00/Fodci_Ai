"""Centralized logging setup without application-global configuration state."""

from __future__ import annotations

import logging
from typing import TextIO

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None) -> logging.Logger:
    """Configure and return the project logger.

    The function owns handlers only for the ``backend_ai`` logger, so importing
    the package does not reconfigure a host application's root logger.
    """

    normalised_level = level.strip().upper()
    numeric_level = logging.getLevelName(normalised_level)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level!r}")

    logger = logging.getLogger("backend_ai")
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(numeric_level)
    logger.propagate = False
    return logger
