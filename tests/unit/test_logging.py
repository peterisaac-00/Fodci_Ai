from __future__ import annotations

from io import StringIO
import logging

from backend_ai.config import configure_logging


def test_logging_initialization_uses_requested_level_and_format() -> None:
    output = StringIO()
    logger = configure_logging("debug", stream=output)

    logger.debug("foundation logger is ready")

    assert logger.name == "backend_ai"
    assert logger.level == logging.DEBUG
    assert "DEBUG" in output.getvalue()
    assert "foundation logger is ready" in output.getvalue()
