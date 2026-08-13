from __future__ import annotations

from io import StringIO
from pathlib import Path

from backend_ai.config import Settings, configure_logging
from backend_ai.core import bootstrap


def test_bootstrap_uses_supplied_settings(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, log_level="WARNING")

    assert bootstrap(settings) is settings


def test_configured_project_logger_is_usable_after_bootstrap() -> None:
    output = StringIO()
    logger = configure_logging("INFO", stream=output)

    logger.info("ready")

    assert "ready" in output.getvalue()
