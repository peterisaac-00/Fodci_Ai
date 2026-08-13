from __future__ import annotations

import logging
from pathlib import Path

from backend_ai.application import Application, start_application
from backend_ai.config import Settings


def test_application_start_uses_existing_configuration_and_logging(
    tmp_path: Path,
) -> None:
    settings = Settings(project_root=tmp_path, log_level="INFO")

    application = Application(settings)
    result = application.start()

    assert result is settings
    assert application.settings is settings
    assert logging.getLogger("backend_ai").handlers
    assert logging.getLogger("backend_ai").level == logging.INFO


def test_start_application_returns_ready_settings(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, log_level="WARNING")

    result = start_application(settings)

    assert result.project_root == tmp_path
    assert result.log_level == "WARNING"
