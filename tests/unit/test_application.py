from __future__ import annotations

import logging
from pathlib import Path
from threading import Thread
from time import sleep

from backend_ai.application import Application, start_application
from backend_ai.terminal import InteractiveSession
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


def test_application_run_enters_injected_session_and_can_stop(tmp_path: Path) -> None:
    session = InteractiveSession()
    application = Application(
        Settings(project_root=tmp_path, log_level="WARNING"),
        session=session,
    )
    thread = Thread(target=application.run)

    thread.start()
    for _ in range(100):
        if session.is_active:
            break
        sleep(0.001)

    assert session.is_active
    application.stop()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert not session.is_active


def test_start_application_returns_ready_settings(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, log_level="WARNING")

    result = start_application(settings)

    assert result.project_root == tmp_path
    assert result.log_level == "WARNING"
