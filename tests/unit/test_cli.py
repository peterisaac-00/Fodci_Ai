from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

from backend_ai.cli.main import main

cli_main = importlib.import_module("backend_ai.cli.main")


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cli_module_can_be_imported() -> None:
    module = importlib.import_module("backend_ai.cli.main")

    assert module.main is main


def test_cli_entry_point_returns_success_and_prints_confirmation(capsys: object) -> None:
    exit_code = main()
    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert exit_code == 0
    assert captured.out == (
        "Backend Engineering Agent\n"
        "Starting application...\n"
        "Application started successfully.\n"
    )
    assert captured.err == ""


def test_console_script_is_configured_in_project_metadata() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["scripts"]["backend-ai"] == "backend_ai.cli.main:main"


def test_cli_entry_point_invokes_application_startup(monkeypatch: object) -> None:
    calls: list[bool] = []

    def fake_start_application() -> None:
        calls.append(True)

    monkeypatch.setattr(cli_main, "start_application", fake_start_application)  # type: ignore[attr-defined]

    assert cli_main.main() == 0
    assert calls == [True]


def test_cli_module_does_not_initialize_agent_or_llm_boundaries() -> None:
    sys.modules.pop("backend_ai.agent", None)
    sys.modules.pop("backend_ai.llm", None)

    importlib.reload(importlib.import_module("backend_ai.cli.main"))

    assert "backend_ai.agent" not in sys.modules
    assert "backend_ai.llm" not in sys.modules
