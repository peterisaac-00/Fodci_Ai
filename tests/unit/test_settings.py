from __future__ import annotations

from pathlib import Path

import pytest

from backend_ai.config import load_settings


def test_settings_use_safe_defaults(tmp_path: Path) -> None:
    settings = load_settings({}, cwd=tmp_path)

    assert settings.log_level == "INFO"
    assert settings.project_root == tmp_path.resolve()


def test_settings_read_and_normalise_environment_values(tmp_path: Path) -> None:
    settings = load_settings(
        {"LOG_LEVEL": "debug", "PROJECT_ROOT": "workspace"},
        cwd=tmp_path,
    )

    assert settings.log_level == "DEBUG"
    assert settings.project_root == (tmp_path / "workspace").resolve()


def test_settings_reject_unsupported_log_level(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported LOG_LEVEL"):
        load_settings({"LOG_LEVEL": "verbose"}, cwd=tmp_path)
