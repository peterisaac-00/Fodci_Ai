from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_exposes_official_fodci_console_script() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["scripts"] == {"fodci": "backend_ai.cli.main:main"}


def test_windows_installer_is_global_and_stable_checkpoint_aware() -> None:
    script = (ROOT / "scripts" / "install_fodci_global.ps1").read_text(encoding="utf-8")

    assert "$HOME \".fodci\"" in script
    assert "SetEnvironmentVariable(\"Path\"" in script
    assert "install --editable" in script
    assert "download_phase1312_checkpoint.ps1" in script
    assert "fodci" in script


def test_install_guide_documents_one_time_install_and_update() -> None:
    guide = (ROOT / "INSTALL.md").read_text(encoding="utf-8")

    assert "install_fodci_global.ps1" in guide
    assert "Get-Command fodci" in guide
    assert "git pull origin main" in guide
    assert "3af5d2b6009f5a0fd0ff98644d9666bd0c30f0dfe8994f91524ae6df11433bfa" in guide
