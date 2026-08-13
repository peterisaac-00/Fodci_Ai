from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REAL_CHECKPOINT = REPOSITORY_ROOT / "artifacts" / "checkpoints" / "fodci-tiny-v1.pt"


@pytest.fixture
def fodci_executable() -> str:
    executable = shutil.which("fodci")
    if executable is None:
        pytest.fail("The installed package must expose the official fodci executable.")
    return executable


@pytest.fixture
def repository_root() -> Path:
    if not REAL_CHECKPOINT.is_file():
        pytest.skip("existing local Fodci Tiny v1 checkpoint is unavailable")
    return REPOSITORY_ROOT


def _environment(project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PROJECT_ROOT"] = str(project_root)
    return environment


def test_fodci_process_runs_help_and_exit_successfully(
    fodci_executable: str,
    tmp_path: Path,
    repository_root: Path,
) -> None:
    completed = subprocess.run(
        [fodci_executable],
        input="/help\n/exit\n",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=_environment(repository_root),
        check=False,
        timeout=20,
    )

    assert completed.returncode == 0
    assert "Backend Engineering Agent" in completed.stdout
    assert "Interactive session started." in completed.stdout
    assert "Available commands:" in completed.stdout
    assert "/help" in completed.stdout
    assert "/exit" in completed.stdout
    assert "Goodbye." in completed.stdout
    assert completed.stderr == ""


def test_fodci_process_reaches_local_provider_and_exit_successfully(
    fodci_executable: str,
    tmp_path: Path,
    repository_root: Path,
) -> None:
    completed = subprocess.run(
        [fodci_executable],
        input="Hi\n/exit\n",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=_environment(repository_root),
        check=False,
        timeout=20,
    )

    assert completed.returncode == 0
    assert "You >" in completed.stdout
    assert "Fodci >" in completed.stdout
    assert "Goodbye." in completed.stdout
    assert "Fodci error:" not in completed.stdout
    assert completed.stderr == ""


def test_fodci_process_preserves_conversation_and_unknown_commands(
    fodci_executable: str,
    tmp_path: Path,
    repository_root: Path,
) -> None:
    completed = subprocess.run(
        [fodci_executable],
        input="Build a REST API\n/status\nWrite SQL\n/exit\n",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=_environment(repository_root),
        check=False,
        timeout=20,
    )

    assert completed.returncode == 0
    assert completed.stdout.count("Fodci >") == 2
    assert "Unknown command: /status" in completed.stdout
    assert "Goodbye." in completed.stdout
    assert completed.stderr == ""


def test_fodci_process_handles_eof_cleanly(
    fodci_executable: str,
    tmp_path: Path,
    repository_root: Path,
) -> None:
    completed = subprocess.run(
        [fodci_executable],
        input="",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=_environment(repository_root),
        check=False,
        timeout=20,
    )

    assert completed.returncode == 0
    assert "Interactive session started." in completed.stdout
    assert "Traceback" not in completed.stderr
    assert completed.stderr == ""


def test_fodci_process_rejects_invalid_project_root_without_traceback(
    fodci_executable: str,
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-project"
    completed = subprocess.run(
        [fodci_executable],
        input="",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=_environment(missing_root),
        check=False,
        timeout=5,
    )

    assert completed.returncode == 1
    assert "Invalid project root:" in completed.stderr
    assert "Path does not exist or is not a directory." in completed.stderr
    assert "Traceback" not in completed.stderr


def test_fodci_process_reports_missing_checkpoint_without_traceback(
    fodci_executable: str,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [fodci_executable],
        input="",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=_environment(tmp_path),
        check=False,
        timeout=20,
    )

    assert completed.returncode == 1
    assert "Fodci checkpoint is unavailable:" in completed.stderr
    assert "no fallback model" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_fodci_process_uses_current_directory_when_project_root_is_unset(
    fodci_executable: str,
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PROJECT_ROOT", None)
    completed = subprocess.run(
        [fodci_executable],
        input="/exit\n",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=environment,
        check=False,
        timeout=20,
    )

    assert completed.returncode == 1
    assert "Fodci checkpoint is unavailable:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_fodci_process_handles_ctrl_c_cleanly(
    fodci_executable: str,
    tmp_path: Path,
    repository_root: Path,
) -> None:
    process = subprocess.Popen(
        [fodci_executable],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
        env=_environment(repository_root),
    )
    try:
        time.sleep(0.2)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)

    assert process.returncode == 0
    assert "Traceback" not in stderr
    assert stderr == ""
