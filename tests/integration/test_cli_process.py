from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def fodci_executable() -> str:
    executable = shutil.which("fodci")
    if executable is None:
        pytest.fail("The installed package must expose the official fodci executable.")
    return executable


def _environment(project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PROJECT_ROOT"] = str(project_root)
    return environment


def test_fodci_process_runs_help_and_exit_successfully(
    fodci_executable: str,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [fodci_executable],
        input="/help\n/exit\n",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=_environment(tmp_path),
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0
    assert "Backend Engineering Agent" in completed.stdout
    assert "Interactive session started." in completed.stdout
    assert "Available commands:" in completed.stdout
    assert "/help" in completed.stdout
    assert "/exit" in completed.stdout
    assert "Goodbye." in completed.stdout
    assert completed.stderr == ""


def test_fodci_process_preserves_normal_input_and_unknown_commands(
    fodci_executable: str,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [fodci_executable],
        input="Build a REST API\n/status\n/exit\n",
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=_environment(tmp_path),
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0
    assert "Received: Build a REST API" in completed.stdout
    assert "Unknown command: /status" in completed.stdout
    assert "Goodbye." in completed.stdout


def test_fodci_process_handles_eof_cleanly(
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
        timeout=5,
    )

    assert completed.returncode == 0
    assert "Interactive session started." in completed.stdout
    assert "Traceback" not in completed.stderr


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
        timeout=5,
    )

    assert completed.returncode == 0
    assert "Goodbye." in completed.stdout
    assert completed.stderr == ""


def test_fodci_process_handles_ctrl_c_cleanly(
    fodci_executable: str,
    tmp_path: Path,
) -> None:
    import signal
    import time

    process = subprocess.Popen(
        [fodci_executable],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
        env=_environment(tmp_path),
    )
    try:
        time.sleep(0.1)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)

    assert process.returncode == 0
    assert "Interactive session started." in stdout
    assert "Traceback" not in stderr
    assert stderr == ""
