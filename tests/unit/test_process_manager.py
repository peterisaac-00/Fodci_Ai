from __future__ import annotations

from pathlib import Path
import sys

import pytest

from backend_ai.tools import (
    CommandPolicy,
    CommandRequest,
    PolicyRunCommandTool,
    ProcessLifecycle,
    ProcessManager,
    ProcessState,
    ToolError,
    ToolErrorCode,
)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def _request(root: Path, code: str, *, timeout: float = 1.0, stdout: int = 1_048_576, stderr: int = 1_048_576) -> CommandRequest:
    return CommandRequest((sys.executable, "-c", code), root, ".", inherit_environment=False, timeout_seconds=timeout, max_stdout_bytes=stdout, max_stderr_bytes=stderr)


def test_lifecycle_state_machine_accepts_valid_path_and_rejects_invalid_transition() -> None:
    lifecycle = ProcessLifecycle.requested()
    lifecycle = lifecycle.transition(ProcessState.VALIDATING).transition(ProcessState.STARTING).transition(ProcessState.RUNNING)
    assert lifecycle.current is ProcessState.RUNNING
    assert lifecycle.history == (ProcessState.REQUESTED, ProcessState.VALIDATING, ProcessState.STARTING, ProcessState.RUNNING)
    with pytest.raises(ToolError) as raised:
        lifecycle.transition(ProcessState.CLEANED_UP)
    assert raised.value.code is ToolErrorCode.PROCESS_INVALID_STATE


def test_process_manager_normal_completion_reports_state_history_and_output_metadata(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = ProcessManager().execute(_request(root, "print('hello')"))

    assert result.started is True
    assert result.completed is True
    assert result.succeeded is True
    assert result.exit_code == 0
    assert result.process_state == "CLEANED_UP"
    assert result.lifecycle_history == (
        "REQUESTED", "VALIDATING", "STARTING", "RUNNING", "COMPLETED", "CLEANED_UP"
    )
    assert result.termination_attempted is False
    assert result.killed is False
    assert result.stdout == "hello\n"
    assert result.stdout_bytes == len(result.stdout.encode("utf-8"))


def test_nonzero_exit_is_completed_but_not_success(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = ProcessManager().execute(_request(root, "import sys; print('err', file=sys.stderr); sys.exit(4)"))

    assert result.started is True
    assert result.completed is True
    assert result.succeeded is False
    assert result.exit_code == 4
    assert result.error_code == ToolErrorCode.PROCESS_NONZERO_EXIT.value
    assert result.stderr == "err\n"


def test_start_failure_is_structured_and_not_started(tmp_path: Path) -> None:
    root = _project(tmp_path)
    request = CommandRequest(("fodci-process-does-not-exist",), root, ".", inherit_environment=False, timeout_seconds=1.0)
    result = ProcessManager().execute(request)

    assert result.started is False
    assert result.completed is False
    assert result.succeeded is False
    assert result.process_state == "CLEANED_UP"
    assert result.error_code == ToolErrorCode.EXECUTABLE_NOT_FOUND.value
    assert result.lifecycle_history == ("REQUESTED", "VALIDATING", "STARTING", "FAILED_TO_START", "CLEANED_UP")


def test_timeout_attempts_termination_reaps_process_and_preserves_partial_output(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = ProcessManager(termination_grace_seconds=0.05).execute(_request(root, "import time; print('before', flush=True); time.sleep(2)", timeout=0.05))

    assert result.started is True
    assert result.completed is False
    assert result.succeeded is False
    assert result.timed_out is True
    assert result.error_code == ToolErrorCode.PROCESS_TIMEOUT.value
    assert result.termination_attempted is True
    assert result.lifecycle_history[:4] == ("REQUESTED", "VALIDATING", "STARTING", "RUNNING")
    assert "TIMED_OUT" in result.lifecycle_history
    assert "TERMINATING" in result.lifecycle_history
    assert result.process_state == "CLEANED_UP"
    assert "before\n" in result.stdout


def test_output_limit_drains_process_without_unbounded_result_or_deadlock(tmp_path: Path) -> None:
    root = _project(tmp_path)
    code = "import sys; sys.stdout.write('x' * 200000); sys.stdout.flush(); print('done')"
    result = ProcessManager().execute(_request(root, code, stdout=64))

    assert result.started is True
    assert result.completed is True
    assert result.succeeded is False
    assert result.error_code == ToolErrorCode.PROCESS_OUTPUT_LIMIT.value
    assert result.stdout_truncated is True
    assert result.stdout_bytes <= 64
    assert "OUTPUT_LIMIT_REACHED" in result.lifecycle_history
    assert result.process_state == "CLEANED_UP"


def test_stderr_output_limit_is_independent_and_invalid_utf8_is_reported(tmp_path: Path) -> None:
    root = _project(tmp_path)
    code = "import sys; sys.stderr.buffer.write(b'\\xff' * 1000); sys.stderr.flush()"
    result = ProcessManager().execute(_request(root, code, stderr=32))

    assert result.stderr_truncated is True
    assert result.stderr_bytes <= 32
    assert result.stderr_utf8_valid is False
    assert any("invalid UTF-8" in warning for warning in result.warnings)
    assert result.error_code == ToolErrorCode.PROCESS_OUTPUT_LIMIT.value


def test_stdin_remains_devnull_and_no_interactive_wait_is_possible(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = ProcessManager().execute(_request(root, "import sys; print(len(sys.stdin.read()))"))
    assert result.succeeded is True
    assert result.stdout == "0\n"


def test_process_manager_preserves_cwd_safety_and_does_not_make_policy_decisions(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ToolError) as raised:
        ProcessManager().execute(CommandRequest((sys.executable, "--version"), root, outside, inherit_environment=False))
    assert raised.value.code is ToolErrorCode.PATH_OUTSIDE_ROOT

    policy = CommandPolicy.default().with_executable_path(sys.executable)
    denied = PolicyRunCommandTool(policy)
    with pytest.raises(ToolError) as raised:
        denied.run({"argv": ["rm", "-rf", "x"], "project_root": str(root), "working_directory": ".", "inherit_environment": False})
    assert raised.value.code is ToolErrorCode.COMMAND_DENIED
