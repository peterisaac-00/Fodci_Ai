from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from backend_ai.agent.registry import ToolRegistry
from backend_ai.tools import (
    CommandRequest,
    RunCommandTool,
    ToolError,
    ToolErrorCode,
    run_command,
)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def _python(root: Path, code: str, *args: str, **kwargs):
    return run_command(
        (sys.executable, "-c", code, *args),
        project_root=root,
        working_directory=".",
        **kwargs,
    )


def test_success_result_has_explicit_argv_cwd_and_lifecycle(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = _python(root, "print('ok')")

    assert result.lifecycle == "completed"
    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    assert result.stderr == ""
    assert result.succeeded is True
    assert result.started is True
    assert result.completed is True
    assert result.timed_out is False
    assert result.working_directory == "."
    assert result.error_code is None
    assert str(root) not in str(result.to_dict())


def test_nonzero_exit_preserves_separate_streams_and_structured_failure(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = _python(root, "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)")

    assert result.lifecycle == "completed"
    assert result.exit_code == 7
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert result.succeeded is False
    assert result.error_code == ToolErrorCode.COMMAND_FAILED.value
    assert result.termination == "nonzero_exit"


def test_timeout_terminates_bounded_process_without_interactive_blocking(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = _python(root, "import time; print('before', flush=True); time.sleep(1)", timeout_seconds=0.05)

    assert result.lifecycle == "timed_out"
    assert result.timed_out is True
    assert result.completed is False
    assert result.error_code == ToolErrorCode.COMMAND_TIMEOUT.value
    assert "before\n" in result.stdout


def test_output_limits_are_bounded_and_reported(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = _python(root, "print('x' * 1000)", max_stdout_bytes=32)

    assert result.lifecycle == "failed"
    assert result.stdout_truncated is True
    assert result.error_code == ToolErrorCode.OUTPUT_LIMIT.value
    assert len(result.stdout.encode("utf-8")) <= 32
    assert result.completed is False


def test_stderr_limit_is_independent(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = _python(root, "import sys; sys.stderr.write('e' * 1000)", max_stderr_bytes=16)

    assert result.stderr_truncated is True
    assert result.stdout_truncated is False
    assert result.error_code == ToolErrorCode.OUTPUT_LIMIT.value
    assert len(result.stderr.encode("utf-8")) <= 16


def test_unicode_and_invalid_utf8_output_are_deterministic(tmp_path: Path) -> None:
    root = _project(tmp_path)
    unicode_result = _python(root, "print('مرحبا')")
    invalid_result = _python(root, "import sys; sys.stdout.buffer.write(b'\\xff\\x00')")

    assert unicode_result.stdout == "مرحبا\n"
    assert unicode_result.stdout_utf8_valid is True
    assert invalid_result.stdout_utf8_valid is False
    assert "�" in invalid_result.stdout
    assert any("invalid UTF-8" in warning for warning in invalid_result.warnings)


def test_explicit_argv_does_not_interpret_shell_syntax(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = _python(root, "import sys; print(sys.argv[1])", "$HOME")

    assert result.exit_code == 0
    assert result.stdout == "$HOME\n"

    with pytest.raises(ToolError) as raised:
        RunCommandTool().run({"command": "echo unsafe", "project_root": str(root), "working_directory": "."})
    assert raised.value.code is ToolErrorCode.COMMAND_INVALID


def test_environment_is_explicit_and_never_returned(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = _python(root, "import os; print(os.environ['FODCI_TEST_VALUE'])", environment={"FODCI_TEST_VALUE": "secret-value"}, inherit_environment=False)

    assert result.stdout == "secret-value\n"
    assert "secret-value" not in str(result.to_dict().get("warnings"))
    assert "environment" not in result.to_dict()


def test_stdin_is_noninteractive_and_reaches_eof(tmp_path: Path) -> None:
    root = _project(tmp_path)
    result = _python(root, "import sys; print(len(sys.stdin.read()))")

    assert result.exit_code == 0
    assert result.stdout == "0\n"


def test_explicit_working_directory_is_root_contained_and_reported_relative(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "subdir").mkdir()
    result = run_command((sys.executable, "-c", "import os; print(os.getcwd().endswith('subdir'))"), project_root=root, working_directory="subdir")

    assert result.exit_code == 0
    assert result.stdout == "True\n"
    assert result.working_directory == "subdir"

    outside = tmp_path / "outside"
    outside.mkdir()
    for cwd in ("../outside", str(outside), r"C:\\outside", r"\\\\server\\share"):
        with pytest.raises(ToolError) as raised:
            run_command((sys.executable, "-c", "print('no')"), project_root=root, working_directory=cwd)
        assert raised.value.code in {ToolErrorCode.PATH_OUTSIDE_ROOT, ToolErrorCode.WORKING_DIRECTORY_INVALID}


def test_invalid_working_directory_root_symlink_and_malformed_argv_are_structured(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with pytest.raises(ToolError) as raised:
        run_command((sys.executable, "-c", "print('no')"), project_root=root, working_directory="missing")
    assert raised.value.code is ToolErrorCode.WORKING_DIRECTORY_INVALID

    with pytest.raises(ToolError) as raised:
        run_command("python -c print('no')", project_root=root, working_directory=".")  # type: ignore[arg-type]
    assert raised.value.code is ToolErrorCode.COMMAND_INVALID

    with pytest.raises(ToolError) as raised:
        run_command(("",), project_root=root, working_directory=".")
    assert raised.value.code is ToolErrorCode.COMMAND_INVALID

    missing_executable = run_command(("fodci-definitely-not-an-executable",), project_root=root, working_directory=".")
    assert missing_executable.start_failed is True
    assert missing_executable.error_code == ToolErrorCode.EXECUTABLE_NOT_FOUND.value
    assert missing_executable.completed is False


def test_command_request_supports_custom_limits_and_rejects_mixed_overrides(tmp_path: Path) -> None:
    root = _project(tmp_path)
    request = CommandRequest((sys.executable, "-c", "print('request')"), root, ".", timeout_seconds=1.0, max_stdout_bytes=128, max_stderr_bytes=128)
    result = run_command(request)
    assert result.stdout == "request\n"

    with pytest.raises(ToolError) as raised:
        run_command(request, project_root=root)
    assert raised.value.code is ToolErrorCode.COMMAND_INVALID


def test_tool_and_registry_are_opt_in_and_agent_default_is_unchanged(tmp_path: Path) -> None:
    root = _project(tmp_path)
    tool = RunCommandTool()
    result = tool.run({"argv": [sys.executable, "-c", "print('tool')"], "project_root": str(root), "working_directory": "."})
    assert result.stdout == "tool\n"
    assert "run_command" not in ToolRegistry.default().names()
    assert "run_command" in ToolRegistry.with_command_execution().names()


def test_executor_itself_does_not_mutate_project_files(tmp_path: Path) -> None:
    root = _project(tmp_path)
    marker = root / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")
    result = _python(root, "print('safe')")

    assert result.succeeded is True
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) == ["marker.txt"]
