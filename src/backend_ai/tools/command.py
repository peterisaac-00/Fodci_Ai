"""Bounded argv-only command execution foundation.

This module intentionally provides process mechanics only. It does not implement
command safety policy, shell parsing, test execution, application running, or
AgentLoop integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import selectors
import stat
import subprocess
import time
from typing import Any, Literal, Mapping, Sequence

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.filesystem import _validate_root
from backend_ai.tools.read_file import _looks_like_windows_absolute, _reject_symlink_components

DEFAULT_COMMAND_TIMEOUT_SECONDS = 10.0
MAX_COMMAND_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_STDOUT_BYTES = 1_048_576
DEFAULT_MAX_STDERR_BYTES = 1_048_576
DEFAULT_MAX_ARGUMENTS = 256
DEFAULT_MAX_ARGUMENT_BYTES = 131_072
DEFAULT_MAX_ENVIRONMENT_BYTES = 131_072

CommandLifecycle = Literal["requested", "validated", "started", "completed", "timed_out", "failed"]


@dataclass(frozen=True, slots=True)
class CommandRequest:
    """Explicit structured command request; no shell command strings are accepted."""

    argv: tuple[str, ...]
    project_root: Path | str
    working_directory: Path | str
    environment: Mapping[str, str] | None = None
    inherit_environment: bool = True
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS
    max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.argv, tuple):
            object.__setattr__(self, "argv", tuple(self.argv))


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Immutable bounded process result without environment values or secrets."""

    argv: tuple[str, ...]
    working_directory: str
    lifecycle: CommandLifecycle
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool
    started: bool
    completed: bool
    succeeded: bool
    start_failed: bool
    stdout_truncated: bool
    stderr_truncated: bool
    stdout_utf8_valid: bool
    stderr_utf8_valid: bool
    termination: str | None
    error_code: str | None
    error_message: str | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "working_directory": self.working_directory,
            "lifecycle": self.lifecycle,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "started": self.started,
            "completed": self.completed,
            "succeeded": self.succeeded,
            "start_failed": self.start_failed,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "stdout_utf8_valid": self.stdout_utf8_valid,
            "stderr_utf8_valid": self.stderr_utf8_valid,
            "termination": self.termination,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _CapturedOutput:
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool
    termination: str
    returncode: int | None


def run_command(
    argv_or_request: Sequence[str] | CommandRequest,
    *,
    project_root: Path | str | None = None,
    working_directory: Path | str | None = None,
    environment: Mapping[str, str] | None = None,
    inherit_environment: bool = True,
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES,
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES,
) -> CommandResult:
    """Run one explicit argv command with no shell and bounded resources."""

    request = _coerce_request(
        argv_or_request,
        project_root=project_root,
        working_directory=working_directory,
        environment=environment,
        inherit_environment=inherit_environment,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )
    root, cwd, relative_cwd = _validate_working_directory(request.project_root, request.working_directory)
    _validate_request(request)
    env = _build_environment(request.environment, request.inherit_environment)
    start = time.monotonic()
    try:
        process = subprocess.Popen(
            request.argv,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
    except FileNotFoundError as exc:
        return _start_failure(request.argv, relative_cwd, start, ToolErrorCode.EXECUTABLE_NOT_FOUND, "Executable was not found.")
    except PermissionError as exc:
        return _start_failure(request.argv, relative_cwd, start, ToolErrorCode.PERMISSION_DENIED, "Permission denied while starting the executable.")
    except OSError as exc:
        return _start_failure(request.argv, relative_cwd, start, ToolErrorCode.COMMAND_FAILED, "The process could not be started.")

    captured = _capture_process(process, request.timeout_seconds, request.max_stdout_bytes, request.max_stderr_bytes)
    stdout, stdout_valid = _decode_output(captured.stdout)
    stderr, stderr_valid = _decode_output(captured.stderr)
    warnings: list[str] = []
    if captured.stdout_truncated:
        warnings.append("stdout exceeded max_stdout_bytes and the process was terminated.")
    if captured.stderr_truncated:
        warnings.append("stderr exceeded max_stderr_bytes and the process was terminated.")
    if not stdout_valid:
        warnings.append("stdout contained invalid UTF-8 and replacement decoding was used.")
    if not stderr_valid:
        warnings.append("stderr contained invalid UTF-8 and replacement decoding was used.")
    if captured.timed_out:
        return CommandResult(request.argv, relative_cwd, "timed_out", captured.returncode, stdout, stderr, _duration(start), True, True, False, False, False, captured.stdout_truncated, captured.stderr_truncated, stdout_valid, stderr_valid, captured.termination, ToolErrorCode.COMMAND_TIMEOUT.value, "The process exceeded the bounded timeout and was terminated.", tuple(warnings))
    if captured.stdout_truncated or captured.stderr_truncated:
        return CommandResult(request.argv, relative_cwd, "failed", captured.returncode, stdout, stderr, _duration(start), False, True, False, False, False, captured.stdout_truncated, captured.stderr_truncated, stdout_valid, stderr_valid, captured.termination, ToolErrorCode.OUTPUT_LIMIT.value, "The process exceeded an output limit and was terminated.", tuple(warnings))
    success = captured.returncode == 0
    return CommandResult(request.argv, relative_cwd, "completed", captured.returncode, stdout, stderr, _duration(start), False, True, True, success, False, False, False, stdout_valid, stderr_valid, "normal" if success else "nonzero_exit", None if success else ToolErrorCode.COMMAND_FAILED.value, None if success else "The process exited with a non-zero status.", tuple(warnings))


class RunCommandTool:
    """Opt-in Tool wrapper; never registered by ToolRegistry.default()."""

    name = "run_command"
    description = "Execute one explicit argv command in a validated project working directory without a shell."
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["argv", "project_root", "working_directory"],
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "project_root": {"type": "string"},
                "working_directory": {"type": "string"},
                "environment": {"type": "object"},
                "inherit_environment": {"type": "boolean"},
                "timeout_seconds": {"type": "number"},
                "max_stdout_bytes": {"type": "integer"},
                "max_stderr_bytes": {"type": "integer"},
            },
        },
    )

    def run(self, arguments: Mapping[str, Any]) -> CommandResult:
        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.COMMAND_INVALID, "run_command arguments must be an object.")
        if "command" in arguments:
            raise ToolError(ToolErrorCode.COMMAND_INVALID, "Shell command strings are not accepted; provide argv.")
        return run_command(
            arguments.get("argv"),
            project_root=arguments.get("project_root"),
            working_directory=arguments.get("working_directory"),
            environment=arguments.get("environment"),
            inherit_environment=arguments.get("inherit_environment", True),
            timeout_seconds=arguments.get("timeout_seconds", DEFAULT_COMMAND_TIMEOUT_SECONDS),
            max_stdout_bytes=arguments.get("max_stdout_bytes", DEFAULT_MAX_STDOUT_BYTES),
            max_stderr_bytes=arguments.get("max_stderr_bytes", DEFAULT_MAX_STDERR_BYTES),
        )


def _coerce_request(
    value: Sequence[str] | CommandRequest,
    *,
    project_root: Path | str | None,
    working_directory: Path | str | None,
    environment: Mapping[str, str] | None,
    inherit_environment: bool,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
) -> CommandRequest:
    if isinstance(value, CommandRequest):
        if any(item is not None for item in (project_root, working_directory, environment)):
            raise ToolError(ToolErrorCode.COMMAND_INVALID, "CommandRequest cannot be combined with project/environment overrides.")
        return value
    if project_root is None or working_directory is None:
        raise ToolError(ToolErrorCode.WORKING_DIRECTORY_INVALID, "project_root and working_directory are required explicitly.")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ToolError(ToolErrorCode.COMMAND_INVALID, "argv must be a non-empty sequence of strings, not a shell command string.")
    return CommandRequest(tuple(value), project_root, working_directory, environment, inherit_environment, timeout_seconds, max_stdout_bytes, max_stderr_bytes)


def _validate_request(request: CommandRequest) -> None:
    if not isinstance(request.inherit_environment, bool):
        raise ToolError(ToolErrorCode.COMMAND_INVALID, "inherit_environment must be boolean.")
    if not request.argv or len(request.argv) > DEFAULT_MAX_ARGUMENTS:
        raise ToolError(ToolErrorCode.COMMAND_INVALID, "argv must contain between 1 and 256 arguments.")
    total_bytes = 0
    for item in request.argv:
        if not isinstance(item, str) or not item or "\x00" in item:
            raise ToolError(ToolErrorCode.COMMAND_INVALID, "argv entries must be non-empty strings without NUL bytes.")
        total_bytes += len(item.encode("utf-8")) + 1
    if total_bytes > DEFAULT_MAX_ARGUMENT_BYTES:
        raise ToolError(ToolErrorCode.COMMAND_INVALID, "argv exceeds the bounded argument-size limit.")
    if not isinstance(request.timeout_seconds, (int, float)) or isinstance(request.timeout_seconds, bool) or not 0 < request.timeout_seconds <= MAX_COMMAND_TIMEOUT_SECONDS:
        raise ToolError(ToolErrorCode.COMMAND_INVALID, f"timeout_seconds must be > 0 and <= {MAX_COMMAND_TIMEOUT_SECONDS}.")
    for name, value in (("max_stdout_bytes", request.max_stdout_bytes), ("max_stderr_bytes", request.max_stderr_bytes)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ToolError(ToolErrorCode.COMMAND_INVALID, f"{name} must be a positive integer.")


def _validate_working_directory(project_root: Path | str, working_directory: Path | str) -> tuple[Path, Path, str]:
    root = _validate_root(project_root)
    raw = str(working_directory)
    if not raw.strip() or "\x00" in raw:
        raise ToolError(ToolErrorCode.WORKING_DIRECTORY_INVALID, "working_directory must be non-empty and NUL-free.")
    normalized = raw.replace("\\", "/")
    if _looks_like_windows_absolute(normalized):
        candidate = Path(normalized)
        if not candidate.is_absolute():
            raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "Windows drive and UNC working directories cannot bypass project_root.")
    else:
        candidate = Path(normalized)
    lexical = Path(os.path.normpath(str(candidate if candidate.is_absolute() else root / candidate)))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "working_directory is outside project_root.") from exc
    relative_text = relative.as_posix() if str(relative) != "." else "."
    if relative_text != ".":
        _reject_symlink_components(root, relative_text, lexical)
    try:
        mode = os.lstat(lexical).st_mode
    except FileNotFoundError as exc:
        raise ToolError(ToolErrorCode.WORKING_DIRECTORY_INVALID, "working_directory does not exist.") from exc
    except PermissionError as exc:
        raise ToolError(ToolErrorCode.PERMISSION_DENIED, "working_directory cannot be inspected.") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ToolError(ToolErrorCode.WORKING_DIRECTORY_INVALID, "working_directory must be a real directory, not a symlink or special file.")
    return root, lexical, relative_text


def _build_environment(overrides: Mapping[str, str] | None, inherit: bool) -> dict[str, str]:
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ToolError(ToolErrorCode.COMMAND_INVALID, "environment must be an object of string keys and values.")
    environment = dict(os.environ) if inherit else {}
    if overrides:
        total = 0
        for key, value in overrides.items():
            if not isinstance(key, str) or not isinstance(value, str) or not key or "\x00" in key or "\x00" in value:
                raise ToolError(ToolErrorCode.COMMAND_INVALID, "environment keys and values must be NUL-free strings.")
            total += len(key.encode("utf-8")) + len(value.encode("utf-8")) + 2
            environment[key] = value
        if total > DEFAULT_MAX_ENVIRONMENT_BYTES:
            raise ToolError(ToolErrorCode.COMMAND_INVALID, "explicit environment exceeds the bounded size limit.")
    return environment


def _capture_process(process: subprocess.Popen[bytes], timeout_seconds: float, max_stdout_bytes: int, max_stderr_bytes: int) -> _CapturedOutput:
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": max_stdout_bytes, "stderr": max_stderr_bytes}
    truncated = {"stdout": False, "stderr": False}
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    termination = "normal"
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                termination = "timeout_kill"
                _terminate_process(process)
                break
            events = selector.select(remaining)
            if not events:
                timed_out = True
                termination = "timeout_kill"
                _terminate_process(process)
                break
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                stream = key.data
                available = limits[stream] - len(buffers[stream])
                if len(chunk) > available:
                    if available > 0:
                        buffers[stream].extend(chunk[:available])
                    truncated[stream] = True
                    termination = "output_limit_kill"
                    _terminate_process(process)
                    break
                buffers[stream].extend(chunk)
            if termination != "normal":
                break
        if termination != "normal":
            selector.close()
        _wait_process(process)
        return _CapturedOutput(bytes(buffers["stdout"]), bytes(buffers["stderr"]), truncated["stdout"], truncated["stderr"], timed_out, termination, process.returncode)
    finally:
        try:
            selector.close()
        finally:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.kill()
    except OSError:
        pass


def _wait_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        process.wait()


def _decode_output(raw: bytes) -> tuple[str, bool]:
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), False


def _duration(start: float) -> float:
    return round(max(0.0, time.monotonic() - start), 6)


def _start_failure(argv: tuple[str, ...], working_directory: str, start: float, code: ToolErrorCode, message: str) -> CommandResult:
    return CommandResult(argv, working_directory, "failed", None, "", "", _duration(start), False, False, False, False, True, False, False, True, True, "start_failed", code.value, message, ())


__all__ = [
    "CommandRequest",
    "CommandResult",
    "DEFAULT_COMMAND_TIMEOUT_SECONDS",
    "DEFAULT_MAX_STDERR_BYTES",
    "DEFAULT_MAX_STDOUT_BYTES",
    "MAX_COMMAND_TIMEOUT_SECONDS",
    "RunCommandTool",
    "run_command",
]
