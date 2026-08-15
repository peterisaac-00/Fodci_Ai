"""Bounded lifecycle management for an already-approved CommandRequest."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from typing import Any

from backend_ai.tools.base import ToolError, ToolErrorCode
from backend_ai.tools.command import (
    CommandRequest,
    CommandResult,
    _build_environment,
    _decode_output,
    _duration,
    _validate_request,
    _validate_working_directory,
)

DEFAULT_TERMINATION_GRACE_SECONDS = 0.25


class ProcessState(str, Enum):
    REQUESTED = "REQUESTED"
    VALIDATING = "VALIDATING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED_TO_START = "FAILED_TO_START"
    TIMED_OUT = "TIMED_OUT"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"
    KILLED = "KILLED"
    OUTPUT_LIMIT_REACHED = "OUTPUT_LIMIT_REACHED"
    CLEANED_UP = "CLEANED_UP"


_ALLOWED_TRANSITIONS: dict[ProcessState, frozenset[ProcessState]] = {
    ProcessState.REQUESTED: frozenset({ProcessState.VALIDATING}),
    ProcessState.VALIDATING: frozenset({ProcessState.STARTING, ProcessState.FAILED_TO_START}),
    ProcessState.STARTING: frozenset({ProcessState.RUNNING, ProcessState.FAILED_TO_START}),
    ProcessState.RUNNING: frozenset({ProcessState.COMPLETED, ProcessState.TIMED_OUT, ProcessState.TERMINATING, ProcessState.OUTPUT_LIMIT_REACHED}),
    ProcessState.OUTPUT_LIMIT_REACHED: frozenset({ProcessState.COMPLETED, ProcessState.TIMED_OUT, ProcessState.TERMINATING}),
    ProcessState.TIMED_OUT: frozenset({ProcessState.TERMINATING}),
    ProcessState.TERMINATING: frozenset({ProcessState.TERMINATED, ProcessState.KILLED}),
    ProcessState.TERMINATED: frozenset({ProcessState.CLEANED_UP}),
    ProcessState.KILLED: frozenset({ProcessState.CLEANED_UP}),
    ProcessState.COMPLETED: frozenset({ProcessState.CLEANED_UP}),
    ProcessState.FAILED_TO_START: frozenset({ProcessState.CLEANED_UP}),
    ProcessState.CLEANED_UP: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ProcessLifecycle:
    """Immutable lifecycle history for one process execution."""

    current: ProcessState
    history: tuple[ProcessState, ...]

    @classmethod
    def requested(cls) -> "ProcessLifecycle":
        return cls(ProcessState.REQUESTED, (ProcessState.REQUESTED,))

    def transition(self, next_state: ProcessState) -> "ProcessLifecycle":
        if next_state not in _ALLOWED_TRANSITIONS[self.current]:
            raise ToolError(ToolErrorCode.PROCESS_INVALID_STATE, f"Invalid process state transition: {self.current.value} -> {next_state.value}.")
        return ProcessLifecycle(next_state, (*self.history, next_state))

    def to_dict(self) -> dict[str, Any]:
        return {"current": self.current.value, "history": [state.value for state in self.history]}


@dataclass(frozen=True, slots=True)
class ProcessTermination:
    """Immutable technical termination outcome; no PID or environment is exposed."""

    attempted: bool
    graceful: bool
    killed: bool
    succeeded: bool
    reason: str
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "graceful": self.graceful,
            "killed": self.killed,
            "succeeded": self.succeeded,
            "reason": self.reason,
            "warning": self.warning,
        }


class ProcessManager:
    """Manage one bounded approved process; policy decisions remain outside this class."""

    def __init__(self, *, termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS) -> None:
        if not isinstance(termination_grace_seconds, (int, float)) or isinstance(termination_grace_seconds, bool) or termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be positive")
        self.termination_grace_seconds = float(termination_grace_seconds)

    def execute(self, request: CommandRequest) -> CommandResult:
        lifecycle = ProcessLifecycle.requested()
        start = time.monotonic()
        lifecycle = lifecycle.transition(ProcessState.VALIDATING)
        root, cwd, relative_cwd = _validate_working_directory(request.project_root, request.working_directory)
        _validate_request(request)
        environment = _build_environment(request.environment, request.inherit_environment)
        lifecycle = lifecycle.transition(ProcessState.STARTING)
        popen_kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        elif os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            process = subprocess.Popen(request.argv, **popen_kwargs)
        except FileNotFoundError:
            lifecycle = lifecycle.transition(ProcessState.FAILED_TO_START).transition(ProcessState.CLEANED_UP)
            return self._result(request, relative_cwd, lifecycle, start, None, b"", b"", False, False, ProcessTermination(False, False, False, True, "start_failed"), "Executable was not found.", ToolErrorCode.EXECUTABLE_NOT_FOUND)
        except PermissionError:
            lifecycle = lifecycle.transition(ProcessState.FAILED_TO_START).transition(ProcessState.CLEANED_UP)
            return self._result(request, relative_cwd, lifecycle, start, None, b"", b"", False, False, ProcessTermination(False, False, False, True, "start_failed"), "Permission denied while starting the executable.", ToolErrorCode.PERMISSION_DENIED)
        except OSError:
            lifecycle = lifecycle.transition(ProcessState.FAILED_TO_START).transition(ProcessState.CLEANED_UP)
            return self._result(request, relative_cwd, lifecycle, start, None, b"", b"", False, False, ProcessTermination(False, False, False, True, "start_failed"), "The process could not be started.", ToolErrorCode.PROCESS_START_FAILED)

        lifecycle = lifecycle.transition(ProcessState.RUNNING)
        stdout, stderr, stdout_truncated, stderr_truncated, timed_out, termination = self._collect(process, request, lifecycle)
        lifecycle = termination[0]
        termination_result = termination[1]
        try:
            if process.returncode is None:
                process.wait(timeout=self.termination_grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait()
                termination_result = ProcessTermination(True, False, True, True, "forced_kill", termination_result.warning)
                if lifecycle.current not in {ProcessState.KILLED, ProcessState.CLEANED_UP}:
                    if lifecycle.current == ProcessState.TERMINATING:
                        lifecycle = lifecycle.transition(ProcessState.KILLED)
            except OSError as exc:
                termination_result = ProcessTermination(True, False, True, False, "forced_kill_failed", str(exc))
        cleanup_succeeded = True
        try:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        except OSError:
            cleanup_succeeded = False
        if lifecycle.current != ProcessState.CLEANED_UP:
            lifecycle = lifecycle.transition(ProcessState.CLEANED_UP)
        return self._result(request, relative_cwd, lifecycle, start, process.returncode, stdout, stderr, stdout_truncated, stderr_truncated, termination_result, None, None, timed_out=timed_out, cleanup_succeeded=cleanup_succeeded)

    def _collect(self, process: subprocess.Popen[bytes], request: CommandRequest, lifecycle: ProcessLifecycle) -> tuple[bytes, bytes, bool, bool, bool, tuple[ProcessLifecycle, ProcessTermination]]:
        assert process.stdout is not None and process.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        counts = {"stdout": 0, "stderr": 0}
        truncated = {"stdout": False, "stderr": False}
        deadline = time.monotonic() + request.timeout_seconds
        termination_result = ProcessTermination(False, False, False, True, "normal")
        timed_out = False
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    lifecycle = lifecycle.transition(ProcessState.TIMED_OUT).transition(ProcessState.TERMINATING)
                    termination_result = self._terminate(process, "timeout")
                    break
                events = selector.select(remaining)
                if not events:
                    timed_out = True
                    lifecycle = lifecycle.transition(ProcessState.TIMED_OUT).transition(ProcessState.TERMINATING)
                    termination_result = self._terminate(process, "timeout")
                    break
                for key, _ in events:
                    chunk = os.read(key.fileobj.fileno(), 65_536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    stream = key.data
                    counts[stream] += len(chunk)
                    available = (request.max_stdout_bytes if stream == "stdout" else request.max_stderr_bytes) - len(buffers[stream])
                    if available > 0:
                        buffers[stream].extend(chunk[:available])
                    if len(chunk) > max(available, 0):
                        truncated[stream] = True
                        if lifecycle.current == ProcessState.RUNNING:
                            lifecycle = lifecycle.transition(ProcessState.OUTPUT_LIMIT_REACHED)
                if process.poll() is not None and not selector.get_map():
                    break
            if timed_out:
                drain_deadline = time.monotonic() + self.termination_grace_seconds
                while selector.get_map() and time.monotonic() < drain_deadline:
                    events = selector.select(max(0.0, drain_deadline - time.monotonic()))
                    if not events:
                        break
                    for key, _ in events:
                        chunk = os.read(key.fileobj.fileno(), 65_536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        stream = key.data
                        counts[stream] += len(chunk)
                        available = (request.max_stdout_bytes if stream == "stdout" else request.max_stderr_bytes) - len(buffers[stream])
                        if available > 0:
                            buffers[stream].extend(chunk[:available])
                        if len(chunk) > max(available, 0):
                            truncated[stream] = True
            if lifecycle.current == ProcessState.TERMINATING:
                lifecycle = lifecycle.transition(ProcessState.TERMINATED if termination_result.succeeded else ProcessState.KILLED)
            elif lifecycle.current in {ProcessState.RUNNING, ProcessState.OUTPUT_LIMIT_REACHED}:
                lifecycle = lifecycle.transition(ProcessState.COMPLETED)
            return bytes(buffers["stdout"]), bytes(buffers["stderr"]), truncated["stdout"], truncated["stderr"], timed_out, (lifecycle, termination_result)
        finally:
            selector.close()

    def _terminate(self, process: subprocess.Popen[bytes], reason: str) -> ProcessTermination:
        if process.poll() is not None:
            return ProcessTermination(True, False, False, True, "already_exited")
        try:
            if os.name == "posix" and process.pid:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=self.termination_grace_seconds)
                return ProcessTermination(True, True, False, True, f"{reason}_graceful")
            except subprocess.TimeoutExpired:
                pass
        except (OSError, ProcessLookupError) as exc:
            warning = str(exc)
        else:
            warning = None
        try:
            if os.name == "posix" and process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=self.termination_grace_seconds)
            return ProcessTermination(True, False, True, True, f"{reason}_forced", warning)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired) as exc:
            return ProcessTermination(True, False, True, False, f"{reason}_forced_failed", str(exc))

    def _result(
        self,
        request: CommandRequest,
        relative_cwd: str,
        lifecycle: ProcessLifecycle,
        start: float,
        exit_code: int | None,
        stdout_raw: bytes,
        stderr_raw: bytes,
        stdout_truncated: bool,
        stderr_truncated: bool,
        termination: ProcessTermination,
        error_message: str | None,
        error_code: ToolErrorCode | None,
        *,
        timed_out: bool = False,
        cleanup_succeeded: bool = True,
    ) -> CommandResult:
        stdout, stdout_valid = _decode_output(stdout_raw)
        stderr, stderr_valid = _decode_output(stderr_raw)
        warnings: list[str] = []
        if stdout_truncated or stderr_truncated:
            warnings.append("Output exceeded a configured limit; excess bytes were drained and discarded.")
        if not stdout_valid:
            warnings.append("stdout contained invalid UTF-8 and replacement decoding was used.")
        if not stderr_valid:
            warnings.append("stderr contained invalid UTF-8 and replacement decoding was used.")
        if not cleanup_succeeded:
            warnings.append("Process pipe cleanup encountered an operating-system error.")
            error_code = error_code or ToolErrorCode.PROCESS_CLEANUP_FAILED
        if timed_out:
            error_code = error_code or ToolErrorCode.PROCESS_TIMEOUT
            error_message = error_message or "The process exceeded the bounded timeout."
        elif termination.attempted and not termination.succeeded:
            error_code = error_code or (ToolErrorCode.PROCESS_KILL_FAILED if termination.killed else ToolErrorCode.PROCESS_TERMINATION_FAILED)
            error_message = error_message or "Process termination did not complete successfully."
        elif stdout_truncated or stderr_truncated:
            error_code = error_code or ToolErrorCode.PROCESS_OUTPUT_LIMIT
            error_message = error_message or "Process output exceeded a configured limit."
        elif exit_code not in (None, 0) and error_code is None:
            error_code = ToolErrorCode.PROCESS_NONZERO_EXIT
            error_message = "The process exited with a non-zero status."
        started = ProcessState.RUNNING in lifecycle.history
        completed = started and lifecycle.current == ProcessState.CLEANED_UP and not timed_out and termination.reason not in {"start_failed"}
        succeeded = started and completed and exit_code == 0 and not timed_out and termination.succeeded and not stdout_truncated and not stderr_truncated
        return CommandResult(
            request.argv,
            relative_cwd,
            "timed_out" if timed_out else ("failed" if not succeeded else "completed"),
            exit_code,
            stdout,
            stderr,
            _duration(start),
            timed_out,
            started,
            completed,
            succeeded,
            not started,
            stdout_truncated,
            stderr_truncated,
            stdout_valid,
            stderr_valid,
            termination.reason,
            error_code.value if error_code else None,
            error_message,
            tuple(warnings),
            lifecycle.current.value,
            tuple(state.value for state in lifecycle.history),
            termination.attempted,
            termination.killed,
            len(stdout_raw),
            len(stderr_raw),
        )


__all__ = ["DEFAULT_TERMINATION_GRACE_SECONDS", "ProcessLifecycle", "ProcessManager", "ProcessState", "ProcessTermination"]
