"""Bounded, read-only inspection of an explicit Git working tree."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import selectors
import subprocess
import time
from typing import Any, Mapping

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.filesystem import _validate_root

DEFAULT_MAX_DIFF_BYTES = 1_048_576
DEFAULT_MAX_DIFF_LINES = 10_000
DEFAULT_MAX_CHANGED_FILES = 2_000
DEFAULT_MAX_COMMAND_OUTPUT_BYTES = 2_097_152
DEFAULT_GIT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class GitChangedFile:
    """Structured metadata for one repository-relative changed path."""

    relative_path: str
    status: str
    staged: bool
    unstaged: bool
    untracked: bool
    old_path: str | None
    is_binary: bool
    insertions: int
    deletions: int
    staged_insertions: int = 0
    staged_deletions: int = 0
    unstaged_insertions: int = 0
    unstaged_deletions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "status": self.status,
            "staged": self.staged,
            "unstaged": self.unstaged,
            "untracked": self.untracked,
            "old_path": self.old_path,
            "is_binary": self.is_binary,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "staged_insertions": self.staged_insertions,
            "staged_deletions": self.staged_deletions,
            "unstaged_insertions": self.unstaged_insertions,
            "unstaged_deletions": self.unstaged_deletions,
        }


@dataclass(frozen=True, slots=True)
class GitDiffResult:
    """Immutable bounded read-only Git inspection result."""

    root: Path
    is_git_repository: bool
    current_branch: str | None
    head: str | None
    changed_files: tuple[GitChangedFile, ...]
    staged_diff: str
    unstaged_diff: str
    combined_diff: str
    files_changed: int
    insertions: int
    deletions: int
    truncated: bool
    truncation_reason: str | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "is_git_repository": self.is_git_repository,
            "current_branch": self.current_branch,
            "head": self.head,
            "changed_files": [item.to_dict() for item in self.changed_files],
            "staged_diff": self.staged_diff,
            "unstaged_diff": self.unstaged_diff,
            "combined_diff": self.combined_diff,
            "files_changed": self.files_changed,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _CommandOutput:
    stdout: bytes
    stderr: bytes
    returncode: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class _StatusEntry:
    relative_path: str
    status: str
    staged: bool
    unstaged: bool
    untracked: bool
    old_path: str | None


@dataclass(frozen=True, slots=True)
class _NumStat:
    insertions: int
    deletions: int
    is_binary: bool


class GitReadOnlyAdapter:
    """Whitelist-only adapter for bounded read-only Git commands."""

    _COMMANDS: dict[str, tuple[str, ...]] = {
        "repo_root": ("rev-parse", "--show-toplevel"),
        "head": ("rev-parse", "--verify", "HEAD"),
        "branch": ("branch", "--show-current"),
        "working_tree": ("status", "--porcelain=v1", "-z", "--untracked-files=all", "--renames"),
        "unstaged_diff": ("diff", "--no-ext-diff", "--no-textconv", "--no-color", "--no-renames", "--unified=3"),
        "staged_diff": ("diff", "--cached", "--no-ext-diff", "--no-textconv", "--no-color", "--no-renames", "--unified=3"),
        "unstaged_numstat": ("diff", "--numstat", "-z", "--no-ext-diff", "--no-textconv", "--no-color", "--no-renames"),
        "staged_numstat": ("diff", "--cached", "--numstat", "-z", "--no-ext-diff", "--no-textconv", "--no-color", "--no-renames"),
    }

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_COMMAND_OUTPUT_BYTES,
    ) -> None:
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(max_output_bytes, int) or isinstance(max_output_bytes, bool) or max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = max_output_bytes

    def run(self, operation: str, root: Path) -> _CommandOutput:
        """Run one exact whitelisted Git argv without a shell or mutation command."""

        try:
            arguments = self._COMMANDS[operation]
        except KeyError as exc:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, f"Git operation is not whitelisted: {operation}") from exc
        environment = os.environ.copy()
        environment.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C", "LANG": "C"})
        try:
            process = subprocess.Popen(
                ("git", *arguments),
                cwd=str(root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise ToolError(ToolErrorCode.GIT_NOT_AVAILABLE, "Git executable is not available.") from exc
        except PermissionError as exc:
            raise ToolError(ToolErrorCode.PERMISSION_DENIED, "Permission denied while starting Git.") from exc
        except OSError as exc:
            raise ToolError(ToolErrorCode.GIT_COMMAND_FAILED, "Unable to start the read-only Git adapter.") from exc

        stdout, stderr, truncated = _bounded_process_output(
            process,
            self.timeout_seconds,
            self.max_output_bytes,
        )
        return _CommandOutput(stdout, stderr, process.returncode or 0, truncated)


def git_diff(
    project_root: Path | str,
    *,
    max_diff_bytes: int = DEFAULT_MAX_DIFF_BYTES,
    max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
    max_changed_files: int = DEFAULT_MAX_CHANGED_FILES,
    max_command_output_bytes: int = DEFAULT_MAX_COMMAND_OUTPUT_BYTES,
    timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
) -> GitDiffResult:
    """Inspect staged/unstaged Git changes without modifying files or Git state."""

    root = _validate_root(project_root)
    _validate_limits(max_diff_bytes, max_diff_lines, max_changed_files, max_command_output_bytes, timeout_seconds)
    adapter = GitReadOnlyAdapter(timeout_seconds=timeout_seconds, max_output_bytes=max_command_output_bytes)
    repo_output = adapter.run("repo_root", root)
    if repo_output.returncode != 0:
        warning = "The explicit project_root is not a Git repository root."
        return _empty_result(root, warning)
    try:
        detected_root = _decode_text(repo_output.stdout).strip().replace("\\", "/")
    except UnicodeDecodeError as exc:
        raise ToolError(ToolErrorCode.GIT_COMMAND_FAILED, "Git returned an invalid repository root.") from exc
    if not detected_root or Path(detected_root).resolve() != root:
        return _empty_result(root, "The explicit project_root is not the Git repository root.")
    if repo_output.truncated:
        return _empty_result(root, "Git repository-root output exceeded the command limit.", truncated=True, reason="command_output_limit")

    branch_output = adapter.run("branch", root)
    head_output = adapter.run("head", root)
    status_output = adapter.run("working_tree", root)
    for operation, output in (("branch", branch_output), ("status", status_output)):
        if output.returncode != 0:
            raise ToolError(ToolErrorCode.GIT_COMMAND_FAILED, f"Read-only Git {operation} inspection failed.")
    branch = _decode_text(branch_output.stdout).strip() or None
    head = _decode_text(head_output.stdout).strip() if head_output.returncode == 0 else None
    warnings: list[str] = []
    if head_output.returncode != 0:
        warnings.append("HEAD is unavailable; the repository may have an unborn or detached state.")

    entries, status_warnings = _parse_status(status_output.stdout)
    warnings.extend(status_warnings)
    truncated = status_output.truncated
    truncation_reasons: list[str] = ["command_output_limit"] if status_output.truncated else []
    if len(entries) > max_changed_files:
        entries = entries[:max_changed_files]
        truncated = True
        truncation_reasons.append("max_changed_files")

    unstaged_diff_output = adapter.run("unstaged_diff", root)
    staged_diff_output = adapter.run("staged_diff", root)
    unstaged_numstat_output = adapter.run("unstaged_numstat", root)
    staged_numstat_output = adapter.run("staged_numstat", root)
    for operation, output in (
        ("unstaged diff", unstaged_diff_output),
        ("staged diff", staged_diff_output),
        ("unstaged numstat", unstaged_numstat_output),
        ("staged numstat", staged_numstat_output),
    ):
        if output.returncode != 0:
            raise ToolError(ToolErrorCode.GIT_COMMAND_FAILED, f"Read-only Git {operation} inspection failed.")
        if output.truncated:
            truncated = True
            truncation_reasons.append("command_output_limit")

    unstaged_diff, unstaged_truncated = _bound_text(_decode_text(unstaged_diff_output.stdout), max_diff_bytes, max_diff_lines)
    staged_diff, staged_truncated = _bound_text(_decode_text(staged_diff_output.stdout), max_diff_bytes, max_diff_lines)
    if unstaged_truncated:
        truncated = True
        truncation_reasons.append("max_diff_bytes_or_lines")
    if staged_truncated:
        truncated = True
        truncation_reasons.append("max_diff_bytes_or_lines")
    combined_diff, combined_truncated = _bound_text(
        _join_diffs(staged_diff, unstaged_diff),
        max_diff_bytes,
        max_diff_lines,
    )
    if combined_truncated:
        truncated = True
        truncation_reasons.append("max_diff_bytes_or_lines")

    staged_stats = _parse_numstat(staged_numstat_output.stdout)
    unstaged_stats = _parse_numstat(unstaged_numstat_output.stdout)
    changed_files = tuple(
        _merge_entries(entries, staged_stats, unstaged_stats)
    )
    insertions = sum(item.insertions for item in changed_files)
    deletions = sum(item.deletions for item in changed_files)
    return GitDiffResult(
        root=root,
        is_git_repository=True,
        current_branch=branch,
        head=head,
        changed_files=changed_files,
        staged_diff=staged_diff,
        unstaged_diff=unstaged_diff,
        combined_diff=combined_diff,
        files_changed=len(changed_files),
        insertions=insertions,
        deletions=deletions,
        truncated=truncated,
        truncation_reason=";".join(dict.fromkeys(truncation_reasons)) if truncation_reasons else None,
        warnings=tuple(warnings),
    )


class GitDiffTool:
    """Tool-protocol wrapper for read-only explicit Git diff inspection."""

    name = "git_diff"
    description = (
        "Inspect staged, unstaged, and untracked changes in an explicit Git repository. "
        "Read-only, bounded, deterministic, and never a generic command executor."
    )
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root"],
            "properties": {
                "project_root": {"type": "string", "description": "Explicit Git repository directory."},
                "max_diff_bytes": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_DIFF_BYTES},
                "max_diff_lines": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_DIFF_LINES},
                "max_changed_files": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_CHANGED_FILES},
                "max_command_output_bytes": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_COMMAND_OUTPUT_BYTES},
                "timeout_seconds": {"type": "number", "exclusiveMinimum": 0, "default": DEFAULT_GIT_TIMEOUT_SECONDS},
            },
        },
    )

    def run(self, arguments: Mapping[str, Any]) -> GitDiffResult:
        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "git_diff arguments must be a mapping.")
        if "project_root" not in arguments:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "git_diff requires: 'project_root'.")
        return git_diff(
            arguments["project_root"],
            max_diff_bytes=arguments.get("max_diff_bytes", DEFAULT_MAX_DIFF_BYTES),
            max_diff_lines=arguments.get("max_diff_lines", DEFAULT_MAX_DIFF_LINES),
            max_changed_files=arguments.get("max_changed_files", DEFAULT_MAX_CHANGED_FILES),
            max_command_output_bytes=arguments.get("max_command_output_bytes", DEFAULT_MAX_COMMAND_OUTPUT_BYTES),
            timeout_seconds=arguments.get("timeout_seconds", DEFAULT_GIT_TIMEOUT_SECONDS),
        )


def _empty_result(
    root: Path,
    warning: str,
    *,
    truncated: bool = False,
    reason: str | None = None,
) -> GitDiffResult:
    return GitDiffResult(root, False, None, None, (), "", "", "", 0, 0, 0, truncated, reason, (warning,))


def _validate_limits(
    max_diff_bytes: int,
    max_diff_lines: int,
    max_changed_files: int,
    max_command_output_bytes: int,
    timeout_seconds: float,
) -> None:
    for name, value in (
        ("max_diff_bytes", max_diff_bytes),
        ("max_diff_lines", max_diff_lines),
        ("max_changed_files", max_changed_files),
        ("max_command_output_bytes", max_command_output_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, f"{name} must be a positive integer.")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "timeout_seconds must be positive.")


def _bounded_process_output(process: subprocess.Popen[bytes], timeout_seconds: float, max_output_bytes: int) -> tuple[bytes, bytes, bool]:
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = False
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                process.wait()
                raise ToolError(ToolErrorCode.GIT_TIMEOUT, "Read-only Git command timed out.")
            events = selector.select(remaining)
            if not events:
                process.kill()
                process.wait()
                raise ToolError(ToolErrorCode.GIT_TIMEOUT, "Read-only Git command timed out.")
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer = buffers[key.data]
                available = max_output_bytes - len(buffer)
                if len(chunk) > available:
                    if available > 0:
                        buffer.extend(chunk[:available])
                    truncated = True
                    process.kill()
                    break
                buffer.extend(chunk)
            if truncated:
                break
        if truncated:
            process.wait()
            return bytes(buffers["stdout"]), bytes(buffers["stderr"]), True
        process.wait()
        return bytes(buffers["stdout"]), bytes(buffers["stderr"]), False
    finally:
        selector.close()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()


def _decode_text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _decode_path(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="surrogateescape").replace("\\", "/")
    try:
        normalized = PurePosixPath(text)
    except Exception as exc:
        raise ToolError(ToolErrorCode.GIT_COMMAND_FAILED, "Git returned an invalid path.") from exc
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ToolError(ToolErrorCode.GIT_COMMAND_FAILED, "Git returned a path outside the repository.")
    return normalized.as_posix()


def _parse_status(raw: bytes) -> tuple[list[_StatusEntry], list[str]]:
    fields = raw.split(b"\0")
    entries: list[_StatusEntry] = []
    warnings: list[str] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        if len(field) < 3:
            warnings.append("Git returned a malformed status record.")
            continue
        xy = field[:2].decode("ascii", errors="replace")
        path = _decode_path(field[3:])
        old_path: str | None = None
        if xy[0] in "RC" or xy[1] in "RC":
            if index < len(fields) and fields[index]:
                old_path = _decode_path(fields[index])
                index += 1
        x, y = xy
        untracked = x == "?" and y == "?"
        staged = x not in (" ", "?")
        unstaged = y not in (" ", "?")
        if untracked:
            status = "untracked"
        elif "R" in xy:
            status = "renamed"
        elif "C" in xy:
            status = "copied"
        elif "D" in xy:
            status = "deleted"
        elif "A" in xy:
            status = "added"
        else:
            status = "modified"
        entries.append(_StatusEntry(path, status, staged, unstaged, untracked, old_path))
    entries.sort(key=lambda item: (item.relative_path.casefold(), item.relative_path))
    return entries, warnings


def _parse_numstat(raw: bytes) -> dict[str, _NumStat]:
    result: dict[str, _NumStat] = {}
    for field in raw.split(b"\0"):
        if not field:
            continue
        pieces = field.split(b"\t", 2)
        if len(pieces) != 3:
            continue
        insertions_raw, deletions_raw, path_raw = pieces
        path = _decode_path(path_raw)
        binary = insertions_raw == b"-" or deletions_raw == b"-"
        insertions = 0 if binary else _safe_int(insertions_raw)
        deletions = 0 if binary else _safe_int(deletions_raw)
        result[path] = _NumStat(insertions, deletions, binary)
    return result


def _safe_int(raw: bytes) -> int:
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _merge_entries(entries: list[_StatusEntry], staged: dict[str, _NumStat], unstaged: dict[str, _NumStat]) -> tuple[GitChangedFile, ...]:
    merged: list[GitChangedFile] = []
    for entry in entries:
        staged_stat = staged.get(entry.relative_path, _NumStat(0, 0, False)) if entry.staged else _NumStat(0, 0, False)
        unstaged_stat = unstaged.get(entry.relative_path, _NumStat(0, 0, False)) if entry.unstaged else _NumStat(0, 0, False)
        merged.append(
            GitChangedFile(
                relative_path=entry.relative_path,
                status=entry.status,
                staged=entry.staged,
                unstaged=entry.unstaged,
                untracked=entry.untracked,
                old_path=entry.old_path,
                is_binary=staged_stat.is_binary or unstaged_stat.is_binary,
                insertions=staged_stat.insertions + unstaged_stat.insertions,
                deletions=staged_stat.deletions + unstaged_stat.deletions,
                staged_insertions=staged_stat.insertions,
                staged_deletions=staged_stat.deletions,
                unstaged_insertions=unstaged_stat.insertions,
                unstaged_deletions=unstaged_stat.deletions,
            )
        )
    return tuple(merged)


def _bound_text(text: str, max_bytes: int, max_lines: int) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines and len(text.encode("utf-8")) <= max_bytes:
        return text, False
    kept: list[str] = []
    total = 0
    marker = "\n[git diff truncated]\n"
    marker_bytes = marker.encode("utf-8")
    budget = max(0, max_bytes - len(marker_bytes))
    for line in lines[:max_lines]:
        size = len(line.encode("utf-8"))
        if total + size > budget:
            break
        kept.append(line)
        total += size
    if max_bytes < len(marker_bytes):
        return marker_bytes[:max_bytes].decode("ascii"), True
    return "".join(kept) + marker, True


def _join_diffs(staged: str, unstaged: str) -> str:
    if staged and unstaged:
        return staged.rstrip("\n") + "\n" + unstaged
    return staged or unstaged


__all__ = [
    "DEFAULT_GIT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_CHANGED_FILES",
    "DEFAULT_MAX_COMMAND_OUTPUT_BYTES",
    "DEFAULT_MAX_DIFF_BYTES",
    "DEFAULT_MAX_DIFF_LINES",
    "GitChangedFile",
    "GitDiffResult",
    "GitDiffTool",
    "GitReadOnlyAdapter",
    "git_diff",
]
