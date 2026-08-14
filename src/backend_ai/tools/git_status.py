"""Bounded, read-only Git working-tree status inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.filesystem import _validate_root
from backend_ai.tools.git_diff import (
    DEFAULT_GIT_TIMEOUT_SECONDS,
    DEFAULT_MAX_COMMAND_OUTPUT_BYTES,
    GitReadOnlyAdapter,
    _decode_path,
    _decode_text,
)

DEFAULT_MAX_STATUS_FILES = 2_000
DEFAULT_MAX_STATUS_PATH_LENGTH = 4_096


@dataclass(frozen=True, slots=True)
class GitStatusFile:
    """Immutable metadata for one repository-relative status entry."""

    relative_path: str
    status: str
    index_status: str
    worktree_status: str
    old_path: str | None
    new_path: str | None
    is_untracked: bool
    is_ignored: bool
    is_conflicted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "status": self.status,
            "index_status": self.index_status,
            "worktree_status": self.worktree_status,
            "old_path": self.old_path,
            "new_path": self.new_path,
            "is_untracked": self.is_untracked,
            "is_ignored": self.is_ignored,
            "is_conflicted": self.is_conflicted,
        }


@dataclass(frozen=True, slots=True)
class GitStatusResult:
    """Immutable bounded read-only Git status result."""

    root: Path
    is_git_repository: bool
    branch: str | None
    head: str | None
    head_state: str
    upstream: str | None
    ahead: int | None
    behind: int | None
    is_clean: bool
    files: tuple[GitStatusFile, ...]
    staged: tuple[GitStatusFile, ...]
    unstaged: tuple[GitStatusFile, ...]
    untracked: tuple[GitStatusFile, ...]
    ignored: tuple[GitStatusFile, ...]
    conflicts: tuple[GitStatusFile, ...]
    renamed: tuple[GitStatusFile, ...]
    deleted: tuple[GitStatusFile, ...]
    modified: tuple[GitStatusFile, ...]
    added: tuple[GitStatusFile, ...]
    warnings: tuple[str, ...]
    truncated: bool
    truncation_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "is_git_repository": self.is_git_repository,
            "branch": self.branch,
            "head": self.head,
            "head_state": self.head_state,
            "upstream": self.upstream,
            "ahead": self.ahead,
            "behind": self.behind,
            "is_clean": self.is_clean,
            "files": [item.to_dict() for item in self.files],
            "staged": [item.to_dict() for item in self.staged],
            "unstaged": [item.to_dict() for item in self.unstaged],
            "untracked": [item.to_dict() for item in self.untracked],
            "ignored": [item.to_dict() for item in self.ignored],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "renamed": [item.to_dict() for item in self.renamed],
            "deleted": [item.to_dict() for item in self.deleted],
            "modified": [item.to_dict() for item in self.modified],
            "added": [item.to_dict() for item in self.added],
            "warnings": list(self.warnings),
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }


@dataclass(frozen=True, slots=True)
class _BranchInfo:
    branch: str | None
    head: str | None
    upstream: str | None
    ahead: int | None
    behind: int | None


class GitStatusTool:
    """Tool-protocol wrapper for explicit, read-only Git status inspection."""

    name = "git_status"
    description = (
        "Inspect the current Git working-tree status for an explicit repository. "
        "Reports staged, unstaged, untracked, ignored, conflict, branch, and upstream "
        "metadata without mutating Git or invoking arbitrary commands."
    )
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root"],
            "properties": {
                "project_root": {"type": "string", "description": "Explicit Git repository directory."},
                "include_ignored": {"type": "boolean", "default": False},
                "max_files": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_STATUS_FILES},
                "max_output_bytes": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_COMMAND_OUTPUT_BYTES},
                "max_path_length": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_STATUS_PATH_LENGTH},
                "timeout_seconds": {"type": "number", "exclusiveMinimum": 0, "default": DEFAULT_GIT_TIMEOUT_SECONDS},
            },
        },
    )

    def run(self, arguments: Mapping[str, Any]) -> GitStatusResult:
        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "git_status arguments must be a mapping.")
        if "project_root" not in arguments:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "git_status requires: 'project_root'.")
        return git_status(
            arguments["project_root"],
            include_ignored=arguments.get("include_ignored", False),
            max_files=arguments.get("max_files", DEFAULT_MAX_STATUS_FILES),
            max_output_bytes=arguments.get("max_output_bytes", DEFAULT_MAX_COMMAND_OUTPUT_BYTES),
            max_path_length=arguments.get("max_path_length", DEFAULT_MAX_STATUS_PATH_LENGTH),
            timeout_seconds=arguments.get("timeout_seconds", DEFAULT_GIT_TIMEOUT_SECONDS),
        )


def git_status(
    project_root: Path | str,
    *,
    include_ignored: bool = False,
    max_files: int = DEFAULT_MAX_STATUS_FILES,
    max_output_bytes: int = DEFAULT_MAX_COMMAND_OUTPUT_BYTES,
    max_path_length: int = DEFAULT_MAX_STATUS_PATH_LENGTH,
    timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
) -> GitStatusResult:
    """Return deterministic structured status for the explicit Git root."""

    root = _validate_root(project_root)
    _validate_limits(include_ignored, max_files, max_output_bytes, max_path_length, timeout_seconds)
    adapter = GitReadOnlyAdapter(timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)
    repo_output = adapter.run("repo_root", root)
    if repo_output.returncode != 0:
        return _empty_status(root, "The explicit project_root is not a Git repository root.")
    detected_root = _decode_text(repo_output.stdout).strip().replace("\\", "/")
    if not detected_root or Path(detected_root).resolve() != root:
        return _empty_status(root, "The explicit project_root is not the Git repository root.")
    if repo_output.truncated:
        return _empty_status(root, "Git repository-root output exceeded the command limit.", True, "command_output_limit")

    operation = "status_branch_ignored" if include_ignored else "status_branch"
    status_output = adapter.run(operation, root)
    if status_output.returncode != 0:
        raise ToolError(ToolErrorCode.GIT_COMMAND_FAILED, "Read-only Git status inspection failed.")
    head_output = adapter.run("head", root)
    entries, branch_info, warnings = _parse_porcelain_status(status_output.stdout, max_path_length)
    head = _decode_text(head_output.stdout).strip() if head_output.returncode == 0 else None
    branch_info = _BranchInfo(branch_info.branch, head, branch_info.upstream, branch_info.ahead, branch_info.behind)
    if head_output.returncode != 0 and branch_info.branch is not None:
        warnings.append("HEAD is unavailable; the repository may have an unborn state.")
    truncated = status_output.truncated
    reasons: list[str] = ["command_output_limit"] if status_output.truncated else []
    if any("max_path_length" in warning for warning in warnings):
        truncated = True
        reasons.append("max_path_length")
    if len(entries) > max_files:
        entries = entries[:max_files]
        truncated = True
        reasons.append("max_files")
    files = tuple(entries)
    ignored = tuple(item for item in files if item.is_ignored)
    tracked = tuple(item for item in files if not item.is_ignored)
    staged = tuple(item for item in tracked if item.index_status not in (" ", "?"))
    unstaged = tuple(item for item in tracked if item.worktree_status not in (" ", "?"))
    untracked = tuple(item for item in files if item.is_untracked)
    conflicts = tuple(item for item in tracked if item.is_conflicted)
    renamed = tuple(item for item in tracked if item.status == "renamed")
    deleted = tuple(item for item in tracked if item.status == "deleted")
    modified = tuple(item for item in tracked if item.status == "modified")
    added = tuple(item for item in tracked if item.status == "added")
    return GitStatusResult(
        root=root,
        is_git_repository=True,
        branch=branch_info.branch,
        head=branch_info.head,
        head_state=_head_state(branch_info),
        upstream=branch_info.upstream,
        ahead=branch_info.ahead,
        behind=branch_info.behind,
        is_clean=not tracked and not untracked,
        files=files,
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
        ignored=ignored,
        conflicts=conflicts,
        renamed=renamed,
        deleted=deleted,
        modified=modified,
        added=added,
        warnings=tuple(warnings),
        truncated=truncated,
        truncation_reason=";".join(dict.fromkeys(reasons)) if reasons else None,
    )


def _empty_status(root: Path, warning: str, truncated: bool = False, reason: str | None = None) -> GitStatusResult:
    return GitStatusResult(
        root=root,
        is_git_repository=False,
        branch=None,
        head=None,
        head_state="unknown",
        upstream=None,
        ahead=None,
        behind=None,
        is_clean=True,
        files=(),
        staged=(),
        unstaged=(),
        untracked=(),
        ignored=(),
        conflicts=(),
        renamed=(),
        deleted=(),
        modified=(),
        added=(),
        warnings=(warning,),
        truncated=truncated,
        truncation_reason=reason,
    )


def _validate_limits(
    include_ignored: bool,
    max_files: int,
    max_output_bytes: int,
    max_path_length: int,
    timeout_seconds: float,
) -> None:
    if not isinstance(include_ignored, bool):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "include_ignored must be boolean.")
    for name, value in (("max_files", max_files), ("max_output_bytes", max_output_bytes), ("max_path_length", max_path_length)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, f"{name} must be a positive integer.")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "timeout_seconds must be positive.")


def _parse_porcelain_status(raw: bytes, max_path_length: int) -> tuple[list[GitStatusFile], _BranchInfo, list[str]]:
    fields = raw.split(b"\0")
    branch = None
    head = None
    upstream = None
    ahead = None
    behind = None
    warnings: list[str] = []
    entries: list[GitStatusFile] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        if field.startswith(b"# "):
            header = _decode_text(field)
            if header.startswith("# branch.head "):
                value = header[len("# branch.head "):]
                branch = None if value in ("", "(detached)") else value
            elif header.startswith("# branch.oid "):
                value = header[len("# branch.oid "):]
                head = None if value in ("", "(initial)") else value
            elif header.startswith("# branch.upstream "):
                upstream = header[len("# branch.upstream "): ] or None
            elif header.startswith("# branch.ab "):
                parts = header[len("# branch.ab "):].split()
                if len(parts) == 2 and parts[0].startswith("+") and parts[1].startswith("-"):
                    ahead = _parse_count(parts[0][1:])
                    behind = _parse_count(parts[1][1:])
                else:
                    warnings.append("Git returned malformed branch ahead/behind metadata.")
            continue
        if field.startswith(b"## "):
            branch_line = _decode_text(field)[3:]
            if branch_line.startswith("No commits yet on "):
                branch = branch_line[len("No commits yet on "):].strip() or None
            elif branch_line.startswith("HEAD (no branch)"):
                branch = None
            else:
                branch_part, separator, tracking_part = branch_line.partition("...")
                branch = branch_part.strip() or None
                if separator:
                    upstream_part = tracking_part.split(" [", 1)[0].strip()
                    upstream = upstream_part or None
                    match = re.search(r"\\[ahead (\\d+)(?:, behind (\\d+))?\\]", tracking_part)
                    if match:
                        ahead = _parse_count(match.group(1))
                        behind = _parse_count(match.group(2) or "0")
                    else:
                        match = re.search(r"\\[behind (\\d+)\\]", tracking_part)
                        if match:
                            behind = _parse_count(match.group(1))
            continue
        if len(field) < 3:
            warnings.append("Git returned a malformed status record.")
            continue
        xy = field[:2].decode("ascii", errors="replace")
        path_bytes = field[3:]
        path = _decode_path(path_bytes)
        if len(path) > max_path_length:
            warnings.append(f"Status path exceeded max_path_length and was omitted: {path[:64]}")
            continue
        old_path: str | None = None
        new_path: str | None = path
        if xy[0] in "RC" or xy[1] in "RC":
            if index < len(fields) and fields[index]:
                old_path = _decode_path(fields[index])
                index += 1
                if len(old_path) > max_path_length:
                    warnings.append("Rename source path exceeded max_path_length and was omitted.")
                    old_path = None
        index_status, worktree_status = xy
        is_untracked = xy == "??"
        is_ignored = xy == "!!"
        is_conflicted = _is_conflict(xy)
        if is_ignored:
            status = "ignored"
        elif is_untracked:
            status = "untracked"
        elif "R" in xy:
            status = "renamed"
        elif "C" in xy:
            status = "copied"
        elif is_conflicted:
            status = "conflicted"
        elif "D" in xy:
            status = "deleted"
        elif "A" in xy:
            status = "added"
        else:
            status = "modified"
        entries.append(GitStatusFile(path, status, index_status, worktree_status, old_path, new_path, is_untracked, is_ignored, is_conflicted))
    entries.sort(key=lambda item: (item.relative_path.casefold(), item.relative_path, item.old_path or ""))
    return entries, _BranchInfo(branch, head, upstream, ahead, behind), warnings


def _is_conflict(xy: str) -> bool:
    return xy in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"} or "U" in xy


def _parse_count(value: str) -> int | None:
    try:
        return max(0, int(value))
    except ValueError:
        return None


def _head_state(info: _BranchInfo) -> str:
    if info.branch is not None:
        return "branch" if info.head is not None else "unborn"
    if info.head is not None:
        return "detached"
    return "unborn" if info.branch is None and info.upstream is None else "unknown"


__all__ = [
    "DEFAULT_MAX_STATUS_FILES",
    "DEFAULT_MAX_STATUS_PATH_LENGTH",
    "GitStatusFile",
    "GitStatusResult",
    "GitStatusTool",
    "git_status",
]
