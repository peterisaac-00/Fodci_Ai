"""Safe, deterministic, bounded source search for the Agent tool layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import stat
from typing import Any

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.filesystem import (
    DEFAULT_IGNORED_DIRECTORIES,
    DEFAULT_MAX_DEPTH,
    _path_sort_key,
    _validate_root,
)
from backend_ai.tools.read_file import _looks_like_windows_absolute

DEFAULT_MAX_RESULTS = 100
MAX_MAX_RESULTS = 10_000
DEFAULT_MAX_FILE_BYTES = 1_048_576
MAX_MAX_FILE_BYTES = 16 * 1_048_576
DEFAULT_MAX_QUERY_LENGTH = 4_096
MAX_SEARCH_DIRECTORIES = 10_000


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """One bounded source-line match with 1-based line and 0-based columns."""

    relative_path: str
    line_number: int
    line: str
    column_start: int
    column_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "line_number": self.line_number,
            "line": self.line,
            "column_start": self.column_start,
            "column_end": self.column_end,
        }


@dataclass(frozen=True, slots=True)
class SearchCodeResult:
    """Structured output from one bounded project search."""

    query: str
    matches: tuple[SearchMatch, ...]
    total_matches: int
    files_searched: int
    files_skipped: int
    skipped_reasons: tuple[str, ...]
    truncated: bool
    truncation_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "matches": [match.to_dict() for match in self.matches],
            "total_matches": self.total_matches,
            "files_searched": self.files_searched,
            "files_skipped": self.files_skipped,
            "skipped_reasons": list(self.skipped_reasons),
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }


class SearchCodeTool:
    """First-class read-only Agent tool for bounded source search."""

    name = "search_code"
    description = (
        "Search regular UTF-8 project files for literal text or an explicitly "
        "enabled regular expression. Returns deterministic line matches and "
        "bounded statistics. Read-only; symlinks and generated directories are skipped."
    )
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root", "query"],
            "properties": {
                "project_root": {"type": "string", "description": "Explicit project directory."},
                "query": {"type": "string", "description": "Literal text or regex pattern."},
                "path": {"type": "string", "description": "Optional root-relative file or directory."},
                "max_results": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_RESULTS},
                "max_file_bytes": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_FILE_BYTES},
                "case_sensitive": {"type": "boolean", "default": True},
                "use_regex": {"type": "boolean", "default": False},
            },
        },
    )

    def run(self, arguments: Mapping[str, Any]) -> SearchCodeResult:
        """Validate a structured request and search the selected scope."""

        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "search_code arguments must be a mapping.")
        missing = [name for name in ("project_root", "query") if name not in arguments]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, f"search_code requires: {names}.")
        return search_code(
            arguments["project_root"],
            arguments["query"],
            path=arguments.get("path"),
            max_results=arguments.get("max_results", DEFAULT_MAX_RESULTS),
            max_file_bytes=arguments.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES),
            case_sensitive=arguments.get("case_sensitive", True),
            use_regex=arguments.get("use_regex", False),
        )


def search_code(
    project_root: Path | str,
    query: str,
    *,
    path: Path | str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    case_sensitive: bool = True,
    use_regex: bool = False,
) -> SearchCodeResult:
    """Search bounded regular UTF-8 files under an explicit project root."""

    root = _validate_root(project_root)
    _validate_search_options(query, max_results, max_file_bytes, case_sensitive, use_regex)
    pattern = _compile_pattern(query, case_sensitive, use_regex)
    scope = _resolve_scope(root, path)
    traversal = _SearchTraversalState()
    files = _iter_regular_files(root, scope, traversal)

    matches: list[SearchMatch] = []
    skipped_reasons: set[str] = set()
    files_searched = 0
    files_skipped = 0
    truncated = False
    truncation_reason: str | None = None

    for file_path, relative_path in files:
        if len(matches) >= max_results:
            truncated = True
            truncation_reason = "max_results"
            break
        try:
            metadata = file_path.stat()
            if metadata.st_size > max_file_bytes:
                files_skipped += 1
                skipped_reasons.add("max_file_bytes")
                truncated = True
                truncation_reason = truncation_reason or "max_file_bytes"
                continue
            with file_path.open("rb") as stream:
                raw = stream.read(max_file_bytes + 1)
            if len(raw) > max_file_bytes:
                files_skipped += 1
                skipped_reasons.add("max_file_bytes")
                truncated = True
                truncation_reason = truncation_reason or "max_file_bytes"
                continue
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            files_skipped += 1
            skipped_reasons.add("invalid_utf8")
            continue
        except PermissionError:
            files_skipped += 1
            skipped_reasons.add("permission_denied")
            continue
        except FileNotFoundError:
            files_skipped += 1
            skipped_reasons.add("path_not_found")
            continue
        except OSError:
            files_skipped += 1
            skipped_reasons.add("filesystem_error")
            continue

        files_searched += 1
        for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
            source_line = line.rstrip("\r\n")
            for match in pattern.finditer(source_line):
                matches.append(
                    SearchMatch(
                        relative_path=relative_path,
                        line_number=line_number,
                        line=source_line,
                        column_start=match.start(),
                        column_end=match.end(),
                    )
                )
                if len(matches) >= max_results:
                    truncated = True
                    truncation_reason = "max_results"
                    break
            if truncation_reason == "max_results":
                break
        if truncation_reason == "max_results":
            break

    skipped_reasons.update(traversal.skipped_reasons)
    if not truncated and traversal.truncated:
        truncated = True
        truncation_reason = traversal.truncation_reason

    return SearchCodeResult(
        query=query,
        matches=tuple(matches),
        total_matches=len(matches),
        files_searched=files_searched,
        files_skipped=files_skipped,
        skipped_reasons=tuple(sorted(skipped_reasons)),
        truncated=truncated,
        truncation_reason=truncation_reason,
    )


def _validate_search_options(
    query: str,
    max_results: int,
    max_file_bytes: int,
    case_sensitive: bool,
    use_regex: bool,
) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "query must be a non-empty string.")
    if len(query) > DEFAULT_MAX_QUERY_LENGTH:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            f"query exceeds the maximum length of {DEFAULT_MAX_QUERY_LENGTH} characters.",
        )
    if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results <= 0:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "max_results must be a positive integer.")
    if max_results > MAX_MAX_RESULTS:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            f"max_results cannot exceed {MAX_MAX_RESULTS}.",
        )
    if not isinstance(max_file_bytes, int) or isinstance(max_file_bytes, bool) or max_file_bytes < 0:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "max_file_bytes must be a non-negative integer.")
    if max_file_bytes > MAX_MAX_FILE_BYTES:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            f"max_file_bytes cannot exceed {MAX_MAX_FILE_BYTES}.",
        )
    if not isinstance(case_sensitive, bool):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "case_sensitive must be a boolean.")
    if not isinstance(use_regex, bool):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "use_regex must be a boolean.")


def _compile_pattern(query: str, case_sensitive: bool, use_regex: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    expression = query if use_regex else re.escape(query)
    try:
        return re.compile(expression, flags)
    except re.error as exc:
        raise ToolError(
            ToolErrorCode.INVALID_REGEX,
            "query is not a valid regular expression.",
        ) from exc


def _resolve_scope(root: Path, requested_path: Path | str | None) -> Path:
    if requested_path is None:
        return root
    if not isinstance(requested_path, (Path, str)):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "path must be a string or pathlib.Path.")
    raw = str(requested_path)
    if not raw.strip() or "\x00" in raw:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "path must be a non-empty valid path.")
    normalized = raw.replace("\\", "/")
    if _looks_like_windows_absolute(normalized):
        candidate = Path(normalized)
        if not candidate.is_absolute():
            raise ToolError(
                ToolErrorCode.PATH_OUTSIDE_ROOT,
                "Absolute Windows and UNC paths cannot bypass the project root.",
                path=candidate,
            )
    else:
        candidate = Path(normalized)
    lexical = candidate if candidate.is_absolute() else root / candidate
    lexical = Path(os.path.normpath(str(lexical)))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ToolError(
            ToolErrorCode.PATH_OUTSIDE_ROOT,
            "Search path is outside the project root.",
            path=lexical,
        ) from exc
    if not relative.parts:
        return root
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ToolError(
                ToolErrorCode.PATH_OUTSIDE_ROOT,
                "Search path contains a symbolic link.",
                path=relative,
            )
    try:
        resolved = lexical.resolve(strict=False)
        resolved.relative_to(root)
    except ValueError as exc:
        raise ToolError(
            ToolErrorCode.PATH_OUTSIDE_ROOT,
            "Search path resolves outside the project root.",
            path=lexical,
        ) from exc
    except PermissionError as exc:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            "Permission denied while resolving search path.",
            path=lexical,
        ) from exc
    except OSError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to resolve search path: {exc}",
            path=lexical,
        ) from exc
    if not lexical.exists():
        raise ToolError(ToolErrorCode.PATH_NOT_FOUND, "Search path does not exist.", path=relative)
    if not lexical.is_dir() and not lexical.is_file():
        raise ToolError(ToolErrorCode.NOT_A_FILE, "Search path is not a regular file or directory.", path=relative)
    return lexical


@dataclass(slots=True)
class _SearchTraversalState:
    truncated: bool = False
    truncation_reason: str | None = None
    directories_seen: int = 0
    skipped_reasons: set[str] = field(default_factory=set)


def _iter_regular_files(
    root: Path,
    scope: Path,
    state: _SearchTraversalState,
):
    if scope != root and scope.name in DEFAULT_IGNORED_DIRECTORIES:
        return
    if scope.is_file():
        try:
            metadata = scope.stat()
        except PermissionError as exc:
            raise ToolError(
                ToolErrorCode.PERMISSION_DENIED,
                "Permission denied while inspecting search file.",
                path=scope,
            ) from exc
        except OSError as exc:
            raise ToolError(
                ToolErrorCode.FILESYSTEM_ERROR,
                f"Unable to inspect search file: {exc}",
                path=scope,
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ToolError(ToolErrorCode.NOT_A_FILE, "Search path is not a regular file.", path=scope)
        yield scope, scope.relative_to(root).as_posix()
        return

    def walk(directory: Path, depth: int):
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(
                    iterator,
                    key=lambda entry: _path_sort_key(
                        Path(entry.path).relative_to(root).as_posix()
                    ),
                )
        except PermissionError:
            state.skipped_reasons.add("permission_denied")
            return
        except OSError:
            state.skipped_reasons.add("filesystem_error")
            return

        for entry in entries:
            if entry.name in DEFAULT_IGNORED_DIRECTORIES or entry.is_symlink():
                continue
            entry_path = Path(entry.path)
            if depth + 1 > DEFAULT_MAX_DEPTH:
                state.truncated = True
                state.truncation_reason = "max_depth"
                return
            try:
                metadata = entry.stat(follow_symlinks=False)
            except PermissionError:
                state.skipped_reasons.add("permission_denied")
                continue
            except OSError:
                state.skipped_reasons.add("filesystem_error")
                continue
            if stat.S_ISDIR(metadata.st_mode):
                if state.directories_seen >= MAX_SEARCH_DIRECTORIES:
                    state.truncated = True
                    state.truncation_reason = "max_directories"
                    return
                state.directories_seen += 1
                yield from walk(entry_path, depth + 1)
                if state.truncated:
                    return
            elif stat.S_ISREG(metadata.st_mode):
                yield entry_path, entry_path.relative_to(root).as_posix()

    yield from walk(scope, len(scope.relative_to(root).parts))


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "MAX_MAX_FILE_BYTES",
    "MAX_MAX_RESULTS",
    "DEFAULT_MAX_QUERY_LENGTH",
    "DEFAULT_MAX_RESULTS",
    "SearchCodeResult",
    "SearchCodeTool",
    "SearchMatch",
    "search_code",
]
