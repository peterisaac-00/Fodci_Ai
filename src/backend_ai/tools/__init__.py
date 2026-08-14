"""Agent tool boundary and the Phase 3.1 filesystem discovery tool."""

from backend_ai.core.contracts import Tool
from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.read_file import DEFAULT_MAX_READ_BYTES, ReadFileResult, ReadFileTool, read_file
from backend_ai.tools.filesystem import (
    DEFAULT_IGNORED_DIRECTORIES,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_DIRECTORIES,
    DEFAULT_MAX_FILES,
    DiscoveredDirectory,
    DiscoveredFile,
    FileDiscoveryResult,
    ListFilesTool,
    list_files,
)

__all__ = [
    "Tool",
    "ToolError",
    "ToolErrorCode",
    "ToolMetadata",
    "DEFAULT_MAX_READ_BYTES",
    "ReadFileResult",
    "ReadFileTool",
    "read_file",
    "DEFAULT_IGNORED_DIRECTORIES",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_DIRECTORIES",
    "DEFAULT_MAX_FILES",
    "DiscoveredDirectory",
    "DiscoveredFile",
    "FileDiscoveryResult",
    "ListFilesTool",
    "list_files",
]
