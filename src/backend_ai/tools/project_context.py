"""Canonical, bounded project context built from structural tool output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.filesystem import DEFAULT_MAX_DEPTH, DEFAULT_MAX_DIRECTORIES, DEFAULT_MAX_FILES
from backend_ai.tools.project_structure import (
    DEFAULT_MAX_INSPECTED_FILES,
    DEFAULT_MAX_STRUCTURE_FILE_BYTES,
    Detection,
    LanguageSummary,
    ProjectStructureResult,
    project_structure,
)


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """Immutable, safe context for future Agent reasoning."""

    root: Path
    project_type: str
    stack_summary: str
    languages: tuple[LanguageSummary, ...]
    frameworks: tuple[Detection, ...]
    package_managers: tuple[Detection, ...]
    databases: tuple[Detection, ...]
    test_frameworks: tuple[Detection, ...]
    infrastructure: tuple[Detection, ...]
    source_directories: tuple[str, ...]
    test_directories: tuple[str, ...]
    documentation_directories: tuple[str, ...]
    config_files: tuple[str, ...]
    dependency_files: tuple[str, ...]
    important_files: tuple[str, ...]
    entry_points: tuple[Detection, ...]
    project_files: tuple[str, ...]
    confidence: str
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]
    truncated: bool
    truncation_reason: str | None
    completeness: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible data without raw source content."""

        return {
            "root": str(self.root),
            "project_type": self.project_type,
            "stack_summary": self.stack_summary,
            "languages": [item.to_dict() for item in self.languages],
            "frameworks": [item.to_dict() for item in self.frameworks],
            "package_managers": [item.to_dict() for item in self.package_managers],
            "databases": [item.to_dict() for item in self.databases],
            "test_frameworks": [item.to_dict() for item in self.test_frameworks],
            "infrastructure": [item.to_dict() for item in self.infrastructure],
            "source_directories": list(self.source_directories),
            "test_directories": list(self.test_directories),
            "documentation_directories": list(self.documentation_directories),
            "config_files": list(self.config_files),
            "dependency_files": list(self.dependency_files),
            "important_files": list(self.important_files),
            "entry_points": [item.to_dict() for item in self.entry_points],
            "project_files": list(self.project_files),
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "completeness": self.completeness,
        }


class ProjectContextBuilder:
    """Build canonical context by composing the existing structure tool."""

    def build(
        self,
        project_root: Path | str,
        *,
        max_files: int = DEFAULT_MAX_FILES,
        max_directories: int = DEFAULT_MAX_DIRECTORIES,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_file_bytes: int = DEFAULT_MAX_STRUCTURE_FILE_BYTES,
        max_inspected_files: int = DEFAULT_MAX_INSPECTED_FILES,
    ) -> ProjectContext:
        """Build context without a second filesystem scanner or project execution."""

        structure = project_structure(
            project_root,
            max_files=max_files,
            max_directories=max_directories,
            max_depth=max_depth,
            max_file_bytes=max_file_bytes,
            max_inspected_files=max_inspected_files,
        )
        return self._from_structure(structure)

    @staticmethod
    def _from_structure(structure: ProjectStructureResult) -> ProjectContext:
        warnings = set(structure.warnings)
        truncated = structure.truncated
        truncation_reason = structure.truncation_reason
        if not truncation_reason:
            for warning, reason in (
                ("max_inspected_files", "max_inspected_files"),
                ("byte limit", "max_file_bytes"),
            ):
                if any(warning in item for item in structure.warnings):
                    truncated = True
                    truncation_reason = reason
                    break
        if truncated and "Context is partial because bounded discovery or inspection was incomplete." not in warnings:
            warnings.add("Context is partial because bounded discovery or inspection was incomplete.")
        warnings_tuple = tuple(sorted(warnings))
        return ProjectContext(
            root=structure.root,
            project_type=structure.project_type,
            stack_summary=_stack_summary(structure),
            languages=structure.languages,
            frameworks=structure.frameworks,
            package_managers=structure.package_managers,
            databases=structure.databases,
            test_frameworks=structure.test_frameworks,
            infrastructure=structure.infrastructure,
            source_directories=tuple(
                item.relative_path for item in structure.directories if item.category == "source"
            ),
            test_directories=tuple(
                item.relative_path for item in structure.directories if item.category == "tests"
            ),
            documentation_directories=tuple(
                item.relative_path for item in structure.directories if item.category == "documentation"
            ),
            config_files=structure.config_files,
            dependency_files=structure.dependency_files,
            important_files=structure.important_files,
            entry_points=structure.entry_points,
            project_files=structure.project_files,
            confidence=structure.confidence,
            evidence=structure.evidence,
            warnings=warnings_tuple,
            truncated=truncated,
            truncation_reason=truncation_reason,
            completeness="partial" if truncated else "complete",
        )


class ProjectContextTool:
    """First-class read-only Agent tool for building canonical project context."""

    name = "project_context"
    description = (
        "Build a compact immutable project context from bounded structural evidence. "
        "Read-only, deterministic, safe, and not full project understanding."
    )
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root"],
            "properties": {
                "project_root": {"type": "string", "description": "Explicit project directory."},
                "max_files": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_FILES},
                "max_directories": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_DIRECTORIES},
                "max_depth": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_DEPTH},
                "max_file_bytes": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_STRUCTURE_FILE_BYTES},
                "max_inspected_files": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_INSPECTED_FILES},
            },
        },
    )

    def __init__(self, builder: ProjectContextBuilder | None = None) -> None:
        self._builder = builder or ProjectContextBuilder()

    def run(self, arguments: Mapping[str, Any]) -> ProjectContext:
        """Validate a structured request and build context."""

        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "project_context arguments must be a mapping.")
        if "project_root" not in arguments:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "project_context requires 'project_root'.")
        return self._builder.build(
            arguments["project_root"],
            max_files=arguments.get("max_files", DEFAULT_MAX_FILES),
            max_directories=arguments.get("max_directories", DEFAULT_MAX_DIRECTORIES),
            max_depth=arguments.get("max_depth", DEFAULT_MAX_DEPTH),
            max_file_bytes=arguments.get("max_file_bytes", DEFAULT_MAX_STRUCTURE_FILE_BYTES),
            max_inspected_files=arguments.get("max_inspected_files", DEFAULT_MAX_INSPECTED_FILES),
        )


def project_context(
    project_root: Path | str,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_directories: int = DEFAULT_MAX_DIRECTORIES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_file_bytes: int = DEFAULT_MAX_STRUCTURE_FILE_BYTES,
    max_inspected_files: int = DEFAULT_MAX_INSPECTED_FILES,
) -> ProjectContext:
    """Build canonical context from an explicit project root."""

    return ProjectContextBuilder().build(
        project_root,
        max_files=max_files,
        max_directories=max_directories,
        max_depth=max_depth,
        max_file_bytes=max_file_bytes,
        max_inspected_files=max_inspected_files,
    )


def _stack_summary(structure: ProjectStructureResult) -> str:
    parts: list[str] = []
    if structure.project_type == "python":
        parts.append("Python")
    elif structure.project_type == "node":
        parts.append("Node.js")
    elif structure.project_type == "mixed":
        names = {item.name for item in structure.languages}
        for name in ("Python", "JavaScript", "TypeScript"):
            if name in names:
                parts.append(name)
    for detection in structure.frameworks:
        if detection.name not in {"Python", "Node.js", "JavaScript", "TypeScript"}:
            parts.append(detection.name)
    for detection in structure.databases:
        parts.append(detection.name)
    for detection in structure.test_frameworks:
        if detection.name != "generic tests":
            parts.append(detection.name)
    for detection in structure.infrastructure:
        parts.append(detection.name)
    unique: list[str] = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    return " + ".join(unique) if unique else "Insufficient structural evidence"


__all__ = [
    "ProjectContext",
    "ProjectContextBuilder",
    "ProjectContextTool",
    "project_context",
]
