"""Bounded dependency-aware parallel tool execution scheduler for Phase 12.4."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    MIXED = "mixed"


class SideEffectType(str, Enum):
    NONE = "none"
    FILESYSTEM = "filesystem"
    PROCESS = "process"
    DATABASE = "database"
    GIT = "git"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class ConcurrencyPolicy(str, Enum):
    PARALLEL_SAFE = "parallel_safe"
    SEQUENTIAL_ONLY = "sequential_only"
    CONDITIONALLY_SAFE = "conditionally_safe"
    UNKNOWN = "unknown"


class DependencyType(str, Enum):
    DATA = "data_dependency"
    RESOURCE = "resource_dependency"
    ORDERING = "ordering_dependency"
    MUTATION = "mutation_dependency"
    VERIFICATION = "verification_dependency"
    UNKNOWN = "unknown"


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass(frozen=True, slots=True)
class ToolExecutionProfile:
    """Execution classification and safety metadata for a tool."""

    tool_name: str
    access_mode: AccessMode
    side_effects: SideEffectType
    concurrency_policy: ConcurrencyPolicy
    resource_scope: str = "global"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "access_mode": self.access_mode.value,
            "side_effects": self.side_effects.value,
            "concurrency_policy": self.concurrency_policy.value,
            "resource_scope": self.resource_scope,
        }


@dataclass(frozen=True, slots=True)
class ToolCallDependency:
    """Relationship between two planned or scheduled tool calls."""

    source_index: int
    target_index: int
    dependency_type: DependencyType
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "dependency_type": self.dependency_type.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ExecutionBatch:
    """A bounded batch of tool calls executed either sequentially or in parallel."""

    batch_id: int
    mode: ExecutionMode
    calls: tuple[Any, ...]  # Tuples of (original_index, tool_name, arguments)
    dependencies: tuple[ToolCallDependency, ...] = ()
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "mode": self.mode.value,
            "calls": [{"index": item[0], "tool": item[1], "arguments": item[2]} for item in self.calls],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ParallelMetrics:
    """Metrics regarding parallel tool execution."""

    total_tool_calls: int = 0
    parallel_tool_calls: int = 0
    sequential_tool_calls: int = 0
    parallel_batches: int = 0
    maximum_concurrency: int = 1
    average_batch_size: float = 1.0
    execution_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tool_calls": self.total_tool_calls,
            "parallel_tool_calls": self.parallel_tool_calls,
            "sequential_tool_calls": self.sequential_tool_calls,
            "parallel_batches": self.parallel_batches,
            "maximum_concurrency": self.maximum_concurrency,
            "average_batch_size": self.average_batch_size,
            "execution_duration_ms": round(self.execution_duration_ms, 2),
        }


class ToolScheduler:
    """Dependency-aware, bounded parallel execution scheduler."""

    DEFAULT_PROFILES: dict[str, ToolExecutionProfile] = {
        "list_files": ToolExecutionProfile("list_files", AccessMode.READ, SideEffectType.FILESYSTEM, ConcurrencyPolicy.PARALLEL_SAFE, "directory"),
        "read_file": ToolExecutionProfile("read_file", AccessMode.READ, SideEffectType.NONE, ConcurrencyPolicy.PARALLEL_SAFE, "file"),
        "search_code": ToolExecutionProfile("search_code", AccessMode.READ, SideEffectType.NONE, ConcurrencyPolicy.PARALLEL_SAFE, "repository"),
        "project_structure": ToolExecutionProfile("project_structure", AccessMode.READ, SideEffectType.NONE, ConcurrencyPolicy.PARALLEL_SAFE, "repository"),
        "project_context": ToolExecutionProfile("project_context", AccessMode.READ, SideEffectType.NONE, ConcurrencyPolicy.PARALLEL_SAFE, "repository"),
        "git_diff": ToolExecutionProfile("git_diff", AccessMode.READ, SideEffectType.GIT, ConcurrencyPolicy.PARALLEL_SAFE, "repository"),
        "git_status": ToolExecutionProfile("git_status", AccessMode.READ, SideEffectType.GIT, ConcurrencyPolicy.PARALLEL_SAFE, "repository"),
        "test_result_parser": ToolExecutionProfile("test_result_parser", AccessMode.READ, SideEffectType.NONE, ConcurrencyPolicy.PARALLEL_SAFE, "memory"),
        
        # Mutations / Sequential Only
        "write_file": ToolExecutionProfile("write_file", AccessMode.WRITE, SideEffectType.FILESYSTEM, ConcurrencyPolicy.SEQUENTIAL_ONLY, "file"),
        "edit_file": ToolExecutionProfile("edit_file", AccessMode.WRITE, SideEffectType.FILESYSTEM, ConcurrencyPolicy.SEQUENTIAL_ONLY, "file"),
        "delete_file": ToolExecutionProfile("delete_file", AccessMode.WRITE, SideEffectType.FILESYSTEM, ConcurrencyPolicy.SEQUENTIAL_ONLY, "file"),
        "run_command": ToolExecutionProfile("run_command", AccessMode.EXECUTE, SideEffectType.PROCESS, ConcurrencyPolicy.SEQUENTIAL_ONLY, "process"),
        "run_command_with_policy": ToolExecutionProfile("run_command_with_policy", AccessMode.EXECUTE, SideEffectType.PROCESS, ConcurrencyPolicy.SEQUENTIAL_ONLY, "process"),
        "run_application": ToolExecutionProfile("run_application", AccessMode.EXECUTE, SideEffectType.PROCESS, ConcurrencyPolicy.SEQUENTIAL_ONLY, "process"),
        "run_tests": ToolExecutionProfile("run_tests", AccessMode.EXECUTE, SideEffectType.PROCESS, ConcurrencyPolicy.SEQUENTIAL_ONLY, "process"),
    }

    def __init__(self, *, max_parallel_tools: int = 3, enabled: bool = True) -> None:
        self.max_parallel_tools = max(1, max_parallel_tools)
        self.enabled = enabled

    def profile_for(self, tool_name: str) -> ToolExecutionProfile:
        return self.DEFAULT_PROFILES.get(
            tool_name,
            ToolExecutionProfile(tool_name, AccessMode.MIXED, SideEffectType.UNKNOWN, ConcurrencyPolicy.UNKNOWN, "global"),
        )

    def extract_resource(self, tool_name: str, arguments: Mapping[str, Any]) -> str | None:
        """Extract a canonical resource identifier for conflict detection."""
        for key in ("path", "file_path", "target", "directory", "dir"):
            val = arguments.get(key)
            if val is not None:
                return str(Path(val).as_posix())
        return None

    def schedule(self, calls: Sequence[tuple[str, Mapping[str, Any]]]) -> tuple[ExecutionBatch, ...]:
        """Group tool calls into sequential and parallel execution batches."""
        if not calls:
            return ()

        if not self.enabled or len(calls) == 1:
            return tuple(
                ExecutionBatch(
                    batch_id=i,
                    mode=ExecutionMode.SEQUENTIAL,
                    calls=((i, name, dict(args)),),
                )
                for i, (name, args) in enumerate(calls)
            )

        batches: list[ExecutionBatch] = []
        current_parallel_group: list[tuple[int, str, dict[str, Any]]] = []
        batch_counter = 0

        for i, (name, args) in enumerate(calls):
            profile = self.profile_for(name)
            is_parallel_safe = (
                profile.concurrency_policy == ConcurrencyPolicy.PARALLEL_SAFE
                and profile.access_mode == AccessMode.READ
            )

            if not is_parallel_safe:
                # Flush any active parallel group first
                if current_parallel_group:
                    batches.extend(self._chunk_parallel_group(batch_counter, current_parallel_group))
                    batch_counter += len(batches) - batch_counter
                    current_parallel_group = []

                # Add sequential batch
                batches.append(
                    ExecutionBatch(
                        batch_id=batch_counter,
                        mode=ExecutionMode.SEQUENTIAL,
                        calls=((i, name, dict(args)),),
                    )
                )
                batch_counter += 1
            else:
                # Check for resource conflicts within the current parallel group
                resource = self.extract_resource(name, args)
                has_conflict = False
                if resource:
                    for _, existing_name, existing_args in current_parallel_group:
                        if self.extract_resource(existing_name, existing_args) == resource:
                            has_conflict = True
                            break

                if has_conflict or len(current_parallel_group) >= self.max_parallel_tools:
                    if current_parallel_group:
                        batches.extend(self._chunk_parallel_group(batch_counter, current_parallel_group))
                        batch_counter = len(batches)
                        current_parallel_group = []
                    if has_conflict:
                        # If conflicting with active group, flush and put in sequential or new group
                        pass

                current_parallel_group.append((i, name, dict(args)))

        if current_parallel_group:
            batches.extend(self._chunk_parallel_group(batch_counter, current_parallel_group))

        return tuple(batches)

    def _chunk_parallel_group(
        self, start_id: int, group: list[tuple[int, str, dict[str, Any]]]
    ) -> list[ExecutionBatch]:
        chunks: list[ExecutionBatch] = []
        for chunk_idx in range(0, len(group), self.max_parallel_tools):
            sub = group[chunk_idx : chunk_idx + self.max_parallel_tools]
            chunks.append(
                ExecutionBatch(
                    batch_id=start_id + len(chunks),
                    mode=ExecutionMode.PARALLEL if len(sub) > 1 else ExecutionMode.SEQUENTIAL,
                    calls=tuple(sub),
                )
            )
        return chunks


__all__ = [
    "AccessMode",
    "SideEffectType",
    "ConcurrencyPolicy",
    "DependencyType",
    "ExecutionMode",
    "ToolExecutionProfile",
    "ToolCallDependency",
    "ExecutionBatch",
    "ParallelMetrics",
    "ToolScheduler",
]
