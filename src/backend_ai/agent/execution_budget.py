"""Centralized deterministic execution budgets for the autonomous loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Callable


class BudgetDimension(str, Enum):
    ITERATIONS = "iterations"
    TOOL_CALLS = "tool_calls"
    MUTATIONS = "mutations"
    COMMAND_EXECUTIONS = "command_executions"
    TEST_EXECUTIONS = "test_executions"
    APPLICATION_LAUNCHES = "application_launches"
    WALL_TIME_SECONDS = "wall_time_seconds"
    TOOL_OUTPUT_BYTES = "tool_output_bytes"
    STDOUT_BYTES = "stdout_bytes"
    STDERR_BYTES = "stderr_bytes"
    CONTEXT_TOKENS = "context_tokens"
    ACTION_STEPS = "action_steps"


class BudgetExhaustion(str, Enum):
    ITERATION_LIMIT_REACHED = "ITERATION_LIMIT_REACHED"
    TOOL_CALL_LIMIT_REACHED = "TOOL_CALL_LIMIT_REACHED"
    MUTATION_LIMIT_REACHED = "MUTATION_LIMIT_REACHED"
    COMMAND_LIMIT_REACHED = "COMMAND_LIMIT_REACHED"
    TEST_LIMIT_REACHED = "TEST_LIMIT_REACHED"
    APPLICATION_LIMIT_REACHED = "APPLICATION_LIMIT_REACHED"
    WALL_TIME_LIMIT_REACHED = "WALL_TIME_LIMIT_REACHED"
    OUTPUT_BYTES_LIMIT_REACHED = "OUTPUT_BYTES_LIMIT_REACHED"
    STDOUT_BYTES_LIMIT_REACHED = "STDOUT_BYTES_LIMIT_REACHED"
    STDERR_BYTES_LIMIT_REACHED = "STDERR_BYTES_LIMIT_REACHED"
    CONTEXT_TOKEN_LIMIT_REACHED = "CONTEXT_TOKEN_LIMIT_REACHED"
    ACTION_LIMIT_REACHED = "ACTION_LIMIT_REACHED"
    UNKNOWN_BUDGET_LIMIT = "UNKNOWN_BUDGET_LIMIT"


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    max_iterations: int = 16
    max_tool_calls: int = 16
    max_mutations: int = 4
    max_command_executions: int = 4
    max_test_executions: int = 4
    max_application_launches: int = 2
    max_wall_time_seconds: float = 300.0
    max_tool_output_bytes: int = 131_072
    max_stdout_bytes: int = 65_536
    max_stderr_bytes: int = 65_536
    max_context_tokens: int = 65_536
    max_action_steps: int = 16

    def __post_init__(self) -> None:
        integer_fields = (
            "max_iterations", "max_tool_calls", "max_mutations", "max_command_executions",
            "max_test_executions", "max_application_launches", "max_tool_output_bytes",
            "max_stdout_bytes", "max_stderr_bytes", "max_context_tokens", "max_action_steps",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer; zero disables that operation")
        if not isinstance(self.max_wall_time_seconds, (int, float)) or isinstance(self.max_wall_time_seconds, bool) or self.max_wall_time_seconds < 0:
            raise ValueError("max_wall_time_seconds must be non-negative")
        ceilings = {
            "max_iterations": 4096, "max_tool_calls": 4096, "max_mutations": 1024,
            "max_command_executions": 1024, "max_test_executions": 1024, "max_application_launches": 1024,
            "max_wall_time_seconds": 86_400.0, "max_tool_output_bytes": 16 * 1024 * 1024,
            "max_stdout_bytes": 16 * 1024 * 1024, "max_stderr_bytes": 16 * 1024 * 1024,
            "max_context_tokens": 1_000_000, "max_action_steps": 4096,
        }
        for name, ceiling in ceilings.items():
            if getattr(self, name) > ceiling:
                raise ValueError(f"{name} exceeds the safety ceiling")

    @classmethod
    def conservative_defaults(cls) -> "ExecutionBudget":
        return cls()

    def limits(self) -> dict[BudgetDimension, int | float]:
        return {
            BudgetDimension.ITERATIONS: self.max_iterations,
            BudgetDimension.TOOL_CALLS: self.max_tool_calls,
            BudgetDimension.MUTATIONS: self.max_mutations,
            BudgetDimension.COMMAND_EXECUTIONS: self.max_command_executions,
            BudgetDimension.TEST_EXECUTIONS: self.max_test_executions,
            BudgetDimension.APPLICATION_LAUNCHES: self.max_application_launches,
            BudgetDimension.WALL_TIME_SECONDS: float(self.max_wall_time_seconds),
            BudgetDimension.TOOL_OUTPUT_BYTES: self.max_tool_output_bytes,
            BudgetDimension.STDOUT_BYTES: self.max_stdout_bytes,
            BudgetDimension.STDERR_BYTES: self.max_stderr_bytes,
            BudgetDimension.CONTEXT_TOKENS: self.max_context_tokens,
            BudgetDimension.ACTION_STEPS: self.max_action_steps,
        }

    def to_dict(self) -> dict[str, int | float]:
        return {dimension.value: limit for dimension, limit in self.limits().items()}


@dataclass(frozen=True, slots=True)
class ExecutionUsage:
    iterations_started: int = 0
    tool_calls_attempted: int = 0
    tool_calls_completed: int = 0
    mutation_operations: int = 0
    command_executions: int = 0
    test_executions: int = 0
    application_launches: int = 0
    elapsed_wall_time_seconds: float = 0.0
    tool_output_bytes: int = 0
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    context_tokens: int = 0
    action_steps: int = 0

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} usage must be non-negative")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "iterations_started": self.iterations_started, "tool_calls_attempted": self.tool_calls_attempted,
            "tool_calls_completed": self.tool_calls_completed, "mutation_operations": self.mutation_operations,
            "command_executions": self.command_executions, "test_executions": self.test_executions,
            "application_launches": self.application_launches, "elapsed_wall_time_seconds": round(self.elapsed_wall_time_seconds, 2),
            "tool_output_bytes": self.tool_output_bytes, "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes, "context_tokens": self.context_tokens, "action_steps": self.action_steps,
        }


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    operation: str
    dimension: BudgetDimension | None
    exhaustion: BudgetExhaustion | None
    configured_limit: int | float | None
    usage_at_decision: int | float
    remaining: int | float
    operation_started: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed, "operation": self.operation,
            "dimension": self.dimension.value if self.dimension else None,
            "exhaustion": self.exhaustion.value if self.exhaustion else None,
            "configured_limit": self.configured_limit, "usage_at_decision": self.usage_at_decision,
            "remaining": self.remaining, "operation_started": self.operation_started,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ExecutionBudgetSnapshot:
    budget: ExecutionBudget
    usage: ExecutionUsage
    remaining: dict[str, int | float]
    exhausted_dimensions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    usage_complete: bool = True

    def to_dict(self) -> dict[str, object]:
        return {"limits": self.budget.to_dict(), "usage": self.usage.to_dict(), "remaining": dict(self.remaining), "exhausted_dimensions": list(self.exhausted_dimensions), "warnings": list(self.warnings), "usage_complete": self.usage_complete}


class ExecutionBudgetLedger:
    """The single authoritative mutable accounting mechanism for one invocation."""

    def __init__(self, budget: ExecutionBudget, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._clock = clock
        self._started_at = clock()
        self._usage = ExecutionUsage()
        self._last_decision: BudgetDecision | None = None

    @property
    def usage(self) -> ExecutionUsage:
        return self._with_elapsed()

    @property
    def last_decision(self) -> BudgetDecision | None:
        return self._last_decision

    def check(self, operation: str, *, dimension: BudgetDimension, amount: int | float = 1) -> BudgetDecision:
        usage = self._with_elapsed()
        limit = self.budget.limits()[dimension]
        current = self._value_for(dimension, usage)
        remaining = max(0, limit - current)
        elapsed = usage.elapsed_wall_time_seconds
        if elapsed >= float(self.budget.max_wall_time_seconds) and dimension is not BudgetDimension.WALL_TIME_SECONDS:
            decision = BudgetDecision(False, operation, BudgetDimension.WALL_TIME_SECONDS, BudgetExhaustion.WALL_TIME_LIMIT_REACHED, self.budget.max_wall_time_seconds, elapsed, 0, False, "autonomous wall-time budget is exhausted")
        elif current + amount > limit:
            decision = BudgetDecision(False, operation, dimension, _exhaustion_for(dimension), limit, current, remaining, False, "configured execution budget is exhausted")
        else:
            decision = BudgetDecision(True, operation, dimension, None, limit, current, remaining, False, "operation is within budget")
        self._last_decision = decision
        return decision

    def consume(self, operation: str, *, dimension: BudgetDimension, amount: int | float = 1) -> BudgetDecision:
        decision = self.check(operation, dimension=dimension, amount=amount)
        if not decision.allowed:
            return decision
        self._usage = self._increment(dimension, amount, self._usage)
        decision = BudgetDecision(True, operation, dimension, None, decision.configured_limit, self._value_for(dimension, self._with_elapsed()), max(0, float(decision.remaining) - amount), True, "operation accounted")
        self._last_decision = decision
        return decision

    def complete_tool(self) -> None:
        fields = self._with_elapsed().to_dict()
        fields["tool_calls_completed"] += 1
        self._usage = ExecutionUsage(**fields)

    def account_tool_result(self, *, tool_name: str, tool_output_bytes: int = 0, stdout_bytes: int = 0, stderr_bytes: int = 0, context_tokens: int = 0, success: bool = True) -> None:
        self._usage = self._increment(BudgetDimension.TOOL_OUTPUT_BYTES, max(0, tool_output_bytes), self._usage)
        self._usage = self._increment(BudgetDimension.STDOUT_BYTES, max(0, stdout_bytes), self._usage)
        self._usage = self._increment(BudgetDimension.STDERR_BYTES, max(0, stderr_bytes), self._usage)
        self._usage = self._increment(BudgetDimension.CONTEXT_TOKENS, max(0, context_tokens), self._usage)
        if success:
            self._usage = ExecutionUsage(**{**self._usage.to_dict(), "elapsed_wall_time_seconds": self._elapsed()})

    def check_tool_operation(self, tool_name: str) -> BudgetDecision:
        normalized = tool_name.casefold()
        decision = self.check(tool_name, dimension=BudgetDimension.TOOL_CALLS)
        if not decision.allowed:
            return decision
        specific = None
        if normalized in {"write_file", "edit_file", "delete_file"}:
            specific = self.check(tool_name, dimension=BudgetDimension.MUTATIONS)
        elif normalized in {"run_command", "run_command_with_policy"}:
            specific = self.check(tool_name, dimension=BudgetDimension.COMMAND_EXECUTIONS)
        elif normalized in {"run_tests"}:
            specific = self.check(tool_name, dimension=BudgetDimension.TEST_EXECUTIONS)
        elif normalized in {"run_application"}:
            specific = self.check(tool_name, dimension=BudgetDimension.APPLICATION_LAUNCHES)
        return specific or decision

    def snapshot(self) -> ExecutionBudgetSnapshot:
        usage = self._with_elapsed()
        remaining = {dimension.value: max(0, limit - self._value_for(dimension, usage)) for dimension, limit in self.budget.limits().items()}
        exhausted = tuple(dimension.value for dimension, limit in self.budget.limits().items() if self._value_for(dimension, usage) >= limit)
        return ExecutionBudgetSnapshot(self.budget, usage, remaining, exhausted, (), True)

    def _elapsed(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    def _with_elapsed(self) -> ExecutionUsage:
        return ExecutionUsage(**{**self._usage.to_dict(), "elapsed_wall_time_seconds": self._elapsed()})

    @staticmethod
    def _value_for(dimension: BudgetDimension, usage: ExecutionUsage) -> int | float:
        return {
            BudgetDimension.ITERATIONS: usage.iterations_started, BudgetDimension.TOOL_CALLS: usage.tool_calls_attempted,
            BudgetDimension.MUTATIONS: usage.mutation_operations, BudgetDimension.COMMAND_EXECUTIONS: usage.command_executions,
            BudgetDimension.TEST_EXECUTIONS: usage.test_executions, BudgetDimension.APPLICATION_LAUNCHES: usage.application_launches,
            BudgetDimension.WALL_TIME_SECONDS: usage.elapsed_wall_time_seconds, BudgetDimension.TOOL_OUTPUT_BYTES: usage.tool_output_bytes,
            BudgetDimension.STDOUT_BYTES: usage.stdout_bytes, BudgetDimension.STDERR_BYTES: usage.stderr_bytes,
            BudgetDimension.CONTEXT_TOKENS: usage.context_tokens, BudgetDimension.ACTION_STEPS: usage.action_steps,
        }[dimension]

    @staticmethod
    def _increment(dimension: BudgetDimension, amount: int | float, usage: ExecutionUsage) -> ExecutionUsage:
        fields = usage.to_dict()
        mapping = {
            BudgetDimension.ITERATIONS: "iterations_started", BudgetDimension.TOOL_CALLS: "tool_calls_attempted",
            BudgetDimension.MUTATIONS: "mutation_operations", BudgetDimension.COMMAND_EXECUTIONS: "command_executions",
            BudgetDimension.TEST_EXECUTIONS: "test_executions", BudgetDimension.APPLICATION_LAUNCHES: "application_launches",
            BudgetDimension.WALL_TIME_SECONDS: "elapsed_wall_time_seconds", BudgetDimension.TOOL_OUTPUT_BYTES: "tool_output_bytes",
            BudgetDimension.STDOUT_BYTES: "stdout_bytes", BudgetDimension.STDERR_BYTES: "stderr_bytes",
            BudgetDimension.CONTEXT_TOKENS: "context_tokens", BudgetDimension.ACTION_STEPS: "action_steps",
        }
        if dimension is BudgetDimension.WALL_TIME_SECONDS:
            return usage
        fields[mapping[dimension]] += amount
        return ExecutionUsage(**fields)


def _exhaustion_for(dimension: BudgetDimension) -> BudgetExhaustion:
    return {
        BudgetDimension.ITERATIONS: BudgetExhaustion.ITERATION_LIMIT_REACHED,
        BudgetDimension.TOOL_CALLS: BudgetExhaustion.TOOL_CALL_LIMIT_REACHED,
        BudgetDimension.MUTATIONS: BudgetExhaustion.MUTATION_LIMIT_REACHED,
        BudgetDimension.COMMAND_EXECUTIONS: BudgetExhaustion.COMMAND_LIMIT_REACHED,
        BudgetDimension.TEST_EXECUTIONS: BudgetExhaustion.TEST_LIMIT_REACHED,
        BudgetDimension.APPLICATION_LAUNCHES: BudgetExhaustion.APPLICATION_LIMIT_REACHED,
        BudgetDimension.WALL_TIME_SECONDS: BudgetExhaustion.WALL_TIME_LIMIT_REACHED,
        BudgetDimension.TOOL_OUTPUT_BYTES: BudgetExhaustion.OUTPUT_BYTES_LIMIT_REACHED,
        BudgetDimension.STDOUT_BYTES: BudgetExhaustion.STDOUT_BYTES_LIMIT_REACHED,
        BudgetDimension.STDERR_BYTES: BudgetExhaustion.STDERR_BYTES_LIMIT_REACHED,
        BudgetDimension.CONTEXT_TOKENS: BudgetExhaustion.CONTEXT_TOKEN_LIMIT_REACHED,
        BudgetDimension.ACTION_STEPS: BudgetExhaustion.ACTION_LIMIT_REACHED,
    }[dimension]


__all__ = ["BudgetDecision", "BudgetDimension", "BudgetExhaustion", "ExecutionBudget", "ExecutionBudgetLedger", "ExecutionBudgetSnapshot", "ExecutionUsage"]
