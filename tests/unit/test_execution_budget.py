from __future__ import annotations

import pytest

from backend_ai.agent import (
    BudgetDimension,
    BudgetExhaustion,
    ExecutionBudget,
    ExecutionBudgetLedger,
)


def test_conservative_defaults_are_finite_and_serializable() -> None:
    budget = ExecutionBudget.conservative_defaults()
    data = budget.to_dict()
    assert data["tool_calls"] == 16
    assert data["wall_time_seconds"] == 300.0
    assert all(value >= 0 for value in data.values())


def test_custom_budget_and_remaining_are_bounded() -> None:
    ledger = ExecutionBudgetLedger(ExecutionBudget(max_tool_calls=2, max_mutations=1))
    first = ledger.consume("read", dimension=BudgetDimension.TOOL_CALLS)
    snapshot = ledger.snapshot()
    assert first.allowed and first.remaining == 1
    assert snapshot.remaining["tool_calls"] == 1
    assert snapshot.remaining["mutations"] == 1


@pytest.mark.parametrize("field", [
    "max_iterations", "max_tool_calls", "max_mutations", "max_command_executions",
    "max_test_executions", "max_application_launches", "max_tool_output_bytes",
    "max_stdout_bytes", "max_stderr_bytes", "max_context_tokens", "max_action_steps",
])
def test_negative_limits_are_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        ExecutionBudget(**{field: -1})


def test_zero_limit_blocks_before_operation() -> None:
    ledger = ExecutionBudgetLedger(ExecutionBudget(max_tool_calls=0))
    decision = ledger.check("tool", dimension=BudgetDimension.TOOL_CALLS)
    assert not decision.allowed
    assert decision.exhaustion is BudgetExhaustion.TOOL_CALL_LIMIT_REACHED
    assert not decision.operation_started
    assert ledger.usage.tool_calls_attempted == 0


def test_tool_call_exhaustion_counts_attempts_not_completed_only() -> None:
    ledger = ExecutionBudgetLedger(ExecutionBudget(max_tool_calls=1))
    assert ledger.consume("tool", dimension=BudgetDimension.TOOL_CALLS).allowed
    ledger.complete_tool()
    denied = ledger.consume("tool", dimension=BudgetDimension.TOOL_CALLS)
    assert not denied.allowed
    assert ledger.usage.tool_calls_attempted == 1
    assert ledger.usage.tool_calls_completed == 1


def test_each_operation_dimension_is_enforced_pre_execution() -> None:
    budget = ExecutionBudget(max_mutations=1, max_command_executions=1, max_test_executions=1, max_application_launches=1)
    ledger = ExecutionBudgetLedger(budget)
    assert ledger.check_tool_operation("write_file").allowed
    assert ledger.consume("mutation", dimension=BudgetDimension.MUTATIONS).allowed
    assert not ledger.check_tool_operation("write_file").allowed
    assert ledger.check_tool_operation("run_command").allowed
    assert ledger.check_tool_operation("run_tests").allowed
    assert ledger.check_tool_operation("run_application").allowed


def test_output_and_context_usage_never_have_negative_remaining() -> None:
    ledger = ExecutionBudgetLedger(ExecutionBudget(max_tool_output_bytes=5, max_context_tokens=3))
    ledger.account_tool_result(tool_name="tool", tool_output_bytes=8, context_tokens=7)
    snapshot = ledger.snapshot()
    assert snapshot.usage.tool_output_bytes == 8
    assert snapshot.remaining["tool_output_bytes"] == 0
    assert snapshot.remaining["context_tokens"] == 0
    assert "tool_output_bytes" in snapshot.exhausted_dimensions


def test_wall_time_uses_monotonic_clock_and_blocks_new_operation() -> None:
    now = [0.0]
    ledger = ExecutionBudgetLedger(ExecutionBudget(max_wall_time_seconds=1), clock=lambda: now[0])
    assert ledger.consume("tool", dimension=BudgetDimension.TOOL_CALLS).allowed
    now[0] = 2.0
    decision = ledger.check("next", dimension=BudgetDimension.TOOL_CALLS)
    assert not decision.allowed
    assert decision.exhaustion is BudgetExhaustion.WALL_TIME_LIMIT_REACHED


def test_action_and_iteration_limits_are_independent() -> None:
    ledger = ExecutionBudgetLedger(ExecutionBudget(max_iterations=1, max_action_steps=1))
    assert ledger.consume("iteration", dimension=BudgetDimension.ITERATIONS).allowed
    assert ledger.consume("action", dimension=BudgetDimension.ACTION_STEPS).allowed
    assert not ledger.consume("iteration", dimension=BudgetDimension.ITERATIONS).allowed
    assert not ledger.consume("action", dimension=BudgetDimension.ACTION_STEPS).allowed


def test_budget_decisions_are_structured_and_deterministic() -> None:
    ledger = ExecutionBudgetLedger(ExecutionBudget(max_tool_calls=0))
    first = ledger.check("tool", dimension=BudgetDimension.TOOL_CALLS).to_dict()
    second = ledger.check("tool", dimension=BudgetDimension.TOOL_CALLS).to_dict()
    assert first == second
    assert first["operation_started"] is False
    assert first["remaining"] == 0
