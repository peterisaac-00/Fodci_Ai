"""Bounded Phase 6.5 execution-budget smoke; no subprocess or project mutation."""
from __future__ import annotations

from backend_ai.agent import BudgetDimension, BudgetExhaustion, ExecutionBudget, ExecutionBudgetLedger


def main() -> None:
    defaults = ExecutionBudget.conservative_defaults()
    assert defaults.max_tool_calls == 16

    ledger = ExecutionBudgetLedger(ExecutionBudget(max_tool_calls=1, max_tool_output_bytes=4))
    allowed = ledger.consume("tool", dimension=BudgetDimension.TOOL_CALLS)
    assert allowed.allowed and allowed.operation_started
    ledger.complete_tool()
    ledger.account_tool_result(tool_name="tool", tool_output_bytes=8)
    denied = ledger.check("next tool", dimension=BudgetDimension.TOOL_CALLS)
    assert not denied.allowed
    assert denied.exhaustion is BudgetExhaustion.TOOL_CALL_LIMIT_REACHED
    snapshot = ledger.snapshot()
    assert snapshot.remaining["tool_calls"] == 0
    assert snapshot.remaining["tool_output_bytes"] == 0
    assert snapshot.to_dict()["usage"]["tool_calls_completed"] == 1
    print("Phase 6.5 execution-budget smoke passed")


if __name__ == "__main__":
    main()
