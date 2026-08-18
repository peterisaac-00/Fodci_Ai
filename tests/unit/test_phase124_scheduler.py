from __future__ import annotations

from backend_ai.agent.scheduler import (
    AccessMode,
    ConcurrencyPolicy,
    ExecutionMode,
    SideEffectType,
    ToolScheduler,
)


def test_tool_classification_defaults() -> None:
    scheduler = ToolScheduler()
    
    read_profile = scheduler.profile_for("read_file")
    assert read_profile.access_mode is AccessMode.READ
    assert read_profile.concurrency_policy is ConcurrencyPolicy.PARALLEL_SAFE

    write_profile = scheduler.profile_for("edit_file")
    assert write_profile.access_mode is AccessMode.WRITE
    assert write_profile.concurrency_policy is ConcurrencyPolicy.SEQUENTIAL_ONLY

    cmd_profile = scheduler.profile_for("run_command")
    assert cmd_profile.access_mode is AccessMode.EXECUTE
    assert cmd_profile.concurrency_policy is ConcurrencyPolicy.SEQUENTIAL_ONLY

    unknown_profile = scheduler.profile_for("unknown_custom_tool")
    assert unknown_profile.concurrency_policy is ConcurrencyPolicy.UNKNOWN


def test_independent_reads_batch_in_parallel() -> None:
    scheduler = ToolScheduler(max_parallel_tools=3, enabled=True)
    calls = [
        ("read_file", {"path": "a.py"}),
        ("read_file", {"path": "b.py"}),
        ("search_code", {"query": "JWT"}),
    ]
    batches = scheduler.schedule(calls)
    assert len(batches) == 1
    assert batches[0].mode is ExecutionMode.PARALLEL
    assert len(batches[0].calls) == 3


def test_conflicting_reads_stay_sequential_or_split() -> None:
    scheduler = ToolScheduler(max_parallel_tools=3, enabled=True)
    calls = [
        ("read_file", {"path": "a.py"}),
        ("read_file", {"path": "a.py"}),  # same resource conflict
    ]
    batches = scheduler.schedule(calls)
    assert len(batches) == 2
    assert all(b.mode is ExecutionMode.SEQUENTIAL for b in batches)


def test_read_then_write_remains_sequential() -> None:
    scheduler = ToolScheduler(max_parallel_tools=3, enabled=True)
    calls = [
        ("read_file", {"path": "a.py"}),
        ("edit_file", {"path": "a.py"}),
    ]
    batches = scheduler.schedule(calls)
    assert len(batches) == 2
    assert batches[0].mode is ExecutionMode.SEQUENTIAL
    assert batches[1].mode is ExecutionMode.SEQUENTIAL


def test_concurrency_limit_enforced() -> None:
    scheduler = ToolScheduler(max_parallel_tools=2, enabled=True)
    calls = [
        ("read_file", {"path": "a.py"}),
        ("read_file", {"path": "b.py"}),
        ("read_file", {"path": "c.py"}),
        ("read_file", {"path": "d.py"}),
    ]
    batches = scheduler.schedule(calls)
    assert len(batches) == 2
    assert all(len(b.calls) <= 2 for b in batches)
    assert all(b.mode is ExecutionMode.PARALLEL for b in batches)


def test_disabled_mode_forces_sequential() -> None:
    scheduler = ToolScheduler(max_parallel_tools=3, enabled=False)
    calls = [
        ("read_file", {"path": "a.py"}),
        ("read_file", {"path": "b.py"}),
        ("read_file", {"path": "c.py"}),
    ]
    batches = scheduler.schedule(calls)
    assert len(batches) == 3
    assert all(b.mode is ExecutionMode.SEQUENTIAL for b in batches)
