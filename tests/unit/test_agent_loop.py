from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from backend_ai.agent import (
    ActionParseError,
    AgentConfig,
    AgentLoop,
    AgentStatus,
    ContextBudget,
    ContextBudgetError,
    ToolRegistry,
    parse_agent_output,
)
from backend_ai.agent.protocol import ParsedFinalAnswer, ParsedToolAction
from backend_ai.tools.project_context import project_context


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class FakeEngine:
    tokenizer = FakeTokenizer()

    def __init__(self, outputs: list[str], *, fail: Exception | None = None) -> None:
        self._outputs = iter(outputs)
        self._fail = fail
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if self._fail is not None:
            raise self._fail
        return SimpleNamespace(generated_text=next(self._outputs))


def _write(root: Path, relative: str, content: str = "x") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_registry_is_deterministic_and_exposes_all_phase_three_tools() -> None:
    registry = ToolRegistry.default()

    assert registry.names() == (
        "list_files",
        "project_context",
        "project_structure",
        "read_file",
        "search_code",
    )
    assert tuple(item.name for item in registry.list()) == registry.names()
    assert registry.metadata_for("read_file").input_schema["required"] == ["project_root", "path"]
    with pytest.raises(Exception, match="Unknown tool"):
        registry.get("not_a_tool")


def test_action_protocol_is_strict_and_deterministic() -> None:
    assert parse_agent_output("FINAL: finished") == ParsedFinalAnswer("finished")
    assert parse_agent_output("plain final") == ParsedFinalAnswer("plain final")
    assert parse_agent_output('ACTION: search_code\nARGS: {"query":"FastAPI"}') == ParsedToolAction(
        name="search_code", arguments={"query": "FastAPI"}
    )
    with pytest.raises(ActionParseError) as malformed:
        parse_agent_output("ACTION: search_code")
    assert malformed.value.code == "MALFORMED_ACTION"
    with pytest.raises(ActionParseError) as invalid_json:
        parse_agent_output("ACTION: search_code\nARGS: []")
    assert invalid_json.value.code == "INVALID_ARGUMENT"
    with pytest.raises(ActionParseError) as invalid_name:
        parse_agent_output("ACTION: search-code\nARGS: {}")
    assert invalid_name.value.code == "MALFORMED_ACTION"


def test_context_budget_marks_compaction_and_rejects_unfit_task(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")
    context = project_context(root)
    budget = ContextBudget(FakeTokenizer(), AgentConfig(max_context_tokens=256, reserve_response_tokens=32))

    compacted = budget.render("Understand this project.", context)
    assert compacted.token_count <= 224
    assert compacted.truncated
    assert compacted.warnings

    tiny = ContextBudget(FakeTokenizer(), AgentConfig(max_context_tokens=64, reserve_response_tokens=16))
    with pytest.raises(ContextBudgetError):
        tiny.render("x" * 500, context)


def test_agent_returns_final_answer_without_model_tool_call(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")
    result = AgentLoop(FakeEngine(["FINAL: done"])).run("Understand this project.", root)

    assert result.status is AgentStatus.COMPLETED
    assert result.final_answer == "done"
    assert result.stop_reason == "final_answer"
    assert [call.name for call in result.tool_calls] == ["project_context"]
    assert result.steps[0].tool_call is None


def test_agent_executes_one_real_tool_then_final(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "app/main.py", "from fastapi import FastAPI\n")
    engine = FakeEngine([
        'ACTION: search_code\nARGS: {"query":"FastAPI"}',
        "FINAL: Found evidence.",
    ])
    before = (root / "app" / "main.py").read_text(encoding="utf-8")
    result = AgentLoop(engine).run("Understand this project.", root)

    assert result.status is AgentStatus.COMPLETED
    assert result.final_answer == "Found evidence."
    assert [call.name for call in result.tool_calls] == ["project_context", "search_code"]
    assert result.tool_results[-1].success
    assert result.tool_results[-1].data.matches[0].relative_path == "app/main.py"
    assert (root / "app" / "main.py").read_text(encoding="utf-8") == before
    assert result.to_dict() == AgentLoop(FakeEngine([
        'ACTION: search_code\nARGS: {"query":"FastAPI"}',
        "FINAL: Found evidence.",
    ])).run("Understand this project.", root).to_dict()


def test_agent_continues_after_tool_error_and_returns_final(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")
    result = AgentLoop(FakeEngine([
        'ACTION: read_file\nARGS: {"path":"missing.py"}',
        "FINAL: The requested file was not found.",
    ])).run("Inspect the missing file.", root)

    assert result.status is AgentStatus.COMPLETED
    assert result.tool_results[-1].success is False
    assert result.tool_results[-1].error_code == "PATH_NOT_FOUND"
    assert result.final_answer == "The requested file was not found."


def test_agent_rejects_unknown_tool_as_structured_result(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")
    result = AgentLoop(FakeEngine([
        "ACTION: unknown_tool\nARGS: {}",
        "FINAL: I cannot use that tool.",
    ])).run("Inspect this project.", root)

    assert result.status is AgentStatus.COMPLETED
    assert result.tool_results[-1].error_code == "UNKNOWN_TOOL"
    assert result.final_answer == "I cannot use that tool."


def test_agent_rejects_path_boundary_violation_without_dispatch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")
    result = AgentLoop(FakeEngine([
        'ACTION: search_code\nARGS: {"project_root":"/outside","query":"ok"}',
        "FINAL: Boundary was preserved.",
    ])).run("Inspect this project.", root)

    assert result.status is AgentStatus.COMPLETED
    assert result.tool_results[-1].success is False
    assert result.tool_results[-1].error_code == "PATH_OUTSIDE_ROOT"


def test_agent_limits_tool_calls_and_steps(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")
    tool_limit = AgentLoop(
        FakeEngine(['ACTION: list_files\nARGS: {}']),
        config=AgentConfig(max_steps=3, max_tool_calls=1),
    ).run("Inspect this project.", root)
    assert tool_limit.status is AgentStatus.MAX_TOOL_CALLS

    step_limit = AgentLoop(
        FakeEngine(['ACTION: list_files\nARGS: {}'] * 4),
        config=AgentConfig(max_steps=2, max_tool_calls=8),
    ).run("Inspect this project.", root)
    assert step_limit.status is AgentStatus.MAX_STEPS
    assert len(step_limit.steps) == 2


def test_agent_reports_inference_and_context_errors(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")
    inference_error = AgentLoop(FakeEngine([], fail=RuntimeError("inference down"))).run("Inspect", root)
    assert inference_error.status is AgentStatus.INFERENCE_ERROR
    assert inference_error.errors == ("inference down",)

    context_error = AgentLoop(
        FakeEngine(["FINAL: unreachable"]),
        config=AgentConfig(max_context_tokens=64, reserve_response_tokens=16),
    ).run("x" * 500, root)
    assert context_error.status is AgentStatus.CONTEXT_LIMIT


def test_agent_handles_malformed_model_action_without_crash(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")
    result = AgentLoop(FakeEngine(["ACTION: search_code"])).run("Inspect", root)

    assert result.status is AgentStatus.INVALID_ACTION
    assert result.stop_reason == "invalid_action"
    assert result.errors


def test_agent_rejects_empty_task_before_any_tool_dispatch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    engine = FakeEngine(["FINAL: should not run"])
    result = AgentLoop(engine).run("   ", root)

    assert result.status is AgentStatus.FAILED
    assert result.stop_reason == "invalid_task"
    assert not engine.prompts
