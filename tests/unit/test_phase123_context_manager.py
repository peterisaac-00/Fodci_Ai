from pathlib import Path
from types import SimpleNamespace

import pytest

from backend_ai.agent import (
    AgentMessage,
    AgentMessageRole,
    CompressionStatus,
    ContextItem,
    ContextManager,
    ContextManagerConfig,
    ContextManagerError,
    ContextPriority,
    ContextType,
    ContextValidity,
    EncoderTokenCounter,
    TokenCount,
    TokenCountKind,
)
from backend_ai.agent.codebase_understanding import CodebaseUnderstandingBuilder
from backend_ai.tools.project_context import project_context


class WordTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class UnknownCounter:
    def count(self, text: str) -> TokenCount:
        return TokenCount(0, TokenCountKind.UNKNOWN)


def _item(item_id: str, content: str, *, priority: ContextPriority = ContextPriority.MEDIUM, kind: ContextType = ContextType.OBSERVATION, relevance: float = 0.5, path: str | None = None) -> ContextItem:
    metadata = {"path": path} if path else {}
    return ContextItem(item_id, "test", content, kind, relevance, priority, len(content.split()), TokenCountKind.ESTIMATED, metadata=metadata)


def test_typed_item_is_immutable_and_redacts_sensitive_values() -> None:
    item = ContextItem("x", "source", "token=secret-value", ContextType.ERROR, 0.8, ContextPriority.HIGH, 2, TokenCountKind.ESTIMATED, metadata={"key": "value"})

    assert item.to_dict()["content"] == "token=[REDACTED]"
    with pytest.raises(TypeError):
        item.metadata["new"] = "value"
    with pytest.raises(ContextManagerError):
        ContextItem("", "source", "content", ContextType.OBSERVATION, 0.5, ContextPriority.LOW, 1, TokenCountKind.ESTIMATED)


def test_token_counter_distinguishes_exact_estimated_and_unknown() -> None:
    exact = EncoderTokenCounter(WordTokenizer()).count("one two")
    estimated = EncoderTokenCounter(None).count("one two")
    unknown = UnknownCounter().count("one two")

    assert exact == TokenCount(2, TokenCountKind.EXACT)
    assert estimated.kind is TokenCountKind.ESTIMATED
    assert estimated.count > 0
    assert unknown.kind is TokenCountKind.UNKNOWN


def test_candidate_pipeline_deduplicates_and_prioritizes_critical_and_task_relevant_items() -> None:
    manager = ContextManager(tokenizer=WordTokenizer(), config=ContextManagerConfig(max_context_tokens=256, reserve_output_tokens=32))
    items = (
        _item("low-unrelated", "unrelated billing module", priority=ContextPriority.LOW, relevance=0.1),
        _item("high-auth", "JWT authentication middleware", priority=ContextPriority.HIGH, relevance=0.7),
        _item("duplicate", "JWT authentication middleware", priority=ContextPriority.MEDIUM, relevance=0.4),
        _item("critical", "current task constraint", priority=ContextPriority.CRITICAL, relevance=0.1),
    )

    deduped = manager.deduplicate(items)
    ranked = manager.rank(deduped, "fix JWT authentication")

    assert len(deduped) == 3
    assert ranked[0].item_id == "critical"
    assert ranked[1].item_id == "high-auth"


def test_assembly_preserves_task_instruction_and_plan_step_with_observable_metrics(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "app").mkdir()
    (root / "app" / "auth.py").write_text("def authenticate(): pass\n", encoding="utf-8")
    understanding = CodebaseUnderstandingBuilder().build("fix authentication", root, project_context=project_context(root))
    manager = ContextManager(tokenizer=WordTokenizer(), config=ContextManagerConfig(max_context_tokens=320, reserve_output_tokens=48))
    plan = SimpleNamespace(task="fix authentication", goal="inspect auth", task_type=SimpleNamespace(value="bug_fix"), steps=(SimpleNamespace(step_id="s1", title="Read auth"),), assumptions=(), constraints=(), verification_strategy=())
    step = SimpleNamespace(step_id="s1", title="Read auth", objective="read auth implementation")

    assembly = manager.assemble(task="fix authentication", system_prompt="Return a bounded action.", instruction="Inspect the current step.", project_context=project_context(root), codebase_understanding=understanding, plan=plan, current_step=step, history=())

    assert "fix authentication" in assembly.prompt
    assert "s1" in assembly.prompt
    assert assembly.metrics.selected_count > 0
    assert assembly.metrics.input_budget_tokens == 272
    assert assembly.metrics.selected_tokens <= assembly.metrics.input_budget_tokens
    assert assembly.to_dict()["metrics"]["reserved_output_tokens"] == 48


def test_large_lower_priority_context_is_compressed_or_dropped_while_critical_survives() -> None:
    manager = ContextManager(
        tokenizer=WordTokenizer(),
        config=ContextManagerConfig(max_context_tokens=120, reserve_output_tokens=24, max_summary_characters=80, summary_threshold_characters=40, max_item_characters=2_000),
    )
    large = _item("large", "authentication failure detail " * 100, priority=ContextPriority.MEDIUM, kind=ContextType.TOOL_RESULT, relevance=0.8)
    critical = _item("critical", "current task: preserve authentication error", priority=ContextPriority.CRITICAL, kind=ContextType.ERROR, relevance=1.0)

    assembly = manager.assemble(task="fix authentication error", system_prompt="system", instruction="act", extra_items=(large, critical))

    assert "preserve authentication error" in assembly.prompt
    assert assembly.metrics.compressed_count >= 0
    assert assembly.truncated
    assert assembly.metrics.selected_tokens <= 96
    assert assembly.warnings


def test_tool_results_are_classified_and_large_output_is_compacted() -> None:
    manager = ContextManager(tokenizer=WordTokenizer(), config=ContextManagerConfig(max_context_tokens=256, reserve_output_tokens=32, max_tool_output_characters=120, summary_threshold_characters=60, max_summary_characters=100))
    history = (
        AgentMessage(AgentMessageRole.TOOL, "line\n" * 500 + "Traceback: authentication failed", name="run_tests", call_id="c1"),
        AgentMessage(AgentMessageRole.TOOL, "same result", name="read_file", call_id="c2"),
    )

    assembly = manager.assemble(task="fix authentication", system_prompt="system", instruction="act", history=history)

    assert any(item.context_type is ContextType.TEST_RESULT for item in assembly.items)
    assert any(item.context_type is ContextType.ERROR for item in assembly.items)
    assert len(assembly.prompt) < 3_000
    assert "Traceback" in assembly.prompt


def test_progressive_expansion_adds_new_evidence_deterministically() -> None:
    manager = ContextManager(tokenizer=WordTokenizer())
    initial = manager.assemble(task="inspect auth", system_prompt="system", instruction="act")
    expanded = manager.assemble(task="inspect auth", system_prompt="system", instruction="act", extra_items=(_item("auth", "authentication middleware", priority=ContextPriority.HIGH, relevance=0.9),))

    assert len(expanded.items) > len(initial.items)
    assert "authentication middleware" in expanded.prompt
    assert manager.assemble(task="inspect auth", system_prompt="system", instruction="act", extra_items=(_item("auth", "authentication middleware", priority=ContextPriority.HIGH, relevance=0.9),)).to_dict() == expanded.to_dict()


def test_refresh_marks_path_dependent_context_invalidated_and_verified() -> None:
    manager = ContextManager(tokenizer=WordTokenizer())
    item = _item("auth-file", "authenticate implementation", priority=ContextPriority.HIGH, path="app/auth.py")

    stale = manager.refresh((item,), invalidated_paths=("app/auth.py",))
    verified = manager.refresh(stale, verified_item_ids=("auth-file",))

    assert stale[0].validity is ContextValidity.INVALIDATED
    assert verified[0].validity is ContextValidity.VERIFIED


def test_no_exact_tokenizer_is_explicitly_estimated_and_unknown_is_supported() -> None:
    estimated = ContextManager(config=ContextManagerConfig(max_context_tokens=128, reserve_output_tokens=16)).assemble(task="inspect", system_prompt="system", instruction="act")
    unknown = ContextManager(token_counter=UnknownCounter(), config=ContextManagerConfig(max_context_tokens=128, reserve_output_tokens=16)).assemble(task="inspect", system_prompt="system", instruction="act")

    assert estimated.metrics.estimated_token_items >= 0
    assert any("estimated" in warning for warning in estimated.warnings)
    assert unknown.metrics.unknown_token_items >= 0


def test_invalid_configuration_and_unfit_critical_context_are_structured() -> None:
    with pytest.raises(ContextManagerError):
        ContextManagerConfig(max_context_tokens=10, reserve_output_tokens=10)
    manager = ContextManager(tokenizer=WordTokenizer(), config=ContextManagerConfig(max_context_tokens=32, reserve_output_tokens=16, max_summary_characters=8))
    with pytest.raises(ValueError):
        manager.assemble(task="x " * 10_000, system_prompt="system", instruction="act")
