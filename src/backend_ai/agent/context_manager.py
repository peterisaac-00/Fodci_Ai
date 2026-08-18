"""Phase 12.3 bounded context management for the autonomous Agent loop.

The manager keeps large repository/task information outside the model prompt until
it is selected, compressed, and packed into an explicit input budget. It is local,
deterministic, model-independent, and stores facts/observations rather than private
reasoning traces.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Protocol, TYPE_CHECKING

from backend_ai.agent.budget import ContextBudgetError
from backend_ai.agent.codebase_understanding import CodebaseUnderstanding
from backend_ai.agent.memory_retrieval import MemoryRetrievalItem, MemoryRetrievalResult
from backend_ai.agent.models import AgentMessage, AgentMessageRole
from backend_ai.agent.long_term_memory import LongTermMemoryEntry
from backend_ai.tools.project_context import ProjectContext

if TYPE_CHECKING:
    from backend_ai.agent.planner import ExecutionPlan, PlanStep
    from backend_ai.agent.selector import ToolSelectionDecision


class ContextManagerError(ValueError):
    """Invalid context-manager input or bound."""


class ContextType(str, Enum):
    USER_TASK = "user_task"
    INSTRUCTION = "instruction"
    PLAN = "plan"
    PLAN_STEP = "plan_step"
    CODEBASE_SUMMARY = "codebase_summary"
    ARCHITECTURE_SUMMARY = "architecture_summary"
    FILE_SUMMARY = "file_summary"
    FILE_CONTENT = "file_content"
    CODE_SEARCH = "code_search"
    SYMBOL = "symbol"
    DEPENDENCY = "dependency"
    TOOL_RESULT = "tool_result"
    TEST_RESULT = "test_result"
    ERROR = "error"
    MEMORY = "memory"
    DECISION = "decision"
    OBSERVATION = "observation"
    VERIFICATION = "verification"


class ContextPriority(IntEnum):
    DISCARDABLE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class CompressionStatus(str, Enum):
    ORIGINAL = "original"
    SUMMARIZED = "summarized"
    TRUNCATED = "truncated"


class ContextValidity(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    INVALIDATED = "invalidated"
    VERIFIED = "verified"


class TokenCountKind(str, Enum):
    EXACT = "exact"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TokenCount:
    count: int
    kind: TokenCountKind

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 0:
            raise ContextManagerError("token count must be a non-negative integer")
        if not isinstance(self.kind, TokenCountKind):
            raise ContextManagerError("token count kind must be TokenCountKind")

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "kind": self.kind.value}


class TokenCounter(Protocol):
    def count(self, text: str) -> TokenCount:
        """Count text exactly or conservatively, with explicit provenance."""


class EncoderTokenCounter:
    """Adapter for a tokenizer exposing encode(), with conservative fallback."""

    def __init__(self, encoder: Any | None = None) -> None:
        self.encoder = encoder

    def count(self, text: str) -> TokenCount:
        if self.encoder is not None and callable(getattr(self.encoder, "encode", None)):
            try:
                encoded = self.encoder.encode(text)
                return TokenCount(len(encoded), TokenCountKind.EXACT)
            except Exception:
                pass
        if not isinstance(text, str):
            return TokenCount(0, TokenCountKind.UNKNOWN)
        if not text:
            return TokenCount(0, TokenCountKind.ESTIMATED)
        return TokenCount(max(1, (len(text.encode("utf-8")) + 3) // 4), TokenCountKind.ESTIMATED)


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One bounded, provenance-bearing candidate for active model context."""

    item_id: str
    source: str
    content: str
    context_type: ContextType
    relevance: float
    priority: ContextPriority
    token_cost: int
    token_count_kind: TokenCountKind
    recency: int = 0
    dependencies: tuple[str, ...] = ()
    compression: CompressionStatus = CompressionStatus.ORIGINAL
    validity: ContextValidity = ContextValidity.FRESH
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        for value, name in ((self.item_id, "item_id"), (self.source, "source"), (self.content, "content")):
            if not isinstance(value, str) or not value.strip():
                raise ContextManagerError(f"{name} must contain text")
        if not isinstance(self.context_type, ContextType):
            raise ContextManagerError("context_type must be ContextType")
        if not isinstance(self.priority, ContextPriority):
            raise ContextManagerError("priority must be ContextPriority")
        if not isinstance(self.relevance, (int, float)) or isinstance(self.relevance, bool) or not 0.0 <= float(self.relevance) <= 1.0:
            raise ContextManagerError("relevance must be between 0 and 1")
        if not isinstance(self.token_cost, int) or isinstance(self.token_cost, bool) or self.token_cost < 0:
            raise ContextManagerError("token_cost must be a non-negative integer")
        if not isinstance(self.token_count_kind, TokenCountKind):
            raise ContextManagerError("token_count_kind must be TokenCountKind")
        if not isinstance(self.recency, int) or isinstance(self.recency, bool) or self.recency < 0:
            raise ContextManagerError("recency must be a non-negative integer")
        if not isinstance(self.compression, CompressionStatus) or not isinstance(self.validity, ContextValidity):
            raise ContextManagerError("invalid compression or validity state")
        object.__setattr__(self, "dependencies", tuple(dict.fromkeys(self.dependencies)))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def normalized_content(self) -> str:
        return " ".join(re.findall(r"\w+", self.content.casefold(), flags=re.UNICODE))

    @property
    def score(self) -> tuple[int, float, int, int, str]:
        return (int(self.priority), float(self.relevance), self.recency, -self.token_cost, self.item_id)

    def with_token_count(self, counter: TokenCounter) -> "ContextItem":
        counted = counter.count(self.content)
        return replace(self, token_cost=counted.count, token_count_kind=counted.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source": self.source,
            "content": _redact(self.content),
            "context_type": self.context_type.value,
            "relevance": round(float(self.relevance), 6),
            "priority": self.priority.name.casefold(),
            "token_cost": self.token_cost,
            "token_count_kind": self.token_count_kind.value,
            "recency": self.recency,
            "dependencies": list(self.dependencies),
            "compression": self.compression.value,
            "validity": self.validity.value,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ContextManagerConfig:
    """Model-independent finite controls for active context assembly."""

    max_context_tokens: int = 2_048
    reserve_output_tokens: int = 32
    max_items: int = 64
    max_item_characters: int = 8_192
    max_tool_output_characters: int = 4_000
    summary_threshold_characters: int = 1_600
    max_summary_characters: int = 1_200
    relevance_threshold: float = 0.0

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("max_context_tokens", 262_144),
            ("reserve_output_tokens", 131_072),
            ("max_items", 512),
            ("max_item_characters", 1_048_576),
            ("max_tool_output_characters", 262_144),
            ("summary_threshold_characters", 262_144),
            ("max_summary_characters", 262_144),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > ceiling:
                raise ContextManagerError(f"{name} is outside its configured bound")
        if self.max_context_tokens <= self.reserve_output_tokens:
            raise ContextManagerError("max_context_tokens must exceed reserve_output_tokens")
        if not isinstance(self.relevance_threshold, (int, float)) or isinstance(self.relevance_threshold, bool) or not 0.0 <= float(self.relevance_threshold) <= 1.0:
            raise ContextManagerError("relevance_threshold must be between 0 and 1")

    @property
    def input_budget_tokens(self) -> int:
        return self.max_context_tokens - self.reserve_output_tokens


@dataclass(frozen=True, slots=True)
class ContextMetrics:
    candidate_count: int
    deduplicated_count: int
    selected_count: int
    compressed_count: int
    dropped_count: int
    stale_count: int
    exact_token_items: int
    estimated_token_items: int
    unknown_token_items: int
    selected_tokens: int
    input_budget_tokens: int
    reserved_output_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "candidate_count": self.candidate_count,
            "deduplicated_count": self.deduplicated_count,
            "selected_count": self.selected_count,
            "compressed_count": self.compressed_count,
            "dropped_count": self.dropped_count,
            "stale_count": self.stale_count,
            "exact_token_items": self.exact_token_items,
            "estimated_token_items": self.estimated_token_items,
            "unknown_token_items": self.unknown_token_items,
            "selected_tokens": self.selected_tokens,
            "input_budget_tokens": self.input_budget_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
        }


@dataclass(frozen=True, slots=True)
class ContextAssembly:
    prompt: str
    items: tuple[ContextItem, ...]
    metrics: ContextMetrics
    truncated: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "warnings", tuple(dict.fromkeys(self.warnings)))

    @property
    def token_count(self) -> int:
        return self.metrics.selected_tokens

    @property
    def selected_item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": _redact(self.prompt),
            "items": [item.to_dict() for item in self.items],
            "metrics": self.metrics.to_dict(),
            "truncated": self.truncated,
            "warnings": list(self.warnings),
        }


class ContextManager:
    """Controlled context pipeline: collect, deduplicate, rank, compress, pack, validate."""

    def __init__(self, *, tokenizer: Any | None = None, config: ContextManagerConfig | None = None, token_counter: TokenCounter | None = None) -> None:
        if token_counter is not None and tokenizer is not None:
            raise ContextManagerError("provide tokenizer or token_counter, not both")
        self.config = config or ContextManagerConfig()
        self.counter = token_counter or EncoderTokenCounter(tokenizer)

    def assemble(
        self,
        *,
        task: str,
        system_prompt: str,
        instruction: str,
        project_context: ProjectContext | None = None,
        codebase_understanding: CodebaseUnderstanding | None = None,
        plan: Any | None = None,
        current_step: Any | None = None,
        selection: Any | None = None,
        history: Sequence[AgentMessage] = (),
        long_term_memories: Sequence[LongTermMemoryEntry] = (),
        retrieval_result: MemoryRetrievalResult | None = None,
        extra_items: Sequence[ContextItem] = (),
        invalidated_paths: Iterable[str] = (),
    ) -> ContextAssembly:
        if not isinstance(task, str) or not task.strip():
            raise ContextManagerError("task must contain text")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ContextManagerError("system_prompt must contain text")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ContextManagerError("instruction must contain text")
        invalidated = frozenset(_normal_path(path) for path in invalidated_paths if str(path).strip())
        candidates = self.collect_candidates(task=task, instruction=instruction, project_context=project_context, codebase_understanding=codebase_understanding, plan=plan, current_step=current_step, selection=selection, history=history, long_term_memories=long_term_memories, retrieval_result=retrieval_result, extra_items=extra_items, invalidated_paths=invalidated)
        deduped = self.deduplicate(candidates)
        ranked = self.rank(deduped, task)
        selected, dropped, compressed, warnings = self.pack(ranked, task=task, system_prompt=system_prompt, instruction=instruction, invalidated_paths=invalidated)
        prompt = self._render_prompt(system_prompt, task, instruction, selected)
        counted_prompt = self.counter.count(prompt)
        if counted_prompt.count > self.config.input_budget_tokens:
            raise ContextBudgetError(f"Context assembly needs {counted_prompt.count} tokens but the available input budget is {self.config.input_budget_tokens}.")
        stale_count = sum(item.validity in {ContextValidity.STALE, ContextValidity.INVALIDATED} for item in selected)
        exact = sum(item.token_count_kind is TokenCountKind.EXACT for item in selected)
        estimated = sum(item.token_count_kind is TokenCountKind.ESTIMATED for item in selected)
        unknown = sum(item.token_count_kind is TokenCountKind.UNKNOWN for item in selected)
        metrics = ContextMetrics(len(candidates), len(deduped), len(selected), len(compressed), len(dropped), stale_count, exact, estimated, unknown, counted_prompt.count, self.config.input_budget_tokens, self.config.reserve_output_tokens)
        final_warnings = list(warnings)
        if counted_prompt.kind is TokenCountKind.ESTIMATED:
            final_warnings.append("Prompt token usage is conservatively estimated because no exact tokenizer was available.")
        return ContextAssembly(prompt, tuple(selected), metrics, bool(dropped or compressed), tuple(final_warnings))

    def collect_candidates(self, *, task: str, instruction: str, project_context: ProjectContext | None, codebase_understanding: CodebaseUnderstanding | None, plan: Any | None, current_step: Any | None, selection: Any | None, history: Sequence[AgentMessage], long_term_memories: Sequence[LongTermMemoryEntry], retrieval_result: MemoryRetrievalResult | None, extra_items: Sequence[ContextItem], invalidated_paths: frozenset[str] = frozenset()) -> tuple[ContextItem, ...]:
        items: list[ContextItem] = []
        items.append(self._item("task", "user", task, ContextType.USER_TASK, ContextPriority.CRITICAL, 1, metadata={"authoritative": True}))
        items.append(self._item("instruction", "loop", instruction, ContextType.INSTRUCTION, ContextPriority.CRITICAL, 1))
        if project_context is not None:
            items.append(self._item("project-context", "project_context", _project_context_text(project_context), ContextType.CODEBASE_SUMMARY, ContextPriority.HIGH, 1, metadata={"root": str(project_context.root)}))
        if codebase_understanding is not None:
            items.append(self._item("codebase-understanding", "codebase_understanding", codebase_understanding.compact_summary(), ContextType.CODEBASE_SUMMARY, ContextPriority.HIGH, 2, metadata={"root": str(codebase_understanding.root), "completeness": codebase_understanding.completeness.value, "confidence": codebase_understanding.confidence.value}))
            architecture = "; ".join(f"{layer.name}: {', '.join(layer.paths[:6])}" for layer in codebase_understanding.architecture[:8])
            if architecture:
                items.append(self._item("codebase-architecture", "codebase_understanding", architecture, ContextType.ARCHITECTURE_SUMMARY, ContextPriority.HIGH, 2, metadata={"root": str(codebase_understanding.root)}))
            for index, relevant in enumerate(codebase_understanding.relevant_files[: self.config.max_items]):
                items.append(self._item(f"relevant-file-{index:04d}", "codebase_understanding", f"{relevant.path} [{relevant.role}; {relevant.relevance}] — {'; '.join(relevant.reasons[:3])}", ContextType.FILE_SUMMARY, ContextPriority.HIGH if relevant.relevance == "high" else ContextPriority.MEDIUM, 2, metadata={"path": relevant.path, "root": str(codebase_understanding.root)}))
            for index, symbol in enumerate(codebase_understanding.symbols[: min(32, self.config.max_items)]):
                items.append(self._item(f"symbol-{index:04d}", "codebase_understanding", f"{symbol.name} ({symbol.kind}) in {symbol.path}:{symbol.line_start}", ContextType.SYMBOL, ContextPriority.MEDIUM, 2, metadata={"path": symbol.path, "symbol": symbol.name, "root": str(codebase_understanding.root)}))
        if plan is not None:
            items.append(self._item("plan", "planner", _plan_text(plan), ContextType.PLAN, ContextPriority.HIGH, 3))
        if current_step is not None:
            items.append(self._item("plan-step", "planner", _object_text(current_step), ContextType.PLAN_STEP, ContextPriority.CRITICAL, 4))
        if selection is not None:
            items.append(self._item("selection", "selector", _object_text(selection), ContextType.DECISION, ContextPriority.HIGH, 4))
        for index, memory in enumerate(long_term_memories[: self.config.max_items]):
            content = getattr(memory, "content", str(memory))
            entry_id = getattr(memory, "entry_id", f"memory-{index:04d}")
            items.append(self._item(f"long-term-{entry_id}", "long_term_memory", content, ContextType.MEMORY, ContextPriority.HIGH, index, metadata={"memory_id": entry_id}))
        if retrieval_result is not None:
            if retrieval_result.context:
                items.append(self._item("memory-retrieval-context", "memory_retrieval", retrieval_result.context, ContextType.MEMORY, ContextPriority.HIGH, 0, metadata={"retrieval": True, "item_count": len(retrieval_result.items)}))
            for index, item in enumerate(retrieval_result.items[: self.config.max_items]):
                items.append(self._item(f"retrieval-{item.memory_id}", item.source.value, item.content, ContextType.MEMORY, ContextPriority.MEDIUM, index, metadata={"memory_id": item.memory_id, "source": item.source.value, "relevance": item.relevance_score}))
        for index, message in enumerate(history[-self.config.max_items:]):
            item_type, priority, content, metadata = _history_item(message, self.config)
            items.append(self._item(f"history-{index:04d}-{message.call_id or message.role.value}", message.name or message.role.value, content, item_type, priority, index + 1, metadata=metadata))
        items.extend(extra_items)
        expanded: list[ContextItem] = []
        for item in items:
            expanded.append(item)
            if item.context_type is ContextType.TEST_RESULT and (item.metadata.get("contains_error") is True or _contains_error_evidence(item.content)):
                expanded.append(replace(item, item_id=f"{item.item_id}-error", context_type=ContextType.ERROR, priority=ContextPriority.HIGH, metadata={**_thaw(item.metadata), "derived_from": item.item_id}))
        return tuple(self._refresh_item(item, invalidated_paths) for item in expanded)

    def deduplicate(self, items: Sequence[ContextItem]) -> tuple[ContextItem, ...]:
        by_content: dict[tuple[ContextType, str], ContextItem] = {}
        for item in items:
            key = (item.context_type, item.normalized_content)
            prior = by_content.get(key)
            if prior is None or item.score > prior.score:
                by_content[key] = item
        return tuple(sorted(by_content.values(), key=lambda item: item.item_id))

    def rank(self, items: Sequence[ContextItem], task: str) -> tuple[ContextItem, ...]:
        task_tokens = _tokens(task)
        ranked: list[ContextItem] = []
        for item in items:
            overlap = len(task_tokens & _tokens(item.content)) / max(1, len(task_tokens))
            relevance = min(1.0, float(item.relevance) * 0.65 + overlap * 0.35)
            if item.priority is ContextPriority.CRITICAL:
                relevance = max(relevance, 0.75)
            ranked.append(replace(item, relevance=relevance))
        ranked.sort(key=lambda item: (item.priority, item.relevance, item.recency, -item.token_cost, item.item_id), reverse=True)
        return tuple(ranked)

    def pack(self, ranked: Sequence[ContextItem], *, task: str, system_prompt: str, instruction: str, invalidated_paths: frozenset[str]) -> tuple[tuple[ContextItem, ...], tuple[ContextItem, ...], tuple[ContextItem, ...], tuple[str, ...]]:
        selected: list[ContextItem] = []
        dropped: list[ContextItem] = []
        compressed: list[ContextItem] = []
        warnings: list[str] = []
        critical = [item for item in ranked if item.priority is ContextPriority.CRITICAL]
        optional = [item for item in ranked if item.priority is not ContextPriority.CRITICAL]
        ordered = critical + optional
        for item in ordered:
            candidate = item if item.context_type in {ContextType.USER_TASK, ContextType.INSTRUCTION} else self._compress_if_needed(item, task)
            if candidate.compression is not CompressionStatus.ORIGINAL:
                compressed.append(candidate)
            candidate = candidate.with_token_count(self.counter)
            if candidate.validity is ContextValidity.INVALIDATED and candidate.priority is not ContextPriority.CRITICAL:
                dropped.append(candidate)
                continue
            if candidate.relevance < self.config.relevance_threshold and candidate.priority < ContextPriority.HIGH:
                dropped.append(candidate)
                continue
            trial = tuple(selected + [candidate])
            prompt = self._render_prompt(system_prompt, task, instruction, trial)
            if self.counter.count(prompt).count <= self.config.input_budget_tokens and len(selected) < self.config.max_items:
                selected.append(candidate)
            elif item.priority is ContextPriority.CRITICAL:
                if item.context_type in {ContextType.USER_TASK, ContextType.INSTRUCTION}:
                    raise ContextBudgetError(f"Critical context item {item.item_id} cannot fit the configured context budget without compressing required task instructions.")
                emergency = self._compress_item(candidate, self.config.max_summary_characters)
                emergency = emergency.with_token_count(self.counter)
                trial = tuple(selected + [emergency])
                if self.counter.count(self._render_prompt(system_prompt, task, instruction, trial)).count <= self.config.input_budget_tokens:
                    selected.append(emergency)
                    if emergency.compression not in {CompressionStatus.ORIGINAL}:
                        compressed.append(emergency)
                    warnings.append(f"Critical context item {item.item_id} was compressed to preserve it.")
                else:
                    raise ContextBudgetError(f"Critical context item {item.item_id} cannot fit the configured context budget.")
            else:
                dropped.append(candidate)
        if dropped:
            warnings.append(f"{len(dropped)} lower-priority context item(s) were omitted from the active prompt budget.")
        if compressed:
            warnings.append(f"{len(compressed)} context item(s) were compressed deterministically.")
        stale = sum(item.validity in {ContextValidity.STALE, ContextValidity.INVALIDATED} for item in selected)
        if stale:
            warnings.append(f"{stale} selected context item(s) are stale and require refreshed repository evidence.")
        return tuple(selected), tuple(dropped), tuple(compressed), tuple(warnings)

    def refresh(self, items: Sequence[ContextItem], *, invalidated_paths: Iterable[str] = (), verified_item_ids: Iterable[str] = ()) -> tuple[ContextItem, ...]:
        invalidated = frozenset(_normal_path(path) for path in invalidated_paths if str(path).strip())
        verified = frozenset(str(item_id) for item_id in verified_item_ids)
        return tuple(self._refresh_item(item, invalidated, verified) for item in items)

    def _item(self, item_id: str, source: str, content: str, context_type: ContextType, priority: ContextPriority, recency: int, *, metadata: Mapping[str, Any] | None = None) -> ContextItem:
        clean = _bounded(_redact(str(content)), self.config.max_item_characters)
        counted = self.counter.count(clean)
        return ContextItem(item_id, source, clean, context_type, 0.5, priority, counted.count, counted.kind, recency, metadata=metadata or {})

    def _compress_if_needed(self, item: ContextItem, task: str) -> ContextItem:
        if len(item.content) <= self.config.summary_threshold_characters:
            return item
        limit = self.config.max_tool_output_characters if item.context_type in {ContextType.TOOL_RESULT, ContextType.TEST_RESULT, ContextType.ERROR} else self.config.max_summary_characters
        return self._compress_item(item, limit, task=task)

    def _compress_item(self, item: ContextItem, limit: int, *, task: str = "") -> ContextItem:
        if len(item.content) <= limit:
            return item
        lines = item.content.splitlines() or [item.content]
        tokens = _tokens(task)
        relevant_lines = [line for line in lines if tokens & _tokens(line)]
        important_lines = [line for line in lines if any(marker in line.casefold() for marker in ("error", "fail", "traceback", "exception", "exit", "status", "test", "warning"))]
        chosen: list[str] = []
        fallback_lines = () if relevant_lines or important_lines else tuple(lines[:4] + lines[-2:])
        for line in relevant_lines + important_lines + list(fallback_lines):
            if line not in chosen:
                chosen.append(line)
        text = "\n".join(chosen) or item.content
        text = _bounded_tool_output(text, limit)
        marker = f"\n[context_compressed: original_chars={len(item.content)}]"
        text = _bounded(text, max(1, limit - len(marker))) + marker
        return replace(item, content=text, compression=CompressionStatus.SUMMARIZED if relevant_lines or important_lines else CompressionStatus.TRUNCATED, metadata={**_thaw(item.metadata), "original_characters": len(item.content)})

    def _refresh_item(self, item: ContextItem, invalidated_paths: frozenset[str], verified_item_ids: frozenset[str] = frozenset()) -> ContextItem:
        if item.item_id in verified_item_ids:
            return replace(item, validity=ContextValidity.VERIFIED)
        paths = {_normal_path(str(item.metadata.get("path", "")))} if item.metadata.get("path") else set()
        paths.update(_normal_path(str(path)) for path in item.metadata.get("paths", ()) if str(path).strip())
        if paths & invalidated_paths:
            return replace(item, validity=ContextValidity.INVALIDATED)
        return item

    @staticmethod
    def _render_prompt(system_prompt: str, task: str, instruction: str, items: Sequence[ContextItem]) -> str:
        lines = ["S:", system_prompt, "T:", task.strip()]
        if instruction.strip():
            lines.extend(("I:", instruction.strip()))
        lines.append("C:")
        for item in items:
            if item.context_type is ContextType.USER_TASK or item.context_type is ContextType.INSTRUCTION:
                continue
            validity = f"; validity={item.validity.value}" if item.validity is not ContextValidity.FRESH else ""
            lines.extend((f"[{item.context_type.value}; id={item.item_id}; priority={item.priority.name.casefold()}; relevance={item.relevance:.3f}{validity}]", item.content))
        return "\n".join(lines)


def _history_item(message: AgentMessage, config: ContextManagerConfig) -> tuple[ContextType, ContextPriority, str, dict[str, Any]]:
    content = message.content
    if message.role is AgentMessageRole.TOOL:
        lowered = content.casefold()
        contains_error = _contains_error_evidence(content)
        kind = ContextType.TEST_RESULT if message.name in {"run_tests", "parse_test_result"} else ContextType.ERROR if contains_error else ContextType.TOOL_RESULT
        content = _bounded_tool_output(content, config.max_tool_output_characters)

        priority = ContextPriority.CRITICAL if kind is ContextType.ERROR else ContextPriority.HIGH if kind is ContextType.TEST_RESULT else ContextPriority.MEDIUM
        return kind, priority, content, {"call_id": message.call_id, "tool_name": message.name, "contains_error": contains_error}
    if message.role is AgentMessageRole.USER:
        return ContextType.OBSERVATION, ContextPriority.HIGH, _bounded(content, config.max_item_characters), {"role": message.role.value}
    return ContextType.OBSERVATION, ContextPriority.LOW, _bounded(content, config.max_item_characters), {"role": message.role.value}


def _project_context_text(context: ProjectContext) -> str:
    payload = {"project_type": context.project_type, "stack_summary": context.stack_summary, "languages": [item.name for item in context.languages[:8]], "frameworks": [item.name for item in context.frameworks[:8]], "databases": [item.name for item in context.databases[:8]], "source_directories": list(context.source_directories[:8]), "test_directories": list(context.test_directories[:8]), "important_files": list(context.important_files[:16]), "entry_points": [item.name for item in context.entry_points[:8]], "confidence": context.confidence, "completeness": context.completeness, "truncated": context.truncated}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _plan_text(plan: Any) -> str:
    payload = {"task": getattr(plan, "task", None), "goal": getattr(plan, "goal", None), "task_type": getattr(getattr(plan, "task_type", None), "value", getattr(plan, "task_type", None)), "steps": [_object_text(step) for step in getattr(plan, "steps", ())[:16]], "assumptions": list(getattr(plan, "assumptions", ())[:8]), "constraints": list(getattr(plan, "constraints", ())[:8]), "verification": list(getattr(plan, "verification_strategy", ())[:8])}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _object_text(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return json.dumps(_thaw(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _contains_error_evidence(value: str) -> bool:
    lowered = value.casefold()
    if re.search(r"\b(traceback|failed|failure|exception|integrityerror|assertionerror)\b", lowered):
        return True
    if '"success":false' in lowered or '"error":' in lowered and '"error":null' not in lowered:
        return True
    return False


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"\w+", value.casefold(), flags=re.UNICODE) if len(token) > 1}


def _normal_path(value: str) -> str:
    return str(value).replace("\\", "/").strip("/").casefold()


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = f"\n[context_truncated: kept_first_{limit}_chars]"
    return value[: max(0, limit - len(marker))] + marker


def _bounded_tool_output(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n[tool_output_middle_truncated]"
    tail_markers = ("error", "failed", "traceback", "exception", "exit", "status", "assert")
    tail_start = max(0, len(value) - max(64, limit // 3))
    tail = value[tail_start:]
    if not any(item in tail.casefold() for item in tail_markers):
        tail = ""
    available = max(1, limit - len(marker) - len(tail))
    return value[:available] + marker + tail


def _redact(value: str) -> str:
    return re.sub(r"(?i)(password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)(\s*[=:]\s*)([^,\s}\]]+|\"[^\"]*\"|'[^']*')", r"\1\2[REDACTED]", value)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContextManagerError("metadata must be a mapping")
    return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


__all__ = [
    "CompressionStatus",
    "ContextAssembly",
    "ContextItem",
    "ContextManager",
    "ContextManagerConfig",
    "ContextManagerError",
    "ContextMetrics",
    "ContextPriority",
    "ContextType",
    "ContextValidity",
    "EncoderTokenCounter",
    "TokenCount",
    "TokenCountKind",
    "TokenCounter",
]
