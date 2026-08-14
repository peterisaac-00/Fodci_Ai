"""Deterministic context budgeting for the 256-token Fodci window."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

from backend_ai.agent.models import AgentConfig, AgentMessage
from backend_ai.tools.project_context import ProjectContext


class TokenEncoder(Protocol):
    def encode(self, text: str) -> list[int]:
        """Encode text into tokenizer IDs for budget estimation."""


class ContextBudgetError(ValueError):
    """Raised when the required task itself cannot fit the configured budget."""


@dataclass(frozen=True, slots=True)
class BudgetedPrompt:
    """Rendered prompt and explicit budget metadata."""

    prompt: str
    token_count: int
    truncated: bool
    warnings: tuple[str, ...]


class ContextBudget:
    """Build compact prompts without unbounded history or tool-result growth."""

    def __init__(self, tokenizer: TokenEncoder, config: AgentConfig) -> None:
        self._tokenizer = tokenizer
        self._config = config
        self._budget = config.max_context_tokens - config.reserve_response_tokens

    def render(
        self,
        task: str,
        project_context: ProjectContext,
        history: tuple[AgentMessage, ...] = (),
    ) -> BudgetedPrompt:
        """Render a prompt, shrinking optional information in deterministic stages."""

        if not isinstance(task, str) or not task.strip():
            raise ContextBudgetError("Agent task must not be empty.")
        warnings: list[str] = []
        context_limits = (8, 3, 0)
        history_items = list(history[-self._config.max_history_items:])
        tool_chars = self._config.max_tool_result_chars
        for context_limit in context_limits:
            for history_limit in range(len(history_items), -1, -1):
                selected_history = history_items[-history_limit:] if history_limit else []
                prompt = self._render_raw(
                    task,
                    project_context,
                    selected_history,
                    context_limit=context_limit,
                    tool_chars=tool_chars,
                )
                token_count = self._count(prompt)
                if token_count <= self._budget:
                    if context_limit < 8:
                        warnings.append("Optional project context was compacted to fit the model context budget.")
                    if history_limit < len(history_items):
                        warnings.append("Oldest Agent history items were dropped to fit the model context budget.")
                    return BudgetedPrompt(prompt, token_count, bool(warnings), tuple(dict.fromkeys(warnings)))
            tool_chars = max(128, tool_chars // 2)
            warnings.append("Tool-result text was bounded more tightly to fit the model context budget.")

        minimal = self._render_raw(task, project_context, [], context_limit=0, tool_chars=128)
        token_count = self._count(minimal)
        if token_count > self._budget:
            raise ContextBudgetError(
                f"Task and required context need {token_count} tokens but the available Agent budget is {self._budget}."
            )
        warnings.append("Optional project and tool context was omitted to fit the model context budget.")
        return BudgetedPrompt(minimal, token_count, True, tuple(dict.fromkeys(warnings)))

    def _render_raw(
        self,
        task: str,
        project_context: ProjectContext,
        history: list[AgentMessage],
        *,
        context_limit: int,
        tool_chars: int,
    ) -> str:
        context = _compact_project_context(project_context, context_limit)
        lines = [
            "S:",
            self._config.system_prompt,
            "T:",
            task.strip(),
            "C:",
            json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ]
        if history:
            lines.extend(("", "### Agent History"))
            for message in history:
                content = message.content
                if message.role.value == "tool":
                    content = _bounded_text(content, tool_chars)
                lines.extend((f"[{message.role.value}]", content))
        return "\n".join(lines)

    def _count(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))


def _compact_project_context(context: ProjectContext, limit: int) -> dict[str, Any]:
    if limit == 0:
        compact: dict[str, Any] = {
            "type": context.project_type,
            "stack": context.stack_summary,
            "confidence": context.confidence,
            "complete": context.completeness == "complete",
        }
        if context.truncated:
            compact["cut"] = True
        return compact
    return {
        "project_type": context.project_type,
        "stack_summary": context.stack_summary,
        "languages": [item.name for item in context.languages[:limit]],
        "frameworks": [item.name for item in context.frameworks[:limit]],
        "databases": [item.name for item in context.databases[:limit]],
        "test_frameworks": [item.name for item in context.test_frameworks[:limit]],
        "infrastructure": [item.name for item in context.infrastructure[:limit]],
        "source_directories": list(context.source_directories[:limit]),
        "test_directories": list(context.test_directories[:limit]),
        "documentation_directories": list(context.documentation_directories[:limit]),
        "important_files": list(context.important_files[:limit]),
        "entry_points": [item.name for item in context.entry_points[:limit]],
        "project_files": list(context.project_files[:limit]),
        "confidence": context.confidence,
        "completeness": context.completeness,
        "truncated": context.truncated,
        "warnings": list(context.warnings[:limit]),
        "marker": "project_context_compacted" if len(context.project_files) > limit else None,
    }


def _bounded_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = f"\n[tool_result_truncated: kept_first_{max_chars}_chars]"
    return text[: max(0, max_chars - len(marker))] + marker


__all__ = ["BudgetedPrompt", "ContextBudget", "ContextBudgetError", "TokenEncoder"]
