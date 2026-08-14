"""The first bounded, read-only Fodci Agent loop."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from backend_ai.agent.budget import ContextBudget, ContextBudgetError
from backend_ai.agent.models import (
    AgentConfig,
    AgentMessage,
    AgentMessageRole,
    AgentResult,
    AgentStatus,
    AgentStep,
    AgentTask,
    AgentUsage,
    ToolCall,
    ToolResult,
)
from backend_ai.agent.protocol import (
    ActionParseError,
    ParsedFinalAnswer,
    parse_agent_output,
    tool_call_from_action,
)
from backend_ai.agent.registry import ToolRegistry, ToolRegistryError, UnknownToolError
from backend_ai.tools.base import ToolError, ToolErrorCode
from backend_ai.tools.project_context import ProjectContext


class AgentLoop:
    """Run a small deterministic investigation loop over registered read-only tools."""

    def __init__(
        self,
        engine: Any,
        *,
        registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        if not callable(getattr(engine, "generate", None)):
            raise TypeError("AgentLoop requires an inference engine with generate().")
        tokenizer = getattr(engine, "tokenizer", None)
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("AgentLoop requires an inference engine exposing tokenizer.encode().")
        self.engine = engine
        self.registry = registry or ToolRegistry.default()
        self.config = config or AgentConfig()
        self._budget = ContextBudget(tokenizer, self.config)

    def run(self, task: str, project_root: Path | str) -> AgentResult:
        """Execute one bounded task inside one explicit project root."""

        try:
            task_model = AgentTask(task=task, project_root=Path(project_root).expanduser().resolve(strict=False))
        except (TypeError, ValueError) as exc:
            return self._failed_result(AgentStatus.FAILED, str(exc), stop_reason="invalid_task")

        context_result, context_call, context_tool_result = self._build_context(task_model)
        if context_result is None:
            return self._failed_result(
                AgentStatus.TOOL_ERROR,
                context_tool_result.message or "Unable to build project context.",
                stop_reason="project_context_error",
                tool_calls=(context_call,) if context_call else (),
                tool_results=(context_tool_result,) if context_tool_result else (),
            )

        steps: list[AgentStep] = []
        calls: list[ToolCall] = [context_call] if context_call else []
        results: list[ToolResult] = [context_tool_result] if context_tool_result else []
        history: list[AgentMessage] = []
        warnings: list[str] = list(context_result.warnings)
        errors: list[str] = []
        prompt_tokens = 0
        context_truncations = 0
        tool_result_chars = 0
        last_output = ""

        for step_index in range(1, self.config.max_steps + 1):
            try:
                budgeted = self._budget.render(task_model.task, context_result, tuple(history))
            except ContextBudgetError as exc:
                errors.append(str(exc))
                return self._result(
                    final_answer="",
                    status=AgentStatus.CONTEXT_LIMIT,
                    steps=steps,
                    calls=calls,
                    results=results,
                    context=context_result,
                    stop_reason="context_limit",
                    usage=AgentUsage(
                        steps=len(steps),
                        tool_calls=len(calls),
                        prompt_tokens=prompt_tokens,
                        tool_result_chars=tool_result_chars,
                        context_truncations=context_truncations,
                    ),
                    warnings=warnings,
                    errors=errors,
                )
            prompt_tokens += budgeted.token_count
            if budgeted.truncated:
                context_truncations += 1
                warnings.extend(budgeted.warnings)
            try:
                model_result = self.engine.generate(budgeted.prompt)
                model_output = getattr(model_result, "generated_text", None)
                if not isinstance(model_output, str):
                    raise TypeError("inference result does not contain generated_text")
            except Exception as exc:
                errors.append(str(exc))
                return self._result(
                    final_answer="",
                    status=AgentStatus.INFERENCE_ERROR,
                    steps=steps,
                    calls=calls,
                    results=results,
                    context=context_result,
                    stop_reason="inference_error",
                    usage=AgentUsage(
                        steps=len(steps) + 1,
                        tool_calls=len(calls),
                        prompt_tokens=prompt_tokens,
                        tool_result_chars=tool_result_chars,
                        context_truncations=context_truncations,
                    ),
                    warnings=warnings,
                    errors=errors,
                )

            last_output = model_output
            try:
                parsed = parse_agent_output(model_output)
            except ActionParseError as exc:
                errors.append(exc.message)
                steps.append(
                    AgentStep(
                        index=step_index,
                        prompt_token_count=budgeted.token_count,
                        model_output=model_output,
                        context_truncated=budgeted.truncated,
                    )
                )
                return self._result(
                    final_answer="",
                    status=AgentStatus.INVALID_ACTION,
                    steps=steps,
                    calls=calls,
                    results=results,
                    context=context_result,
                    stop_reason="invalid_action",
                    usage=AgentUsage(
                        steps=len(steps),
                        tool_calls=len(calls),
                        prompt_tokens=prompt_tokens,
                        tool_result_chars=tool_result_chars,
                        context_truncations=context_truncations,
                    ),
                    warnings=warnings,
                    errors=errors,
                )

            if isinstance(parsed, ParsedFinalAnswer):
                steps.append(
                    AgentStep(
                        index=step_index,
                        prompt_token_count=budgeted.token_count,
                        model_output=model_output,
                        context_truncated=budgeted.truncated,
                    )
                )
                if not parsed.text:
                    warnings.append("Model returned an empty final answer.")
                return self._result(
                    final_answer=parsed.text,
                    status=AgentStatus.COMPLETED,
                    steps=steps,
                    calls=calls,
                    results=results,
                    context=context_result,
                    stop_reason="final_answer",
                    usage=AgentUsage(
                        steps=len(steps),
                        tool_calls=len(calls),
                        prompt_tokens=prompt_tokens,
                        tool_result_chars=tool_result_chars,
                        context_truncations=context_truncations,
                    ),
                    warnings=warnings,
                    errors=errors,
                )

            call_id = f"call-{len(calls) + 1:04d}"
            try:
                call = tool_call_from_action(parsed, call_id)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
                return self._result(
                    final_answer="",
                    status=AgentStatus.INVALID_ACTION,
                    steps=steps,
                    calls=calls,
                    results=results,
                    context=context_result,
                    stop_reason="invalid_action",
                    usage=AgentUsage(
                        steps=len(steps) + 1,
                        tool_calls=len(calls),
                        prompt_tokens=prompt_tokens,
                        tool_result_chars=tool_result_chars,
                        context_truncations=context_truncations,
                    ),
                    warnings=warnings,
                    errors=errors,
                )
            if len(calls) >= self.config.max_tool_calls:
                steps.append(
                    AgentStep(
                        index=step_index,
                        prompt_token_count=budgeted.token_count,
                        model_output=model_output,
                        tool_call=call,
                        context_truncated=budgeted.truncated,
                    )
                )
                return self._result(
                    final_answer="",
                    status=AgentStatus.MAX_TOOL_CALLS,
                    steps=steps,
                    calls=calls,
                    results=results,
                    context=context_result,
                    stop_reason="max_tool_calls",
                    usage=AgentUsage(
                        steps=len(steps),
                        tool_calls=len(calls),
                        prompt_tokens=prompt_tokens,
                        tool_result_chars=tool_result_chars,
                        context_truncations=context_truncations,
                    ),
                    warnings=warnings,
                    errors=errors,
                )

            try:
                call = self._bind_project_root(call, task_model.project_root)
            except ToolError as exc:
                calls.append(call)
                tool_result = ToolResult(
                    call_id=call.call_id,
                    tool_name=call.name,
                    success=False,
                    error_code=exc.code.value,
                    message=exc.message,
                )
                results.append(tool_result)
                serialized_result = json.dumps(tool_result.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                history.append(AgentMessage(
                    role=AgentMessageRole.ASSISTANT,
                    content=model_output,
                    call_id=call.call_id,
                ))
                history.append(AgentMessage(
                    role=AgentMessageRole.TOOL,
                    content=serialized_result,
                    name=call.name,
                    call_id=call.call_id,
                ))
                history = history[-self.config.max_history_items:]
                tool_result_chars += len(serialized_result)
                steps.append(
                    AgentStep(
                        index=step_index,
                        prompt_token_count=budgeted.token_count,
                        model_output=model_output,
                        tool_call=call,
                        tool_result=tool_result,
                        context_truncated=budgeted.truncated,
                    )
                )
                continue
            calls.append(call)
            tool_result = self._execute(call)
            results.append(tool_result)
            serialized_result = json.dumps(tool_result.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            bounded_serialized = serialized_result
            if len(bounded_serialized) > self.config.max_tool_result_chars:
                bounded_serialized = bounded_serialized[: self.config.max_tool_result_chars] + "\n[tool_result_truncated]"
                tool_result = ToolResult(
                    call_id=tool_result.call_id,
                    tool_name=tool_result.tool_name,
                    success=tool_result.success,
                    data={"serialized_result": bounded_serialized, "truncated": True},
                    error_code=tool_result.error_code,
                    message=tool_result.message,
                    truncated=True,
                )
                results[-1] = tool_result
            tool_result_chars += len(bounded_serialized)
            history.append(AgentMessage(
                role=AgentMessageRole.ASSISTANT,
                content=model_output,
                call_id=call.call_id,
            ))
            history.append(AgentMessage(
                role=AgentMessageRole.TOOL,
                content=bounded_serialized,
                name=call.name,
                call_id=call.call_id,
            ))
            history = history[-self.config.max_history_items:]
            steps.append(
                AgentStep(
                    index=step_index,
                    prompt_token_count=budgeted.token_count,
                    model_output=model_output,
                    tool_call=call,
                    tool_result=tool_result,
                    context_truncated=budgeted.truncated,
                )
            )

        return self._result(
            final_answer="",
            status=AgentStatus.MAX_STEPS,
            steps=steps,
            calls=calls,
            results=results,
            context=context_result,
            stop_reason="max_steps",
            usage=AgentUsage(
                steps=len(steps),
                tool_calls=len(calls),
                prompt_tokens=prompt_tokens,
                tool_result_chars=tool_result_chars,
                context_truncations=context_truncations,
            ),
            warnings=warnings,
            errors=errors,
        )

    def _build_context(self, task: AgentTask) -> tuple[ProjectContext | None, ToolCall | None, ToolResult | None]:
        call = ToolCall(
            call_id="call-0001",
            name="project_context",
            arguments={"project_root": str(task.project_root)},
        )
        result = self._execute(call)
        if result.success and isinstance(result.data, ProjectContext):
            return result.data, call, result
        return None, call, result

    def _bind_project_root(self, call: ToolCall, project_root: Path) -> ToolCall:
        arguments = dict(call.arguments)
        supplied_root = arguments.get("project_root")
        if supplied_root is not None and Path(str(supplied_root)).expanduser().resolve(strict=False) != project_root:
            raise ToolError(
                code=ToolErrorCode.PATH_OUTSIDE_ROOT,
                message="Agent tool calls must remain inside the explicit task project root.",
                path=project_root,
            )
        arguments["project_root"] = str(project_root)
        return ToolCall(call_id=call.call_id, name=call.name, arguments=arguments)

    def _execute(self, call: ToolCall) -> ToolResult:
        try:
            data = self.registry.dispatch(call.name, call.arguments)
            truncated = bool(getattr(data, "truncated", False))
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                success=True,
                data=data,
                truncated=truncated,
            )
        except UnknownToolError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                error_code="UNKNOWN_TOOL",
                message=str(exc),
            )
        except ToolError as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                error_code=exc.code.value,
                message=exc.message,
            )
        except (ToolRegistryError, ValueError, TypeError) as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                error_code="INVALID_ARGUMENT",
                message=str(exc),
            )
        except Exception as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                error_code="TOOL_ERROR",
                message=str(exc),
            )

    @staticmethod
    def _result(
        *,
        final_answer: str,
        status: AgentStatus,
        steps: list[AgentStep],
        calls: list[ToolCall],
        results: list[ToolResult],
        context: ProjectContext | None,
        stop_reason: str,
        usage: AgentUsage,
        warnings: list[str],
        errors: list[str],
    ) -> AgentResult:
        return AgentResult(
            final_answer=final_answer,
            status=status,
            steps=tuple(steps),
            tool_calls=tuple(calls),
            tool_results=tuple(results),
            project_context=context,
            stop_reason=stop_reason,
            usage=usage,
            warnings=tuple(dict.fromkeys(warnings)),
            errors=tuple(dict.fromkeys(errors)),
        )

    @staticmethod
    def _failed_result(status: AgentStatus, error: str, *, stop_reason: str, **kwargs: Any) -> AgentResult:
        return AgentResult(
            final_answer="",
            status=status,
            steps=kwargs.get("steps", ()),
            tool_calls=kwargs.get("tool_calls", ()),
            tool_results=kwargs.get("tool_results", ()),
            project_context=kwargs.get("project_context"),
            stop_reason=stop_reason,
            usage=kwargs.get("usage", AgentUsage()),
            warnings=kwargs.get("warnings", ()),
            errors=(error,),
        )


__all__ = ["AgentLoop"]
