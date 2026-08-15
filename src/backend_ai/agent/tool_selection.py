"""Deterministic declarative tool selection for Phase 6.2.

This module maps validated declarative plan steps to capabilities exposed by an
existing ToolRegistry. It never dispatches, executes, inspects files, or enables
tools. Phase 6.3 owns eventual execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from backend_ai.agent.planner import ExecutionPlan, PlanStep, PlanRiskLevel
from backend_ai.agent.registry import ToolRegistry
from backend_ai.tools.project_context import ProjectContext


class ToolCategory(str, Enum):
    READ_ONLY = "READ_ONLY"
    MUTATING = "MUTATING"
    EXECUTION = "EXECUTION"
    DESTRUCTIVE = "DESTRUCTIVE"
    UNKNOWN = "UNKNOWN"


class ToolSelectionStatus(str, Enum):
    SELECTED = "SELECTED"
    NO_SUITABLE_TOOL = "NO_SUITABLE_TOOL"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    AMBIGUOUS_SELECTION = "AMBIGUOUS_SELECTION"
    MISSING_PREREQUISITES = "MISSING_PREREQUISITES"
    INVALID_REQUEST = "INVALID_REQUEST"
    INCOMPLETE = "INCOMPLETE"


class ToolSelectionConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class ToolSelectionRisk(str, Enum):
    READ_ONLY = "READ_ONLY"
    MUTATING = "MUTATING"
    EXECUTION = "EXECUTION"
    DESTRUCTIVE = "DESTRUCTIVE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ToolSelectionConfig:
    """Conservative selection budgets."""

    max_candidates_per_step: int = 8
    max_alternatives: int = 3
    max_plan_steps: int = 32
    max_prerequisites: int = 8
    max_warnings: int = 12
    max_text_length: int = 1_024

    def __post_init__(self) -> None:
        for name in (
            "max_candidates_per_step",
            "max_alternatives",
            "max_plan_steps",
            "max_prerequisites",
            "max_warnings",
            "max_text_length",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_candidates_per_step > 64 or self.max_alternatives > 32 or self.max_plan_steps > 256:
            raise ValueError("selection budget exceeds the safety ceiling")


@dataclass(frozen=True, slots=True)
class ToolCapability:
    """Extensible capability description for one available registered tool."""

    tool_name: str
    category: ToolCategory
    description: str
    supported_intents: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("ToolCapability.tool_name must be non-empty")
        if not isinstance(self.category, ToolCategory):
            raise ValueError("ToolCapability.category must be ToolCategory")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("ToolCapability.description must contain text")
        for name in ("supported_intents", "required_inputs", "optional_inputs", "safety_notes"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "category": self.category.value,
            "description": self.description,
            "supported_intents": list(self.supported_intents),
            "required_inputs": list(self.required_inputs),
            "optional_inputs": list(self.optional_inputs),
            "safety_notes": list(self.safety_notes),
        }


@dataclass(frozen=True, slots=True)
class ToolCandidate:
    """One bounded available candidate; it is not an executable call."""

    tool_name: str
    category: ToolCategory
    capability: str
    reason: str
    confidence: ToolSelectionConfidence
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    prerequisites: tuple[str, ...]
    expected_output: str
    risk_level: ToolSelectionRisk
    priority: int
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "category": self.category.value,
            "capability": self.capability,
            "reason": self.reason,
            "confidence": self.confidence.value,
            "required_inputs": list(self.required_inputs),
            "optional_inputs": list(self.optional_inputs),
            "prerequisites": list(self.prerequisites),
            "expected_output": self.expected_output,
            "risk_level": self.risk_level.value,
            "priority": self.priority,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ToolSelectionRequest:
    """Inputs to selection; registry metadata is read-only discovery only."""

    plan: ExecutionPlan
    registry: ToolRegistry | None = None
    project_context: ProjectContext | None = None
    available_inputs: tuple[str, ...] | list[str] = ()
    selected_step_ids: tuple[str, ...] | list[str] | None = None
    strict_prerequisites: bool = False
    config: ToolSelectionConfig = ToolSelectionConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExecutionPlan):
            raise ValueError("ToolSelectionRequest.plan must be ExecutionPlan")
        if not isinstance(self.available_inputs, tuple):
            object.__setattr__(self, "available_inputs", tuple(self.available_inputs))
        if self.selected_step_ids is not None and not isinstance(self.selected_step_ids, tuple):
            object.__setattr__(self, "selected_step_ids", tuple(self.selected_step_ids))
        if not isinstance(self.strict_prerequisites, bool):
            raise ValueError("strict_prerequisites must be boolean")
        if not isinstance(self.config, ToolSelectionConfig):
            raise ValueError("config must be ToolSelectionConfig")


@dataclass(frozen=True, slots=True)
class ToolSelectionDecision:
    """Declarative decision for one plan step."""

    plan_step_id: str
    status: ToolSelectionStatus
    selected_tool: str | None
    tool_category: ToolCategory
    selection_reason: str
    confidence: ToolSelectionConfidence
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    prerequisites: tuple[str, ...]
    missing_prerequisites: tuple[str, ...]
    expected_output: str
    alternatives: tuple[str, ...]
    forbidden_tools: tuple[str, ...]
    risk_level: ToolSelectionRisk
    warnings: tuple[str, ...]
    candidates: tuple[ToolCandidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_step_id": self.plan_step_id,
            "status": self.status.value,
            "selected_tool": self.selected_tool,
            "tool_category": self.tool_category.value,
            "selection_reason": self.selection_reason,
            "confidence": self.confidence.value,
            "required_inputs": list(self.required_inputs),
            "optional_inputs": list(self.optional_inputs),
            "prerequisites": list(self.prerequisites),
            "missing_prerequisites": list(self.missing_prerequisites),
            "expected_output": self.expected_output,
            "alternatives": list(self.alternatives),
            "forbidden_tools": list(self.forbidden_tools),
            "risk_level": self.risk_level.value,
            "warnings": list(self.warnings),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class ToolSelectionResult:
    """Complete bounded declarative selection output."""

    status: ToolSelectionStatus
    plan_task: str
    decisions: tuple[ToolSelectionDecision, ...]
    available_tools: tuple[str, ...]
    capabilities: tuple[ToolCapability, ...]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "plan_task": self.plan_task,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "available_tools": list(self.available_tools),
            "capabilities": [capability.to_dict() for capability in self.capabilities],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


class ToolSelectionValidationError(ValueError):
    """Raised by explicit validation of malformed selection results."""


class ToolSelectionValidator:
    """Validate selection structure without dispatching or executing tools."""

    def validate(self, result: ToolSelectionResult, *, plan: ExecutionPlan | None = None) -> tuple[str, ...]:
        errors: list[str] = []
        if not isinstance(result, ToolSelectionResult):
            return ("result must be ToolSelectionResult",)
        available = set(result.available_tools)
        capability_names = {capability.tool_name for capability in result.capabilities}
        if capability_names != available:
            errors.append("capabilities must correspond exactly to available_tools")
        seen_steps: set[str] = set()
        plan_steps = {step.step_id: step for step in plan.steps} if plan else {}
        for decision in result.decisions:
            if not isinstance(decision.status, ToolSelectionStatus):
                errors.append(f"invalid selection status for {decision.plan_step_id}")
            if not isinstance(decision.tool_category, ToolCategory):
                errors.append(f"invalid tool category for {decision.plan_step_id}")
            if not isinstance(decision.confidence, ToolSelectionConfidence):
                errors.append(f"invalid confidence for {decision.plan_step_id}")
            if not isinstance(decision.risk_level, ToolSelectionRisk):
                errors.append(f"invalid risk level for {decision.plan_step_id}")
            if decision.plan_step_id in seen_steps:
                errors.append(f"duplicate selection for plan step {decision.plan_step_id}")
            seen_steps.add(decision.plan_step_id)
            if plan and decision.plan_step_id not in plan_steps:
                errors.append(f"unknown plan_step_id: {decision.plan_step_id}")
            if decision.selected_tool is not None and not _tool_available(decision.selected_tool, available):
                errors.append(f"unavailable selected tool: {decision.selected_tool}")
            if decision.selected_tool in decision.alternatives:
                errors.append(f"selected tool is also an alternative: {decision.selected_tool}")
            if len(set(decision.alternatives)) != len(decision.alternatives):
                errors.append(f"duplicate alternatives for {decision.plan_step_id}")
            if any(not _tool_available(tool, available) for tool in decision.alternatives):
                errors.append(f"unavailable alternative for {decision.plan_step_id}")
            if decision.selected_tool in {"write_file", "edit_file", "delete_file"} and plan and decision.plan_step_id in plan_steps:
                if not _has_mutation_intent(plan_steps[decision.plan_step_id]):
                    errors.append(f"mutation tool selected without explicit mutation intent for {decision.plan_step_id}")
            if decision.selected_tool in {"run_command", "run_command_with_policy", "run_application", "run_tests"} and plan and decision.plan_step_id in plan_steps:
                if not _has_execution_intent(plan_steps[decision.plan_step_id]):
                    errors.append(f"execution tool selected without explicit execution intent for {decision.plan_step_id}")
            if decision.status is ToolSelectionStatus.SELECTED and decision.selected_tool is None:
                errors.append(f"selected status without selected tool for {decision.plan_step_id}")
        return tuple(_unique_sorted(errors))

    def validate_or_raise(self, result: ToolSelectionResult, *, plan: ExecutionPlan | None = None) -> ToolSelectionResult:
        errors = self.validate(result, plan=plan)
        if errors:
            raise ToolSelectionValidationError("; ".join(errors))
        return result


class ToolSelector:
    """Select existing registered capabilities without granting execution permission."""

    def __init__(self, *, config: ToolSelectionConfig | None = None, validator: ToolSelectionValidator | None = None) -> None:
        self.config = config or ToolSelectionConfig()
        self.validator = validator or ToolSelectionValidator()

    def capabilities_for(self, registry: ToolRegistry | None = None) -> tuple[ToolCapability, ...]:
        actual_registry = registry or ToolRegistry.default()
        capabilities = tuple(_capability_for_name(name, actual_registry.metadata_for(name)) for name in actual_registry.names())
        return tuple(sorted(capabilities, key=lambda item: item.tool_name))

    def select(self, request: ToolSelectionRequest | ExecutionPlan) -> ToolSelectionResult:
        if isinstance(request, ExecutionPlan):
            request = ToolSelectionRequest(request, config=self.config)
        if not isinstance(request, ToolSelectionRequest):
            return ToolSelectionResult(ToolSelectionStatus.INVALID_REQUEST, "", (), (), (), errors=("request must be ToolSelectionRequest",))
        plan = request.plan
        config = request.config
        registry = request.registry or ToolRegistry.default()
        available = registry.names()
        capabilities = self.capabilities_for(registry)
        if len(plan.steps) > config.max_plan_steps:
            return ToolSelectionResult(ToolSelectionStatus.INCOMPLETE, plan.task, (), available, capabilities, errors=("plan exceeds max_plan_steps",))
        requested_ids = set(request.selected_step_ids) if request.selected_step_ids is not None else {step.step_id for step in plan.steps}
        known_ids = {step.step_id for step in plan.steps}
        unknown = tuple(sorted(requested_ids - known_ids))
        if unknown:
            return ToolSelectionResult(ToolSelectionStatus.INVALID_REQUEST, plan.task, (), available, capabilities, errors=(f"unknown plan step IDs: {', '.join(unknown)}",))
        decisions: list[ToolSelectionDecision] = []
        for step in plan.steps:
            if step.step_id not in requested_ids:
                continue
            decisions.append(self._select_step(step, plan, capabilities, request))
        decisions_tuple = tuple(decisions)
        warnings = _unique_sorted(item for decision in decisions_tuple for item in decision.warnings)
        warnings = warnings[: config.max_warnings]
        status = _result_status(decisions_tuple)
        result = ToolSelectionResult(status, plan.task, decisions_tuple, available, capabilities, warnings=warnings)
        errors = self.validator.validate(result, plan=plan)
        if errors:
            return ToolSelectionResult(ToolSelectionStatus.INVALID_REQUEST, plan.task, decisions_tuple, available, capabilities, warnings=warnings, errors=errors)
        return result

    def select_for_step(self, request: ToolSelectionRequest, step_id: str) -> ToolSelectionDecision:
        result = self.select(ToolSelectionRequest(
            plan=request.plan,
            registry=request.registry,
            project_context=request.project_context,
            available_inputs=request.available_inputs,
            selected_step_ids=(step_id,),
            strict_prerequisites=request.strict_prerequisites,
            config=request.config,
        ))
        if result.errors:
            raise ToolSelectionValidationError("; ".join(result.errors))
        return result.decisions[0]

    def _select_step(self, step: PlanStep, plan: ExecutionPlan, capabilities: tuple[ToolCapability, ...], request: ToolSelectionRequest) -> ToolSelectionDecision:
        text = " ".join((step.title, step.objective, step.rationale)).casefold()
        available = {capability.tool_name: capability for capability in capabilities}
        ranked = _rank_intents(text, step, plan)
        forbidden = _forbidden_tools(text, available)
        if not ranked:
            return _decision_without_selection(step, ToolSelectionStatus.NO_SUITABLE_TOOL, "No existing capability can be mapped safely to this declarative step.", ToolSelectionConfidence.UNKNOWN, forbidden, "The step requires clarification before a tool can be selected.")
        candidates: list[ToolCandidate] = []
        for tool_name, priority, reason, intent in ranked:
            actual_name = _resolve_available_name(tool_name, available)
            capability = available.get(actual_name) if actual_name else None
            if capability is None:
                continue
            if tool_name in forbidden:
                continue
            prerequisites, missing = _prerequisites(tool_name, text, request, step)
            candidate = ToolCandidate(
                tool_name=capability.tool_name,
                category=capability.category,
                capability=intent,
                reason=reason,
                confidence=_candidate_confidence(priority, request.project_context, step),
                required_inputs=capability.required_inputs,
                optional_inputs=capability.optional_inputs,
                prerequisites=prerequisites,
                expected_output=_expected_output(tool_name),
                risk_level=_risk_for_category(capability.category),
                priority=priority,
                warnings=_selection_safety_warnings(capability),
            )
            candidates.append(candidate)
        candidates = sorted(candidates, key=lambda item: (-item.priority, item.tool_name))[: request.config.max_candidates_per_step]
        if not candidates:
            unavailable = _unavailable_name(ranked, available)
            status = ToolSelectionStatus.TOOL_UNAVAILABLE if unavailable else ToolSelectionStatus.NO_SUITABLE_TOOL
            reason = f"Required capability is unavailable in the supplied ToolRegistry: {unavailable}." if unavailable else "All matching capabilities were explicitly forbidden for this step."
            return _decision_without_selection(step, status, reason, ToolSelectionConfidence.LOW, forbidden, "Supply the required tool explicitly in a later execution phase.")
        top = candidates[0]
        alternatives = tuple(candidate.tool_name for candidate in candidates[1:request.config.max_alternatives + 1])
        tied = tuple(candidate.tool_name for candidate in candidates if candidate.priority == top.priority)
        if len(tied) > 1 and _is_equal_choice(text):
            return ToolSelectionDecision(step.step_id, ToolSelectionStatus.AMBIGUOUS_SELECTION, None, ToolCategory.UNKNOWN, "Multiple available capabilities are equally appropriate and the step does not distinguish them.", ToolSelectionConfidence.LOW, (), (), (), (), "No tool is selected until the plan clarifies the intended information or mutation.", tuple(tied), forbidden, ToolSelectionRisk.UNKNOWN, ("Selection ambiguity must be resolved before execution.",), tuple(candidates))
        prerequisites = top.prerequisites
        missing = tuple(item for item in prerequisites if not _prerequisite_satisfied(item, request.available_inputs))
        status = ToolSelectionStatus.MISSING_PREREQUISITES if request.strict_prerequisites and missing else ToolSelectionStatus.SELECTED
        warnings = list(top.warnings)
        if missing:
            warnings.append("Selection identifies prerequisites that must be satisfied before execution.")
        return ToolSelectionDecision(step.step_id, status, top.tool_name if status is not ToolSelectionStatus.AMBIGUOUS_SELECTION else None, top.category, top.reason, top.confidence, top.required_inputs, top.optional_inputs, prerequisites, missing, top.expected_output, alternatives, forbidden, top.risk_level, tuple(_unique_sorted(warnings)), tuple(candidates))


# The name emphasizes that the input is a plan and that selection is not execution.
PlanToolSelector = ToolSelector


def create_tool_selection(request: ToolSelectionRequest | ExecutionPlan) -> ToolSelectionResult:
    return ToolSelector().select(request)


def _capability_for_name(name: str, metadata: Any) -> ToolCapability:
    canonical_name = next((logical for logical, actual in _TOOL_ALIASES.items() if actual == name), name)
    catalog = _CAPABILITY_CATALOG.get(name) or _CAPABILITY_CATALOG.get(canonical_name)
    if catalog is not None:
        if catalog.tool_name == name:
            return catalog
        return ToolCapability(name, catalog.category, catalog.description, catalog.supported_intents, catalog.required_inputs, catalog.optional_inputs, catalog.safety_notes)
    description = getattr(metadata, "description", None) or "Registered tool with no declared selection capability."
    return ToolCapability(name, ToolCategory.UNKNOWN, str(description), safety_notes=("Unknown capabilities are not selected automatically.",))


def _rank_intents(text: str, step: PlanStep, plan: ExecutionPlan) -> list[tuple[str, int, str, str]]:
    if any(token in text for token in ("interpret test", "parse test", "test result", "test failure", "understand test")):
        return [("test_result_parser", 100, "The step explicitly interprets an existing test result.", "interpret_test_result")]
    if any(token in text for token in ("run tests", "run relevant verification", "execute tests", "test suite", "test execution")):
        return [("run_tests", 100, "The step explicitly requires test execution.", "run_tests")]
    if any(token in text for token in ("run application", "launch application", "start application")):
        return [("run_application", 100, "The step explicitly requires an application launch.", "run_application")]
    if any(token in text for token in ("run command", "run an approved command", "execute command", "approved command")):
        return [("run_command", 100, "The step explicitly requires an approved command.", "run_command")]
    if any(token in text for token in ("delete file", "remove file", "delete the", "remove the")):
        return [("delete_file", 100, "The step explicitly requires destructive file removal.", "delete_file")]
    if any(token in text for token in ("create file", "new file", "create a", "write a new")):
        return [("write_file", 100, "The step explicitly requires creation of a new file.", "create_file")]
    if any(token in text for token in ("modify existing", "modify file", "edit file", "change existing", "update existing")):
        return [("edit_file", 100, "The step explicitly requires modification of an existing file.", "modify_file")]
    if any(token in text for token in ("current repository changes", "repository status", "git status", "working tree status")):
        return [("git_status", 100, "The step requires read-only repository status inspection.", "inspect_git_status")]
    if any(token in text for token in ("actual changes", "git diff", "resulting change", "review the final change", "inspect the final change")):
        return [("git_diff", 100, "The step requires inspection of actual repository changes.", "inspect_git_diff"), ("git_status", 80, "Repository status is a bounded alternative for change review.", "inspect_git_status")]
    if any(token in text for token in ("canonical project context", "project context", "context is missing")):
        return [("project_context", 100, "The step requires canonical project-context discovery.", "discover_project_context")]
    if any(token in text for token in ("project structure", "discover project", "frameworks and languages", "project discovery")):
        return [("project_structure", 100, "The step requires bounded project-structure discovery.", "discover_project_structure")]
    if any(token in text for token in ("exact file", "file contents", "discovered file", "discovered source file", "source file", "read the file", "inspect source file")):
        return [("read_file", 100, "The step refers to inspecting known or discovered file contents.", "inspect_file_contents")]
    if any(token in text for token in ("locate", "find", "search", "relevant architecture", "inspect the existing", "inspect authentication")):
        return [("search_code", 100, "The step requires locating symbols, references, or implementation areas.", "locate_code"), ("list_files", 75, "Listing files is a bounded alternative when the location is unknown.", "discover_files")]
    if any(token in text for token in ("list files", "unknown files", "directory", "file tree")):
        return [("list_files", 100, "The step requires bounded file/directory discovery.", "discover_files")]
    if any(token in text for token in ("inspect", "understand", "review")):
        return [("search_code", 60, "Code search may locate the relevant implementation.", "locate_code"), ("read_file", 60, "Reading a known file may inspect the relevant details.", "inspect_file_contents"), ("list_files", 60, "File discovery may establish the relevant location.", "discover_files")]
    if any(token in text for token in ("implement", "add", "update", "change", "fix", "refactor")):
        return [("edit_file", 100, "The step explicitly describes an implementation or modification intent.", "modify_file")]
    return []


def _forbidden_tools(text: str, available: Mapping[str, ToolCapability]) -> tuple[str, ...]:
    names: set[str] = set()
    known = _known_capabilities(available)
    inspection = any(token in text for token in ("inspect", "locate", "find", "search", "discover", "review", "understand")) and not _has_mutation_word(text) and not any(token in text for token in ("run", "execute", "launch", "start"))
    if inspection:
        names.update(name for name, capability in _known_capabilities(available).items() if capability.category in {ToolCategory.MUTATING, ToolCategory.EXECUTION, ToolCategory.DESTRUCTIVE})
    if "parse" in text or "interpret" in text:
        names.update(name for name in ("run_tests", "run_command", "run_application", "run_command_with_policy") if name in known)
    if "run tests" in text or "execute tests" in text:
        names.update(name for name in ("write_file", "edit_file", "delete_file") if name in known)
    if "create" in text or "new file" in text:
        if "delete_file" in known:
            names.add("delete_file")
    if "delete" in text or "remove" in text:
        if "write_file" in known:
            names.add("write_file")
    return tuple(sorted(names))


def _known_capabilities(available: Mapping[str, ToolCapability]) -> dict[str, ToolCapability]:
    known = dict(available)
    for logical, actual in _TOOL_ALIASES.items():
        if actual in available:
            known.setdefault(logical, available[actual])
    for name, capability in _CAPABILITY_CATALOG.items():
        known.setdefault(name, capability)
    return known


def _resolve_available_name(logical_name: str, available: Mapping[str, ToolCapability]) -> str | None:
    if logical_name in available:
        return logical_name
    alias = _TOOL_ALIASES.get(logical_name)
    if alias in available:
        return alias
    return None


def _tool_available(name: str | None, available: Mapping[str, ToolCapability]) -> bool:
    return name is not None and _resolve_available_name(name, available) is not None


def _prerequisites(tool_name: str, text: str, request: ToolSelectionRequest, step: PlanStep) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if tool_name == "read_file":
        return ("target file path is known",), ("target file path is known",)
    if tool_name == "edit_file":
        return ("target file exists", "target file path is known", "expected modification is defined", "explicit mutation intent is confirmed"), ("target file is confirmed", "expected modification is confirmed")
    if tool_name == "write_file":
        return ("target path is known", "creation is explicitly planned", "target does not already exist at execution time"), ("target path is confirmed",)
    if tool_name == "delete_file":
        return ("target regular file path is known", "explicit deletion is confirmed", "safe-edit policy permits the operation"), ("target path is confirmed",)
    if tool_name in {"run_tests", "run_application", "run_command"}:
        return ("execution intent is explicit", "approved argv/plan is available", "existing CommandPolicy permits execution"), ("approved execution plan is available",)
    if tool_name == "test_result_parser":
        return (("TestRunResult exists",), ("TestRunResult exists",))
    if tool_name == "project_context":
        return (("project root is explicitly supplied",), ())
    if tool_name in {"project_structure", "list_files", "search_code"}:
        return (("project root is explicitly supplied",), ())
    if tool_name in {"git_status", "git_diff"}:
        return (("project root is a repository boundary",), ())
    return ((), ())


def _prerequisite_satisfied(item: str, available_inputs: Sequence[str]) -> bool:
    lowered = {value.casefold() for value in available_inputs}
    return item.casefold() in lowered or any(item.casefold() in value for value in lowered)


def _candidate_confidence(priority: int, context: ProjectContext | None, step: PlanStep) -> ToolSelectionConfidence:
    if context is None:
        return ToolSelectionConfidence.MEDIUM if priority >= 95 else ToolSelectionConfidence.LOW
    if context.completeness == "partial" or context.truncated:
        return ToolSelectionConfidence.MEDIUM
    return ToolSelectionConfidence.HIGH if priority >= 95 else ToolSelectionConfidence.MEDIUM


def _selection_safety_warnings(capability: ToolCapability) -> tuple[str, ...]:
    if capability.category in {ToolCategory.MUTATING, ToolCategory.DESTRUCTIVE}:
        return tuple(capability.safety_notes) + ("Execution remains subject to the existing safe-edit policy and SafeEditSession/SafeEditPolicy boundaries.",)
    if capability.category is ToolCategory.EXECUTION:
        return tuple(capability.safety_notes) + ("Execution remains subject to existing CommandPolicy and ProcessManager boundaries.",)
    return tuple(capability.safety_notes)


def _expected_output(tool_name: str) -> str:
    return {
        "list_files": "Bounded file/directory discovery facts.",
        "read_file": "Bounded UTF-8 file contents and metadata.",
        "search_code": "Bounded matching locations and context.",
        "project_structure": "Bounded structural project detections.",
        "project_context": "Canonical bounded ProjectContext facts.",
        "write_file": "Structured create-only mutation result, subject to safe editing policy.",
        "edit_file": "Structured exact replacement result, subject to safe editing policy.",
        "delete_file": "Structured regular-file deletion result, subject to safe editing policy.",
        "git_diff": "Bounded read-only repository diff facts.",
        "git_status": "Bounded read-only repository status facts.",
        "run_command": "Bounded approved command execution result.",
        "run_application": "Bounded application lifecycle and output result.",
        "run_tests": "Bounded raw TestRunResult for later interpretation.",
        "test_result_parser": "Bounded semantic TestParseResult from an existing TestRunResult.",
    }.get(tool_name, "Structured result from the selected registered capability.")


def _risk_for_category(category: ToolCategory) -> ToolSelectionRisk:
    return ToolSelectionRisk(category.value) if category.value in ToolSelectionRisk._value2member_map_ else ToolSelectionRisk.UNKNOWN


def _unavailable_name(ranked: Sequence[tuple[str, int, str, str]], available: Mapping[str, ToolCapability]) -> str | None:
    for name, _, _, _ in ranked:
        if name not in available:
            return name
    return None


def _is_equal_choice(text: str) -> bool:
    return not any(token in text for token in ("locate", "find", "search", "exact", "known", "directory", "file tree"))


def _decision_without_selection(step: PlanStep, status: ToolSelectionStatus, reason: str, confidence: ToolSelectionConfidence, forbidden: tuple[str, ...], expected: str) -> ToolSelectionDecision:
    return ToolSelectionDecision(step.step_id, status, None, ToolCategory.UNKNOWN, reason, confidence, (), (), (), (), expected, (), forbidden, ToolSelectionRisk.UNKNOWN, (), ())


def _result_status(decisions: Sequence[ToolSelectionDecision]) -> ToolSelectionStatus:
    if not decisions:
        return ToolSelectionStatus.INCOMPLETE
    statuses = {decision.status for decision in decisions}
    if ToolSelectionStatus.INVALID_REQUEST in statuses:
        return ToolSelectionStatus.INVALID_REQUEST
    if ToolSelectionStatus.AMBIGUOUS_SELECTION in statuses:
        return ToolSelectionStatus.AMBIGUOUS_SELECTION
    if ToolSelectionStatus.MISSING_PREREQUISITES in statuses:
        return ToolSelectionStatus.MISSING_PREREQUISITES
    if ToolSelectionStatus.TOOL_UNAVAILABLE in statuses:
        return ToolSelectionStatus.TOOL_UNAVAILABLE
    if all(status is ToolSelectionStatus.NO_SUITABLE_TOOL for status in statuses):
        return ToolSelectionStatus.NO_SUITABLE_TOOL
    return ToolSelectionStatus.SELECTED


def _has_mutation_word(text: str) -> bool:
    import re

    return any(re.search(rf"\b{re.escape(token)}\b", text) for token in ("create", "new file", "modify", "edit", "change", "implement", "update", "add", "refactor", "fix", "delete", "remove"))


def _has_mutation_intent(step: PlanStep) -> bool:
    text = " ".join((step.title, step.objective, step.expected_result)).casefold()
    return any(token in text for token in ("create", "new file", "modify", "edit", "change", "implement", "update", "add", "refactor", "fix"))


def _has_execution_intent(step: PlanStep) -> bool:
    text = " ".join((step.title, step.objective, step.expected_result)).casefold()
    return any(token in text for token in ("run", "execute", "launch", "start", "test suite", "tests"))


_TOOL_ALIASES = {
    "run_command": "run_command_with_policy",
    "test_result_parser": "parse_test_result",
}


def _unique_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value}, key=str.casefold))


_CAPABILITY_CATALOG: dict[str, ToolCapability] = {
    "list_files": ToolCapability("list_files", ToolCategory.READ_ONLY, "Inspect project files and directories.", ("discover_files",), ("project root",), ("path",), ("read-only",)),
    "read_file": ToolCapability("read_file", ToolCategory.READ_ONLY, "Inspect exact bounded UTF-8 file contents.", ("inspect_file_contents",), ("known file path",), (), ("read-only",)),
    "search_code": ToolCapability("search_code", ToolCategory.READ_ONLY, "Locate symbols, definitions, references, and patterns.", ("locate_code",), ("project root", "search query"), ("path",), ("read-only",)),
    "project_structure": ToolCapability("project_structure", ToolCategory.READ_ONLY, "Detect project structure, frameworks, languages, dependencies, and entry points.", ("discover_project_structure",), ("project root",), (), ("read-only",)),
    "project_context": ToolCapability("project_context", ToolCategory.READ_ONLY, "Build canonical bounded ProjectContext facts.", ("discover_project_context",), ("project root",), (), ("read-only",)),
    "write_file": ToolCapability("write_file", ToolCategory.MUTATING, "Create a new regular file through safe editing boundaries.", ("create_file",), ("target path", "content"), (), ("mutation requires explicit planned creation",)),
    "edit_file": ToolCapability("edit_file", ToolCategory.MUTATING, "Modify an existing file through exact bounded replacement.", ("modify_file",), ("target path", "expected replacement"), (), ("mutation requires explicit planned modification",)),
    "delete_file": ToolCapability("delete_file", ToolCategory.DESTRUCTIVE, "Delete one existing regular file through explicit safety boundaries.", ("delete_file",), ("target path",), (), ("destructive operation requires explicit planned deletion",)),
    "git_diff": ToolCapability("git_diff", ToolCategory.READ_ONLY, "Inspect bounded repository changes.", ("inspect_git_diff",), ("repository root",), (), ("read-only",)),
    "git_status": ToolCapability("git_status", ToolCategory.READ_ONLY, "Inspect bounded repository status.", ("inspect_git_status",), ("repository root",), (), ("read-only",)),
    "run_command": ToolCapability("run_command", ToolCategory.EXECUTION, "Execute one approved argv command behind CommandPolicy.", ("run_command",), ("approved argv",), (), ("never bypass CommandPolicy",)),
    "run_application": ToolCapability("run_application", ToolCategory.EXECUTION, "Launch one evidence-backed application behind policy and lifecycle controls.", ("run_application",), ("approved application plan",), (), ("never bypass CommandPolicy or ProcessManager",)),
    "run_tests": ToolCapability("run_tests", ToolCategory.EXECUTION, "Execute one approved bounded test plan.", ("run_tests",), ("approved test plan",), (), ("never bypass CommandPolicy or ProcessManager",)),
    "parse_test_result": ToolCapability("parse_test_result", ToolCategory.READ_ONLY, "Interpret an existing TestRunResult without executing tests.", ("interpret_test_result",), ("TestRunResult",), (), ("read-only; no rerun",)),
}


__all__ = [
    "PlanToolSelector",
    "ToolCandidate",
    "ToolCapability",
    "ToolCategory",
    "ToolSelectionConfidence",
    "ToolSelectionConfig",
    "ToolSelectionDecision",
    "ToolSelectionRequest",
    "ToolSelectionResult",
    "ToolSelectionRisk",
    "ToolSelectionStatus",
    "ToolSelectionValidationError",
    "ToolSelectionValidator",
    "ToolSelector",
    "create_tool_selection",
]
