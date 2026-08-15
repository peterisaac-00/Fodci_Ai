"""Deterministic, side-effect-free planning for Phase 6.1.

The planner consumes only the caller's task, optional ProjectContext, and explicit
configuration. It creates a structured plan and never selects or executes tools.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any

from backend_ai.tools.project_context import ProjectContext


class PlannerTaskType(str, Enum):
    FEATURE = "FEATURE"
    BUG_FIX = "BUG_FIX"
    REFACTOR = "REFACTOR"
    TEST_ADDITION = "TEST_ADDITION"
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"
    DOCUMENTATION_CHANGE = "DOCUMENTATION_CHANGE"
    DEPENDENCY_CHANGE = "DEPENDENCY_CHANGE"
    INVESTIGATION = "INVESTIGATION"
    UNKNOWN = "UNKNOWN"


class PlanRiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class PlanStepStatus(str, Enum):
    PLANNED = "PLANNED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class PlannerConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class PlanCompleteness(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    REQUIRES_CLARIFICATION = "REQUIRES_CLARIFICATION"


class PlannerResultStatus(str, Enum):
    CREATED = "CREATED"
    INCOMPLETE = "INCOMPLETE"
    INVALID_REQUEST = "INVALID_REQUEST"
    VALIDATION_ERROR = "VALIDATION_ERROR"


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Conservative deterministic planning budgets."""

    max_steps: int = 12
    max_assumptions: int = 8
    max_constraints: int = 12
    max_risks: int = 8
    max_warnings: int = 12
    max_task_length: int = 4_000
    max_plan_text_length: int = 1_024

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "max_assumptions",
            "max_constraints",
            "max_risks",
            "max_warnings",
            "max_task_length",
            "max_plan_text_length",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_steps > 64:
            raise ValueError("max_steps exceeds the planner safety ceiling")
        if self.max_task_length > 1_000_000 or self.max_plan_text_length > 1_000_000:
            raise ValueError("planner text budget exceeds the safety ceiling")


@dataclass(frozen=True, slots=True)
class PlannerRequest:
    """Inputs supplied to the planner; no project path is inferred or inspected."""

    task: str
    project_context: ProjectContext | None = None
    config: PlannerConfig = PlannerConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.task, str):
            raise ValueError("PlannerRequest.task must be text")
        if not isinstance(self.config, PlannerConfig):
            raise ValueError("PlannerRequest.config must be PlannerConfig")
        if self.project_context is not None and not isinstance(self.project_context, ProjectContext):
            raise ValueError("PlannerRequest.project_context must be ProjectContext or None")


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One declarative step describing what should happen, never how a tool is called."""

    step_id: str
    title: str
    objective: str
    rationale: str
    expected_result: str
    dependencies: tuple[str, ...] = ()
    risk_level: PlanRiskLevel = PlanRiskLevel.UNKNOWN
    verification_required: bool = True
    status: PlanStepStatus = PlanStepStatus.PLANNED

    def __post_init__(self) -> None:
        for name in ("step_id", "title", "objective", "rationale", "expected_result"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"PlanStep.{name} must contain text")
        if not isinstance(self.dependencies, tuple):
            object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if any(not isinstance(item, str) or not item.strip() for item in self.dependencies):
            raise ValueError("PlanStep.dependencies must contain non-empty step IDs")
        if not isinstance(self.risk_level, PlanRiskLevel):
            raise ValueError("PlanStep.risk_level must be PlanRiskLevel")
        if not isinstance(self.status, PlanStepStatus):
            raise ValueError("PlanStep.status must be PlanStepStatus")
        if not isinstance(self.verification_required, bool):
            raise ValueError("PlanStep.verification_required must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "objective": self.objective,
            "rationale": self.rationale,
            "expected_result": self.expected_result,
            "dependencies": list(self.dependencies),
            "risk_level": self.risk_level.value,
            "verification_required": self.verification_required,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class PlanRisk:
    """One risk statement with a bounded mitigation description."""

    risk_id: str
    description: str
    level: PlanRiskLevel
    mitigation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "risk_id": self.risk_id,
            "description": self.description,
            "level": self.level.value,
            "mitigation": self.mitigation,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Complete declarative plan handed to future Phase 6.2, not executed here."""

    task: str
    normalized_task: str
    goal: str
    task_type: PlannerTaskType
    steps: tuple[PlanStep, ...]
    assumptions: tuple[str, ...]
    constraints: tuple[str, ...]
    risks: tuple[PlanRisk, ...]
    expected_changes: tuple[str, ...]
    verification_strategy: tuple[str, ...]
    confidence: PlannerConfidence
    warnings: tuple[str, ...]
    completeness: PlanCompleteness

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            object.__setattr__(self, "steps", tuple(self.steps))
        for name in ("assumptions", "constraints", "expected_changes", "verification_strategy", "warnings"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
        if not isinstance(self.task_type, PlannerTaskType):
            raise ValueError("ExecutionPlan.task_type must be PlannerTaskType")
        if not isinstance(self.confidence, PlannerConfidence):
            raise ValueError("ExecutionPlan.confidence must be PlannerConfidence")
        if not isinstance(self.completeness, PlanCompleteness):
            raise ValueError("ExecutionPlan.completeness must be PlanCompleteness")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "normalized_task": self.normalized_task,
            "goal": self.goal,
            "task_type": self.task_type.value,
            "steps": [step.to_dict() for step in self.steps],
            "assumptions": list(self.assumptions),
            "constraints": list(self.constraints),
            "risks": [risk.to_dict() for risk in self.risks],
            "expected_changes": list(self.expected_changes),
            "verification_strategy": list(self.verification_strategy),
            "confidence": self.confidence.value,
            "warnings": list(self.warnings),
            "completeness": self.completeness.value,
        }


@dataclass(frozen=True, slots=True)
class PlanValidationResult:
    """Structured validator output instead of silently accepting malformed plans."""

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": list(self.errors), "warnings": list(self.warnings)}


@dataclass(frozen=True, slots=True)
class PlannerResult:
    """Structured planner operation result."""

    status: PlannerResultStatus
    plan: ExecutionPlan | None
    validation: PlanValidationResult
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "validation": self.validation.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


class PlanValidationError(ValueError):
    """Raised only by explicit validation APIs, never by plan creation for ambiguity."""


class PlanValidator:
    """Validate a plan's shape, bounds, DAG, and no-tool-call safety boundary."""

    def validate(self, plan: ExecutionPlan, *, config: PlannerConfig | None = None) -> PlanValidationResult:
        if not isinstance(plan, ExecutionPlan):
            return PlanValidationResult(False, ("plan must be an ExecutionPlan",))
        budget = config or PlannerConfig()
        errors: list[str] = []
        warnings: list[str] = []
        if len(plan.steps) > budget.max_steps:
            errors.append("step count exceeds max_steps")
        if len(plan.assumptions) > budget.max_assumptions:
            errors.append("assumption count exceeds max_assumptions")
        if len(plan.constraints) > budget.max_constraints:
            errors.append("constraint count exceeds max_constraints")
        if len(plan.risks) > budget.max_risks:
            errors.append("risk count exceeds max_risks")
        if len(plan.warnings) > budget.max_warnings:
            errors.append("warning count exceeds max_warnings")
        for field_name, value in (
            ("task", plan.task),
            ("normalized_task", plan.normalized_task),
            ("goal", plan.goal),
        ):
            if len(value) > budget.max_task_length:
                errors.append(f"{field_name} exceeds max_task_length")
        ids = [step.step_id for step in plan.steps]
        if len(ids) != len(set(ids)):
            errors.append("step IDs must be unique")
        known = set(ids)
        for step in plan.steps:
            for dependency in step.dependencies:
                if dependency not in known:
                    errors.append(f"step {step.step_id} depends on unknown step {dependency}")
            if any(_contains_execution_payload(text) for text in _step_texts(step)):
                errors.append(f"step {step.step_id} contains a tool call or execution payload")
            if len(step.title) > budget.max_plan_text_length or len(step.objective) > budget.max_plan_text_length or len(step.rationale) > budget.max_plan_text_length or len(step.expected_result) > budget.max_plan_text_length:
                errors.append(f"step {step.step_id} exceeds max_plan_text_length")
        if _has_cycle(plan.steps):
            errors.append("plan step dependencies must form a DAG")
        if any(_contains_execution_payload(text) for text in (plan.task, plan.normalized_task, plan.goal, *plan.assumptions, *plan.constraints, *plan.expected_changes, *plan.verification_strategy)):
            errors.append("plan contains an executable tool or shell payload")
        if plan.completeness is PlanCompleteness.PARTIAL:
            warnings.append("plan is partial and requires inspection before implementation")
        return PlanValidationResult(not errors, tuple(_unique_sorted(errors)), tuple(_unique_sorted(warnings)))

    def validate_or_raise(self, plan: ExecutionPlan, *, config: PlannerConfig | None = None) -> ExecutionPlan:
        result = self.validate(plan, config=config)
        if not result.valid:
            raise PlanValidationError("; ".join(result.errors))
        return plan


class Planner:
    """Create a plan from supplied text/context only; no tool or execution access exists."""

    def __init__(self, *, config: PlannerConfig | None = None, validator: PlanValidator | None = None) -> None:
        self.config = config or PlannerConfig()
        self.validator = validator or PlanValidator()

    def create_plan(
        self,
        task: str | PlannerRequest,
        *,
        project_context: ProjectContext | None = None,
        config: PlannerConfig | None = None,
    ) -> ExecutionPlan:
        request = task if isinstance(task, PlannerRequest) else PlannerRequest(task, project_context, config or self.config)
        result = self.plan(request)
        if result.plan is None:
            raise ValueError("Planner could not create a plan: " + "; ".join(result.errors))
        return result.plan

    def plan(self, request: PlannerRequest) -> PlannerResult:
        if not isinstance(request, PlannerRequest):
            validation = PlanValidationResult(False, ("request must be a PlannerRequest",))
            return PlannerResult(PlannerResultStatus.INVALID_REQUEST, None, validation, errors=validation.errors)
        config = request.config
        task = _bounded_text(request.task, config.max_task_length)
        if not task:
            validation = PlanValidationResult(False, ("task must not be empty",))
            return PlannerResult(PlannerResultStatus.INVALID_REQUEST, None, validation, errors=validation.errors)
        normalized = _normalize_task(task)
        task_type = _classify_task(normalized)
        warnings: list[str] = []
        if len(request.task) > config.max_task_length:
            warnings.append("task exceeded max_task_length and was safely truncated")
        context = request.project_context
        if context is None:
            warnings.append("ProjectContext was not supplied; project-aware assumptions remain unconfirmed")
        elif context.completeness == "partial" or context.truncated:
            warnings.append("ProjectContext is partial; missing project evidence must be inspected later")
        if task_type is PlannerTaskType.UNKNOWN:
            warnings.append("task category is ambiguous; the plan preserves ambiguity instead of inventing implementation details")
        if _is_ambiguous(normalized):
            warnings.append("task is ambiguous or underspecified; clarification may be required before implementation")
        steps = _build_steps(task_type, context, normalized, config.max_plan_text_length)
        assumptions = _build_assumptions(task_type, context, normalized)
        constraints = _build_constraints(task_type, context)
        risks = _build_risks(task_type, context, normalized)
        expected_changes = _expected_changes(task_type)
        verification = _verification_strategy(task_type)
        confidence = _confidence(task_type, context, normalized)
        completeness = _completeness(task_type, context, normalized)
        plan = ExecutionPlan(
            task=task,
            normalized_task=normalized,
            goal=_bounded_text(_goal(task_type, normalized), config.max_task_length),
            task_type=task_type,
            steps=tuple(steps[: config.max_steps]),
            assumptions=tuple(_bounded_items(assumptions, config.max_assumptions, warnings, "assumptions")),
            constraints=tuple(_bounded_items(constraints, config.max_constraints, warnings, "constraints")),
            risks=tuple(_bounded_risks(risks, config.max_risks, warnings)),
            expected_changes=tuple(_bounded_text(item, config.max_plan_text_length) for item in expected_changes),
            verification_strategy=tuple(_bounded_text(item, config.max_plan_text_length) for item in verification),
            confidence=confidence,
            warnings=tuple(_bounded_items(_unique_sorted(warnings), config.max_warnings, warnings, "warnings")),
            completeness=completeness,
        )
        validation = self.validator.validate(plan, config=config)
        if not validation.valid:
            return PlannerResult(PlannerResultStatus.VALIDATION_ERROR, None, validation, errors=validation.errors)
        status = PlannerResultStatus.INCOMPLETE if plan.completeness is not PlanCompleteness.COMPLETE else PlannerResultStatus.CREATED
        return PlannerResult(status, plan, validation, warnings=plan.warnings)


def create_plan(
    task: str,
    *,
    project_context: ProjectContext | None = None,
    config: PlannerConfig | None = None,
) -> ExecutionPlan:
    """Convenience API for deterministic plan creation."""

    return Planner(config=config).create_plan(task, project_context=project_context)


def _normalize_task(task: str) -> str:
    normalized = " ".join(task.replace("\x00", " ").split())
    return normalized


def _classify_task(task: str) -> PlannerTaskType:
    text = task.casefold()
    if any(word in text for word in ("readme", "documentation", "document", "docs", "docstring")):
        return PlannerTaskType.DOCUMENTATION_CHANGE
    if any(word in text for word in ("dependency", "dependencies", "package", "requirements", "upgrade library", "install library")):
        return PlannerTaskType.DEPENDENCY_CHANGE
    if any(word in text for word in ("config", "configuration", "settings", ".env", "environment variable")):
        return PlannerTaskType.CONFIGURATION_CHANGE
    if any(word in text for word in ("test", "tests", "pytest", "unittest", "jest", "vitest")) and not any(word in text for word in ("fix test", "failing test", "broken test")):
        return PlannerTaskType.TEST_ADDITION
    if any(word in text for word in ("refactor", "restructure", "cleanup", "clean up", "rename", "extract")):
        return PlannerTaskType.REFACTOR
    if any(word in text for word in ("investigate", "diagnose", "why does", "understand", "debug")):
        return PlannerTaskType.INVESTIGATION
    if any(word in text for word in ("bug", "fix", "broken", "failing", "regression", "error")):
        return PlannerTaskType.BUG_FIX
    if any(word in text for word in ("add", "implement", "create", "build", "feature", "endpoint")):
        return PlannerTaskType.FEATURE
    return PlannerTaskType.UNKNOWN


def _goal(task_type: PlannerTaskType, task: str) -> str:
    if task_type is PlannerTaskType.UNKNOWN:
        return "Clarify and address the requested engineering task without assuming unsupported implementation details."
    return f"Deliver the requested {task_type.value.casefold().replace('_', ' ')} while preserving the existing project architecture and conventions."


def _build_steps(task_type: PlannerTaskType, context: ProjectContext | None, task: str, text_limit: int) -> list[PlanStep]:
    context_label = context.stack_summary if context and context.stack_summary else "the supplied project context"
    steps: list[PlanStep] = [
        PlanStep("step-1", "Bound the task", "Clarify the requested outcome, scope, and acceptance conditions.", "Planning must preserve ambiguity instead of inventing requirements.", "A bounded engineering objective and explicit open questions.", risk_level=PlanRiskLevel.UNKNOWN, status=PlanStepStatus.NEEDS_CLARIFICATION if _is_ambiguous(task) else PlanStepStatus.PLANNED),
    ]
    if task_type is PlannerTaskType.DOCUMENTATION_CHANGE:
        steps.extend([
            PlanStep("step-2", "Inspect documentation conventions", "Review the supplied documentation context and identify the relevant documentation surface.", "Documentation changes should follow existing terminology and structure.", "The minimal documentation locations and consistency requirements are understood.", ("step-1",), PlanRiskLevel.LOW),
            PlanStep("step-3", "Update the documentation", "Apply only the requested documentation change using the existing project conventions.", "A targeted update avoids unrelated rewrites.", "Requested documentation content is updated without unrelated code changes.", ("step-2",), PlanRiskLevel.LOW),
            PlanStep("step-4", "Verify the documentation change", "Check wording, references, formatting, and the absence of unintended changes.", "Documentation work needs content and scope verification rather than unnecessary execution.", "The documentation is coherent and the change is limited to the requested scope.", ("step-3",), PlanRiskLevel.LOW),
        ])
    elif task_type is PlannerTaskType.INVESTIGATION:
        steps.extend([
            PlanStep("step-2", "Inspect the relevant architecture", f"Inspect the supplied project evidence for components related to: {task}.", "The planner cannot inspect the repository itself; future inspection must establish facts first.", "Relevant components, conventions, and unknowns are identified.", ("step-1",), PlanRiskLevel.MEDIUM),
            PlanStep("step-3", "Collect behavioral evidence", "Reproduce or observe the reported behavior using the approved future workflow and record evidence.", "Diagnosis requires evidence before a fix is selected.", "A bounded evidence record distinguishes symptoms from confirmed causes.", ("step-2",), PlanRiskLevel.MEDIUM),
            PlanStep("step-4", "Define the smallest corrective direction", "Select a minimal evidence-supported next change or clarification request.", "Root-cause diagnosis is not justified before evidence is collected.", "A reviewable corrective direction or explicit clarification is available.", ("step-3",), PlanRiskLevel.MEDIUM),
        ])
    else:
        steps.extend([
            PlanStep("step-2", "Inspect existing conventions", f"Inspect the existing {context_label} architecture, relevant components, and project conventions before deciding implementation details.", "Inspection-first planning prevents invented filenames, dependencies, and architecture.", "Affected areas, existing patterns, and confirmed constraints are identified.", ("step-1",), PlanRiskLevel.MEDIUM),
            PlanStep("step-3", "Design the minimal change", "Define the smallest change that satisfies the bounded task and reuses existing abstractions.", "Minimal targeted changes reduce regression and compatibility risk.", "A reviewable change boundary and acceptance conditions are defined.", ("step-2",), _step_risk(task_type)),
            PlanStep("step-4", "Implement the requested change", "Apply the required source, test, configuration, dependency, or documentation changes within the confirmed change boundary.", "Implementation details must follow inspection evidence rather than assumptions.", "The requested behavior or content is implemented with no unrelated changes.", ("step-3",), _step_risk(task_type)),
            PlanStep("step-5", "Add or update verification coverage", "Add or update relevant tests or other verification evidence using the project's existing conventions.", "Verification should cover the requested behavior and protect existing behavior.", "Relevant verification coverage is present or an explicit reason is documented.", ("step-4",), PlanRiskLevel.MEDIUM),
            PlanStep("step-6", "Run relevant verification", "Run the relevant bounded verification workflow and preserve raw results for later interpretation.", "Execution belongs to a future explicitly selected workflow, not to the Planner.", "Verification results are available for review.", ("step-5",), PlanRiskLevel.MEDIUM),
            PlanStep("step-7", "Review the final change", "Inspect the resulting change against the task, constraints, and expected behavior.", "A final review catches scope drift and unexpected modifications.", "The change is ready for explicit completion verification or further clarification.", ("step-6",), PlanRiskLevel.MEDIUM),
        ])
    return [_limit_step_text(step, text_limit) for step in steps]


def _step_risk(task_type: PlannerTaskType) -> PlanRiskLevel:
    if task_type in {PlannerTaskType.DEPENDENCY_CHANGE, PlannerTaskType.CONFIGURATION_CHANGE, PlannerTaskType.BUG_FIX}:
        return PlanRiskLevel.HIGH if task_type is PlannerTaskType.DEPENDENCY_CHANGE else PlanRiskLevel.MEDIUM
    if task_type is PlannerTaskType.DOCUMENTATION_CHANGE:
        return PlanRiskLevel.LOW
    return PlanRiskLevel.MEDIUM


def _limit_step_text(step: PlanStep, limit: int) -> PlanStep:
    return PlanStep(step.step_id, _bounded_text(step.title, limit), _bounded_text(step.objective, limit), _bounded_text(step.rationale, limit), _bounded_text(step.expected_result, limit), step.dependencies, step.risk_level, step.verification_required, step.status)


def _build_assumptions(task_type: PlannerTaskType, context: ProjectContext | None, task: str) -> list[str]:
    assumptions = ["The exact affected files and implementation details are not confirmed until inspection."]
    if context is None:
        assumptions.append("The project structure, conventions, and available dependencies are not confirmed because ProjectContext was not supplied.")
    elif context.completeness == "partial" or context.truncated:
        assumptions.append("The supplied ProjectContext is partial, so unobserved project areas remain unconfirmed.")
    if task_type is PlannerTaskType.DEPENDENCY_CHANGE:
        assumptions.append("Dependency compatibility and lockfile policy require confirmation from the existing project evidence.")
    if _is_ambiguous(task):
        assumptions.append("The user's acceptance criteria may require clarification before implementation.")
    return assumptions


def _build_constraints(task_type: PlannerTaskType, context: ProjectContext | None) -> list[str]:
    constraints = [
        "Do not invent filenames, modules, dependencies, or architecture before inspection.",
        "Prefer minimal targeted changes and preserve backward-compatible existing behavior where relevant.",
        "Use the project's existing conventions and verification approach after they are confirmed.",
        "This Planner creates a declarative plan only; tool selection and execution belong to later phases.",
    ]
    if context is not None:
        if context.stack_summary:
            constraints.append(f"Respect the supplied project stack evidence: {context.stack_summary}.")
        if context.test_frameworks:
            names = ", ".join(sorted(item.name for item in context.test_frameworks))
            constraints.append(f"Prefer the supplied test-framework evidence when designing verification: {names}.")
        if context.warnings:
            constraints.append("Treat all ProjectContext warnings as unresolved constraints during later inspection.")
    if task_type is PlannerTaskType.DOCUMENTATION_CHANGE:
        constraints.append("Avoid unnecessary code execution for documentation-only work.")
    return constraints


def _build_risks(task_type: PlannerTaskType, context: ProjectContext | None, task: str) -> list[PlanRisk]:
    risks: list[PlanRisk] = []
    if task_type is PlannerTaskType.DEPENDENCY_CHANGE:
        risks.append(PlanRisk("risk-1", "Dependency changes can affect compatibility, lockfiles, and reproducibility.", PlanRiskLevel.HIGH, "Inspect existing dependency metadata and verify the smallest compatible change."))
    elif task_type is PlannerTaskType.CONFIGURATION_CHANGE:
        risks.append(PlanRisk("risk-1", "Configuration changes can alter runtime behavior or expose sensitive settings.", PlanRiskLevel.HIGH, "Inspect existing configuration conventions and keep secrets out of plans and changes."))
    elif task_type is PlannerTaskType.BUG_FIX:
        risks.append(PlanRisk("risk-1", "A symptom may have multiple causes and a narrow fix may miss regression coverage.", PlanRiskLevel.MEDIUM, "Collect evidence first and run targeted plus relevant regression verification."))
    elif task_type is PlannerTaskType.REFACTOR:
        risks.append(PlanRisk("risk-1", "Broad refactors can change behavior outside the requested scope.", PlanRiskLevel.MEDIUM, "Keep the change boundary minimal and preserve existing tests and interfaces."))
    elif task_type is PlannerTaskType.DOCUMENTATION_CHANGE:
        risks.append(PlanRisk("risk-1", "Inconsistent documentation can mislead future implementation work.", PlanRiskLevel.LOW, "Review terminology, references, and scope before completion."))
    else:
        risks.append(PlanRisk("risk-1", "The affected architecture is not fully known before inspection.", PlanRiskLevel.UNKNOWN if context is None else PlanRiskLevel.MEDIUM, "Inspect supplied context and existing conventions before implementation."))
    if _is_ambiguous(task):
        risks.append(PlanRisk("risk-2", "Ambiguous acceptance criteria may lead to an incorrect change.", PlanRiskLevel.HIGH, "Request clarification or preserve open questions before implementation."))
    if context is None or (context is not None and (context.completeness == "partial" or context.truncated)):
        risks.append(PlanRisk("risk-3", "Incomplete project evidence may hide affected components or constraints.", PlanRiskLevel.UNKNOWN, "Treat the plan as inspection-dependent and reduce confidence."))
    return risks


def _expected_changes(task_type: PlannerTaskType) -> list[str]:
    if task_type is PlannerTaskType.DOCUMENTATION_CHANGE:
        return ["Documentation content or references; exact files require inspection."]
    if task_type is PlannerTaskType.TEST_ADDITION:
        return ["Test source files and, only if required by confirmed conventions, minimal related source changes."]
    if task_type is PlannerTaskType.CONFIGURATION_CHANGE:
        return ["Configuration files and possibly narrowly related validation; exact files require inspection."]
    if task_type is PlannerTaskType.DEPENDENCY_CHANGE:
        return ["Dependency metadata and lockfile-related files only if confirmed necessary."]
    if task_type is PlannerTaskType.UNKNOWN:
        return ["Unknown until the task is clarified and the project is inspected."]
    return ["Minimal affected source files and relevant verification files; exact filenames require inspection."]


def _verification_strategy(task_type: PlannerTaskType) -> list[str]:
    if task_type is PlannerTaskType.DOCUMENTATION_CHANGE:
        return ["Review requested wording, references, formatting, and scope; avoid unnecessary code execution."]
    if task_type is PlannerTaskType.INVESTIGATION:
        return ["Review collected evidence, confirm the stated behavior, and document remaining uncertainty."]
    return [
        "Verify expected files and behavior against the task and confirmed project conventions.",
        "Inspect the resulting change for scope drift and unexpected modifications.",
        "Run relevant bounded tests or checks in a later explicitly selected execution phase.",
        "Preserve raw execution results for later interpretation and completion verification.",
    ]


def _confidence(task_type: PlannerTaskType, context: ProjectContext | None, task: str) -> PlannerConfidence:
    if task_type is PlannerTaskType.UNKNOWN:
        return PlannerConfidence.UNKNOWN
    if _is_ambiguous(task):
        return PlannerConfidence.LOW
    if context is None:
        return PlannerConfidence.LOW
    if context.completeness == "partial" or context.truncated:
        return PlannerConfidence.MEDIUM
    return PlannerConfidence.HIGH


def _completeness(task_type: PlannerTaskType, context: ProjectContext | None, task: str) -> PlanCompleteness:
    if task_type is PlannerTaskType.UNKNOWN or _is_ambiguous(task):
        return PlanCompleteness.REQUIRES_CLARIFICATION
    if context is None or context.completeness == "partial" or context.truncated:
        return PlanCompleteness.PARTIAL
    return PlanCompleteness.COMPLETE


def _is_ambiguous(task: str) -> bool:
    text = task.casefold()
    if len(text.split()) <= 2:
        return True
    return any(phrase in text for phrase in ("improve the api", "fix authentication", "make it better", "update the system", "handle the issue"))


def _bounded_text(value: str, limit: int) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value
    marker = "…"
    marker_bytes = len(marker.encode("utf-8"))
    prefix = encoded[: max(0, limit - marker_bytes)].decode("utf-8", errors="replace")
    return prefix + marker


def _bounded_items(items: Sequence[str], limit: int, warnings: list[str], label: str) -> list[str]:
    values = list(items)
    if len(values) > limit:
        warnings.append(f"{label} exceeded its budget and was safely truncated")
    return values[:limit]


def _bounded_risks(risks: Sequence[PlanRisk], limit: int, warnings: list[str]) -> list[PlanRisk]:
    values = list(risks)
    if len(values) > limit:
        warnings.append("risks exceeded their budget and were safely truncated")
    return values[:limit]


def _step_texts(step: PlanStep) -> tuple[str, ...]:
    return (step.title, step.objective, step.rationale, step.expected_result)


def _contains_execution_payload(value: str) -> bool:
    lowered = value.casefold()
    forbidden = (
        "read_file(", "write_file(", "edit_file(", "delete_file(", "run_command(", "run_tests(", "run_application(",
        "parse_test_result(", "subprocess", "shell=true", "bash -c", "sh -c", "powershell", "cmd.exe",
        "&&", "||", "$()", "`", "git commit", "git push", "pip install", "npm install",
    )
    return any(token in lowered for token in forbidden)


def _has_cycle(steps: Sequence[PlanStep]) -> bool:
    graph = {step.step_id: tuple(step.dependencies) for step in steps}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in graph.get(node, ())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _unique_sorted(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value}, key=str.casefold))


__all__ = [
    "ExecutionPlan",
    "PlanCompleteness",
    "PlanRisk",
    "PlanRiskLevel",
    "PlanStep",
    "PlanStepStatus",
    "PlanValidationError",
    "PlanValidationResult",
    "PlanValidator",
    "Planner",
    "PlannerConfidence",
    "PlannerConfig",
    "PlannerRequest",
    "PlannerResult",
    "PlannerResultStatus",
    "PlannerTaskType",
    "create_plan",
]
