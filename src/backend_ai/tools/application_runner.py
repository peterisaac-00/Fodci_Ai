"""Bounded, evidence-backed application launching above policy and process layers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.command import CommandRequest, CommandResult, DEFAULT_MAX_STDERR_BYTES, DEFAULT_MAX_STDOUT_BYTES, DEFAULT_COMMAND_TIMEOUT_SECONDS
from backend_ai.tools.command_policy import CommandDecision, CommandPolicy, PolicyRunCommandTool
from backend_ai.tools.filesystem import _validate_root
from backend_ai.tools.process_manager import ProcessManager
from backend_ai.tools.project_context import ProjectContext, ProjectContextBuilder
from backend_ai.tools.read_file import read_file


class ApplicationRunStatus(str, Enum):
    RESOLVED = "resolved"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    POLICY_DENIED = "policy_denied"
    NO_APPLICATION_ENTRYPOINT = "no_application_entrypoint"
    AMBIGUOUS_ENTRYPOINT = "ambiguous_entrypoint"
    UNSUPPORTED_PROJECT = "unsupported_project"
    INVALID_WORKING_DIRECTORY = "invalid_working_directory"
    EXECUTABLE_NOT_FOUND = "executable_not_found"
    PERMISSION_DENIED = "permission_denied"
    START_FAILED = "start_failed"
    PROCESS_FAILED = "process_failed"
    OUTPUT_LIMIT_REACHED = "output_limit_reached"
    RESOLUTION_FAILED = "resolution_failed"


class ApplicationFailureCode(str, Enum):
    NO_APPLICATION_ENTRYPOINT = "NO_APPLICATION_ENTRYPOINT"
    AMBIGUOUS_ENTRYPOINT = "AMBIGUOUS_ENTRYPOINT"
    UNSUPPORTED_PROJECT = "UNSUPPORTED_PROJECT"
    POLICY_DENIED = "POLICY_DENIED"
    INVALID_WORKING_DIRECTORY = "INVALID_WORKING_DIRECTORY"
    EXECUTABLE_NOT_FOUND = "EXECUTABLE_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    START_FAILED = "START_FAILED"
    TIMEOUT = "TIMEOUT"
    PROCESS_FAILED = "PROCESS_FAILED"
    OUTPUT_LIMIT_REACHED = "OUTPUT_LIMIT_REACHED"
    RESOLUTION_FAILED = "RESOLUTION_FAILED"


@dataclass(frozen=True, slots=True)
class ApplicationRunRequest:
    project_root: Path | str
    working_directory: Path | str | None = None
    argv: tuple[str, ...] | list[str] | str | None = None
    timeout_seconds: float | None = None
    max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES
    environment: Mapping[str, str] | None = None
    inherit_environment: bool = False


@dataclass(frozen=True, slots=True)
class ApplicationLaunchCandidate:
    argv: tuple[str, ...]
    working_directory: str
    source: str
    project_type: str
    confidence: str
    warnings: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(_safe_argv(self.argv)),
            "working_directory": self.working_directory,
            "source": self.source,
            "project_type": self.project_type,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class ApplicationRunPlan:
    argv: tuple[str, ...]
    working_directory: str
    source: str
    project_type: str
    confidence: str
    explicit: bool
    evidence: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(_safe_argv(self.argv)),
            "working_directory": self.working_directory,
            "source": self.source,
            "project_type": self.project_type,
            "confidence": self.confidence,
            "explicit": self.explicit,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ApplicationResolution:
    status: ApplicationRunStatus
    plan: ApplicationRunPlan | None
    candidates: tuple[ApplicationLaunchCandidate, ...]
    project_type: str
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]
    failure_code: ApplicationFailureCode | None = None


@dataclass(frozen=True, slots=True)
class ApplicationRunResult:
    status: ApplicationRunStatus
    plan: ApplicationRunPlan | None
    command_result: CommandResult | None
    decision: CommandDecision | None
    failure_code: ApplicationFailureCode | None
    project_type: str
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]
    candidates: tuple[ApplicationLaunchCandidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "command_result": self.command_result.to_dict() if self.command_result else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "failure_code": self.failure_code.value if self.failure_code else None,
            "project_type": self.project_type,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "candidates": [item.to_dict() for item in self.candidates],
        }


class ApplicationCommandResolver:
    """Resolve only evidence-backed, bounded Python/Node launch candidates."""

    def __init__(self, *, context_builder: ProjectContextBuilder | None = None) -> None:
        self._context_builder = context_builder or ProjectContextBuilder()

    def resolve(self, request: ApplicationRunRequest) -> ApplicationResolution:
        root = _validate_root(request.project_root)
        working_directory = _relative_working_directory(root, request.working_directory)
        if request.argv is not None:
            if isinstance(request.argv, str) or not isinstance(request.argv, Sequence) or not request.argv or any(not isinstance(item, str) for item in request.argv):
                return ApplicationResolution(ApplicationRunStatus.RESOLUTION_FAILED, None, (), "unknown", (), ("Explicit argv must be a non-empty sequence of strings, not a command string.",), ApplicationFailureCode.RESOLUTION_FAILED)
            context = self._context_builder.build(root)
            plan = ApplicationRunPlan(tuple(request.argv), working_directory, "explicit argv supplied by caller", context.project_type, "explicit", True, ("Explicit argv mode; no automatic command was invented.",), ())
            return ApplicationResolution(ApplicationRunStatus.RESOLVED, plan, (), context.project_type, context.evidence, context.warnings)
        context = self._context_builder.build(root)
        if context.truncated:
            return ApplicationResolution(ApplicationRunStatus.RESOLUTION_FAILED, None, (), context.project_type, context.evidence, (*context.warnings, "Automatic application resolution requires complete bounded structural evidence."), ApplicationFailureCode.RESOLUTION_FAILED)
        candidates = self._candidates(root, context, working_directory)
        if not candidates:
            if context.project_type in {"python", "node", "mixed"}:
                status = ApplicationRunStatus.NO_APPLICATION_ENTRYPOINT
                code = ApplicationFailureCode.NO_APPLICATION_ENTRYPOINT
            else:
                status = ApplicationRunStatus.UNSUPPORTED_PROJECT
                code = ApplicationFailureCode.UNSUPPORTED_PROJECT
            return ApplicationResolution(status, None, (), context.project_type, context.evidence, context.warnings, code)
        if context.project_type == "mixed" or len(candidates) > 1:
            return ApplicationResolution(ApplicationRunStatus.AMBIGUOUS_ENTRYPOINT, None, candidates, context.project_type, context.evidence, context.warnings, ApplicationFailureCode.AMBIGUOUS_ENTRYPOINT)
        selected = candidates[0]
        plan = ApplicationRunPlan(selected.argv, selected.working_directory, selected.source, selected.project_type, selected.confidence, False, (*selected.evidence,), selected.warnings)
        return ApplicationResolution(ApplicationRunStatus.RESOLVED, plan, candidates, context.project_type, context.evidence, context.warnings)

    def _candidates(self, root: Path, context: ProjectContext, working_directory: str) -> tuple[ApplicationLaunchCandidate, ...]:
        paths = set(context.project_files)
        candidates: list[ApplicationLaunchCandidate] = []
        python_paths = [path for path in context.entry_points if path.name.casefold() in {"main.py", "app.py", "server.py"}]
        frameworks = {item.name for item in context.frameworks}
        if "Django" in frameworks and any(Path(path).name.casefold() == "manage.py" for path in paths):
            candidates.append(ApplicationLaunchCandidate((sys.executable, "manage.py", "runserver"), working_directory, "manage.py plus Django structural evidence", "python", "high", evidence=("manage.py present", "Django detected")))
        else:
            for detection in python_paths:
                if _python_entrypoint_supported(root, detection.name):
                    candidates.append(ApplicationLaunchCandidate((sys.executable, detection.name), working_directory, f"{detection.name}: bounded Python entry-point evidence", "python", detection.confidence, evidence=detection.evidence))
        package = _read_package_json(root, paths)
        if package:
            node_candidate = _node_candidate(package, paths, working_directory)
            if node_candidate:
                candidates.append(node_candidate)
        return tuple(sorted(candidates, key=lambda item: (item.project_type, item.argv, item.source)))


class ApplicationRunner:
    """Run one resolved application through policy and ProcessManager."""

    def __init__(self, *, resolver: ApplicationCommandResolver | None = None, policy: CommandPolicy | None = None) -> None:
        self._resolver = resolver or ApplicationCommandResolver()
        if policy is not None:
            self._policy = policy
        else:
            default_policy = CommandPolicy.default().with_executable_path(sys.executable)
            node_path = shutil.which("node")
            self._policy = default_policy.with_executable_path(node_path) if node_path else default_policy

    def run(self, request: ApplicationRunRequest) -> ApplicationRunResult:
        try:
            resolution = self._resolver.resolve(request)
        except ToolError as exc:
            if exc.code in {ToolErrorCode.PATH_OUTSIDE_ROOT, ToolErrorCode.WORKING_DIRECTORY_INVALID, ToolErrorCode.PERMISSION_DENIED}:
                return ApplicationRunResult(ApplicationRunStatus.INVALID_WORKING_DIRECTORY, None, None, None, ApplicationFailureCode.INVALID_WORKING_DIRECTORY, "unknown", (), (exc.message,), ())
            return ApplicationRunResult(ApplicationRunStatus.RESOLUTION_FAILED, None, None, None, ApplicationFailureCode.RESOLUTION_FAILED, "unknown", (), (exc.message,), ())
        if resolution.status is not ApplicationRunStatus.RESOLVED or resolution.plan is None:
            return ApplicationRunResult(resolution.status, None, None, None, resolution.failure_code, resolution.project_type, resolution.evidence, resolution.warnings, resolution.candidates)
        plan = resolution.plan
        safe_plan = _safe_plan(plan)
        policy = self._policy
        if not plan.explicit:
            policy = policy.with_exact_argv(plan.argv, name=f"evidence:{plan.source}")
        command_request = CommandRequest(plan.argv, request.project_root, plan.working_directory, request.environment, request.inherit_environment, request.timeout_seconds or DEFAULT_COMMAND_TIMEOUT_SECONDS, request.max_stdout_bytes, request.max_stderr_bytes)
        decision = policy.evaluate(command_request)
        if not decision.allowed:
            return ApplicationRunResult(ApplicationRunStatus.POLICY_DENIED, safe_plan, None, decision, ApplicationFailureCode.POLICY_DENIED, resolution.project_type, resolution.evidence, (*resolution.warnings, decision.reason), resolution.candidates)
        try:
            executed = PolicyRunCommandTool(policy).run({
                "argv": list(plan.argv),
                "project_root": str(request.project_root),
                "working_directory": plan.working_directory,
                "environment": request.environment,
                "inherit_environment": request.inherit_environment,
                "timeout_seconds": command_request.timeout_seconds,
                "max_stdout_bytes": request.max_stdout_bytes,
                "max_stderr_bytes": request.max_stderr_bytes,
            })
        except ToolError as exc:
            return ApplicationRunResult(ApplicationRunStatus.POLICY_DENIED, safe_plan, None, decision, ApplicationFailureCode.POLICY_DENIED, resolution.project_type, resolution.evidence, (*resolution.warnings, exc.message), resolution.candidates)
        command_result = _safe_command_result(executed.command_result)
        status, failure = _status_from_command(command_result)
        return ApplicationRunResult(status, safe_plan, command_result, executed.decision, failure, resolution.project_type, resolution.evidence, (*resolution.warnings, *command_result.warnings), resolution.candidates)


class RunApplicationTool:
    name = "run_application"
    description = "Resolve and run one bounded existing application through evidence, policy, and ProcessManager."
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root"],
            "properties": {
                "project_root": {"type": "string"},
                "working_directory": {"type": "string"},
                "argv": {"type": "array", "items": {"type": "string"}},
                "timeout_seconds": {"type": "number"},
                "max_stdout_bytes": {"type": "integer"},
                "max_stderr_bytes": {"type": "integer"},
                "environment": {"type": "object"},
                "inherit_environment": {"type": "boolean"},
            },
        },
    )

    def __init__(self, runner: ApplicationRunner | None = None) -> None:
        self._runner = runner or ApplicationRunner()

    def run(self, arguments: Mapping[str, Any]) -> ApplicationRunResult:
        if not isinstance(arguments, Mapping) or "project_root" not in arguments:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "run_application requires a project_root object argument.")
        return self._runner.run(ApplicationRunRequest(
            project_root=arguments["project_root"],
            working_directory=arguments.get("working_directory"),
            argv=arguments.get("argv"),
            timeout_seconds=arguments.get("timeout_seconds"),
            max_stdout_bytes=arguments.get("max_stdout_bytes", DEFAULT_MAX_STDOUT_BYTES),
            max_stderr_bytes=arguments.get("max_stderr_bytes", DEFAULT_MAX_STDERR_BYTES),
            environment=arguments.get("environment"),
            inherit_environment=arguments.get("inherit_environment", False),
        ))


def run_application(request: ApplicationRunRequest) -> ApplicationRunResult:
    return ApplicationRunner().run(request)


def _relative_working_directory(root: Path, raw: Path | str | None) -> str:
    if raw is None:
        return "."
    text = str(raw)
    candidate = Path(text.replace("\\", "/"))
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "working_directory is outside project_root.") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise ToolError(ToolErrorCode.WORKING_DIRECTORY_INVALID, "working_directory must be an existing real directory.")
    return relative.as_posix() if str(relative) != "." else "."


def _python_entrypoint_supported(root: Path, relative: str) -> bool:
    try:
        content = read_file(root, relative, max_bytes=65_536).content
    except ToolError:
        return False
    return bool(re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", content) or "app.run(" in content or "uvicorn.run(" in content)


def _read_package_json(root: Path, paths: set[str]) -> dict[str, Any] | None:
    package_path = next((path for path in paths if Path(path).name.casefold() == "package.json"), None)
    if package_path is None:
        return None
    try:
        content = read_file(root, package_path, max_bytes=65_536).content
        value = json.loads(content)
    except (ToolError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _node_candidate(package: dict[str, Any], paths: set[str], working_directory: str) -> ApplicationLaunchCandidate | None:
    scripts = package.get("scripts")
    if isinstance(scripts, dict) and isinstance(scripts.get("start"), str):
        tokens = scripts["start"].strip().split()
        if len(tokens) == 2 and tokens[0].casefold() in {"node", "nodejs"} and _safe_script_target(tokens[1]) and tokens[1].lstrip("./") in {Path(path).as_posix() for path in paths}:
            target = tokens[1].lstrip("./")
            node_executable = shutil.which("node") or "node"
            return ApplicationLaunchCandidate((node_executable, target), working_directory, "package.json scripts.start exact node entry", "node", "high", evidence=("package.json: scripts.start", f"{target}: target exists"))
    main = package.get("main")
    if isinstance(main, str) and _safe_script_target(main):
        target = main.lstrip("./")
        if target in {Path(path).as_posix() for path in paths}:
            node_executable = shutil.which("node") or "node"
            return ApplicationLaunchCandidate((node_executable, target), working_directory, "package.json main field with existing target", "node", "high", evidence=("package.json: main", f"{target}: target exists"))
    return None


def _safe_script_target(value: str) -> bool:
    return bool(value) and not any(token in value for token in ("..", ";", "|", "&", ">", "<", "$", "`", "\\")) and Path(value).suffix.casefold() in {".js", ".mjs", ".cjs", ".ts"}


def _safe_plan(plan: ApplicationRunPlan) -> ApplicationRunPlan:
    return replace(plan, argv=_safe_argv(plan.argv))


def _safe_command_result(result: CommandResult) -> CommandResult:
    return replace(result, argv=_safe_argv(result.argv))


def _status_from_command(result: CommandResult) -> tuple[ApplicationRunStatus, ApplicationFailureCode | None]:
    if result.timed_out:
        return ApplicationRunStatus.TIMED_OUT, ApplicationFailureCode.TIMEOUT
    if result.stdout_truncated or result.stderr_truncated:
        return ApplicationRunStatus.OUTPUT_LIMIT_REACHED, ApplicationFailureCode.OUTPUT_LIMIT_REACHED
    if result.start_failed:
        if result.error_code == ToolErrorCode.EXECUTABLE_NOT_FOUND.value:
            return ApplicationRunStatus.EXECUTABLE_NOT_FOUND, ApplicationFailureCode.EXECUTABLE_NOT_FOUND
        if result.error_code == ToolErrorCode.PERMISSION_DENIED.value:
            return ApplicationRunStatus.PERMISSION_DENIED, ApplicationFailureCode.PERMISSION_DENIED
        return ApplicationRunStatus.START_FAILED, ApplicationFailureCode.START_FAILED
    if result.succeeded:
        return ApplicationRunStatus.COMPLETED, None
    return ApplicationRunStatus.PROCESS_FAILED, ApplicationFailureCode.PROCESS_FAILED


def _safe_argv(argv: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    for value in argv:
        lowered = value.lower()
        if any(marker in lowered for marker in ("password=", "token=", "secret=", "api_key=", "apikey=")):
            output.append("<redacted>")
        else:
            output.append(value if len(value) <= 512 else value[:512] + "…")
    return tuple(output)


__all__ = [
    "ApplicationCommandResolver",
    "ApplicationFailureCode",
    "ApplicationLaunchCandidate",
    "ApplicationResolution",
    "ApplicationRunPlan",
    "ApplicationRunRequest",
    "ApplicationRunResult",
    "ApplicationRunStatus",
    "ApplicationRunner",
    "RunApplicationTool",
    "run_application",
]
