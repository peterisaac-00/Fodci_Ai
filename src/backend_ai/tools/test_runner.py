"""Bounded, evidence-backed test execution above policy and process layers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import sys
from typing import Any

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.command import (
    CommandRequest,
    CommandResult,
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_MAX_STDERR_BYTES,
    DEFAULT_MAX_STDOUT_BYTES,
    _validate_working_directory,
)
from backend_ai.tools.command_policy import (
    CommandDecision,
    CommandPolicy,
    CommandRiskLevel,
    PolicyRunCommandTool,
)
from backend_ai.tools.filesystem import _validate_root
from backend_ai.tools.project_context import ProjectContext, ProjectContextBuilder
from backend_ai.tools.read_file import read_file


class TestRunStatus(str, Enum):
    """Execution-level state; this enum intentionally has no semantic result states."""

    NOT_STARTED = "NOT_STARTED"
    RESOLVING = "RESOLVING"
    POLICY_DENIED = "POLICY_DENIED"
    START_FAILED = "START_FAILED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    TIMED_OUT = "TIMED_OUT"
    OUTPUT_LIMIT_REACHED = "OUTPUT_LIMIT_REACHED"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    NO_TEST_COMMAND = "NO_TEST_COMMAND"
    AMBIGUOUS_TEST_COMMAND = "AMBIGUOUS_TEST_COMMAND"
    INVALID_WORKING_DIRECTORY = "INVALID_WORKING_DIRECTORY"
    RESOLUTION_FAILED = "RESOLUTION_FAILED"


class TestRunFailureCode(str, Enum):
    """Technical failure classifications, not framework output interpretations."""

    NO_TEST_COMMAND = "NO_TEST_COMMAND"
    AMBIGUOUS_TEST_COMMAND = "AMBIGUOUS_TEST_COMMAND"
    POLICY_DENIED = "POLICY_DENIED"
    INVALID_WORKING_DIRECTORY = "INVALID_WORKING_DIRECTORY"
    START_FAILED = "START_FAILED"
    TIMEOUT = "TIMEOUT"
    OUTPUT_LIMIT_REACHED = "OUTPUT_LIMIT_REACHED"
    NONZERO_EXIT = "NONZERO_EXIT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    RESOLUTION_FAILED = "RESOLUTION_FAILED"


@dataclass(frozen=True, slots=True)
class TestRunRequest:
    """One explicit bounded test-run request."""

    project_root: Path | str
    working_directory: Path | str | None = None
    argv: tuple[str, ...] | list[str] | str | None = None
    test_target: str | None = None
    test_args: tuple[str, ...] | list[str] | None = None
    timeout_seconds: float | None = None
    max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES
    environment: Mapping[str, str] | None = None
    inherit_environment: bool = False


@dataclass(frozen=True, slots=True)
class TestFrameworkCandidate:
    """One deterministic evidence-backed test command candidate."""

    framework: str
    argv: tuple[str, ...]
    working_directory: str
    source: str
    confidence: str
    evidence: tuple[str, ...] = ()
    priority: int = 99
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "argv": list(_safe_argv(self.argv)),
            "working_directory": self.working_directory,
            "source": self.source,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "priority": self.priority,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class TestRunPlan:
    """Resolved command plan before policy evaluation."""

    argv: tuple[str, ...]
    working_directory: str
    framework: str
    source: str
    evidence: tuple[str, ...]
    confidence: str
    explicit: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(_safe_argv(self.argv)),
            "working_directory": self.working_directory,
            "framework": self.framework,
            "source": self.source,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "explicit": self.explicit,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class TestResolution:
    """Deterministic resolver output, including candidates when no choice is safe."""

    status: TestRunStatus
    plan: TestRunPlan | None
    candidates: tuple[TestFrameworkCandidate, ...]
    project_type: str
    frameworks: tuple[str, ...]
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]
    failure_code: TestRunFailureCode | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "project_type": self.project_type,
            "frameworks": list(self.frameworks),
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "failure_code": self.failure_code.value if self.failure_code else None,
        }


@dataclass(frozen=True, slots=True)
class TestRunResult:
    """Raw bounded execution facts for Phase 5.6; no output interpretation is done."""

    status: TestRunStatus
    plan: TestRunPlan | None
    command_result: CommandResult | None
    decision: CommandDecision | None
    failure_code: TestRunFailureCode | None
    project_type: str
    framework: str | None
    frameworks: tuple[str, ...]
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]
    candidates: tuple[TestFrameworkCandidate, ...] = ()

    @property
    def exit_code(self) -> int | None:
        return self.command_result.exit_code if self.command_result else None

    @property
    def stdout(self) -> str:
        return self.command_result.stdout if self.command_result else ""

    @property
    def stderr(self) -> str:
        return self.command_result.stderr if self.command_result else ""

    def to_dict(self) -> dict[str, Any]:
        raw = self.command_result.to_dict() if self.command_result else None
        return {
            "status": self.status.value,
            "plan": self.plan.to_dict() if self.plan else None,
            "command_result": raw,
            "decision": self.decision.to_dict() if self.decision else None,
            "failure_code": self.failure_code.value if self.failure_code else None,
            "project_type": self.project_type,
            "framework": self.framework,
            "frameworks": list(self.frameworks),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_bytes": raw.get("stdout_bytes", 0) if raw else 0,
            "stderr_bytes": raw.get("stderr_bytes", 0) if raw else 0,
            "stdout_truncated": raw.get("stdout_truncated", False) if raw else False,
            "stderr_truncated": raw.get("stderr_truncated", False) if raw else False,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class TestCommandResolver:
    """Resolve only bounded commands backed by existing structural/config evidence."""

    def __init__(self, *, context_builder: ProjectContextBuilder | None = None) -> None:
        self._context_builder = context_builder or ProjectContextBuilder()

    def resolve(self, request: TestRunRequest) -> TestResolution:
        root = _validate_root(request.project_root)
        _, _, working_directory = _validate_working_directory(root, request.working_directory or ".")
        context = self._context_builder.build(root)
        frameworks = tuple(sorted(
            (item.name for item in context.test_frameworks if item.name.casefold() != "generic tests"),
            key=str.casefold,
        ))
        evidence = context.evidence
        warnings = context.warnings
        target = _validate_target(root, request.test_target)
        test_args = _validate_test_args(request.test_args)

        if request.argv is not None:
            return self._resolve_explicit(
                request,
                context,
                working_directory,
                frameworks,
                evidence,
                warnings,
            )

        if context.truncated:
            return TestResolution(
                TestRunStatus.RESOLUTION_FAILED,
                None,
                (),
                context.project_type,
                frameworks,
                evidence,
                (*warnings, "Automatic test resolution requires complete bounded structural evidence."),
                TestRunFailureCode.RESOLUTION_FAILED,
            )

        package = _read_root_package_json(root, context)
        package_script = _package_test_script(package)
        candidates: list[TestFrameworkCandidate] = []

        if package_script is not None:
            script_framework = _script_framework(frameworks, package_script)
            if target is not None or test_args:
                warnings = (*warnings, "test_target/test_args are not appended to npm test; package scripts are kept as an exact argv boundary.")
            npm = shutil.which("npm") or "npm"
            candidates.append(TestFrameworkCandidate(
                script_framework,
                (npm, "test"),
                working_directory,
                "package.json scripts.test",
                "high",
                ("package.json: scripts.test is explicitly declared",),
                10,
            ))
        else:
            for detection in context.test_frameworks:
                name = detection.name
                if name == "pytest":
                    argv = [sys.executable, "-m", "pytest"]
                    argv.extend(_target_and_args(target, test_args))
                    candidates.append(TestFrameworkCandidate("pytest", tuple(argv), working_directory, "pytest project evidence", detection.confidence, detection.evidence, 20))
                elif name == "unittest":
                    argv = [sys.executable, "-m", "unittest"]
                    if target is None:
                        argv.extend(("discover", "-s", context.test_directories[0] if context.test_directories else "."))
                    else:
                        argv.append(target)
                    argv.extend(test_args)
                    candidates.append(TestFrameworkCandidate("unittest", tuple(argv), working_directory, "unittest project evidence", detection.confidence, detection.evidence, 20))
                elif name in {"Jest", "Vitest"}:
                    candidate = _direct_node_candidate(name, context, working_directory, target, test_args)
                    if candidate is not None:
                        candidates.append(candidate)

        candidates = sorted(candidates, key=lambda item: (item.priority, item.framework.casefold(), item.argv, item.source.casefold()))
        if not candidates:
            status = TestRunStatus.NO_TEST_COMMAND
            code = TestRunFailureCode.NO_TEST_COMMAND
            if context.project_type not in {"python", "node", "mixed"} and not frameworks:
                warning = "No supported test framework or explicit project test command was detected."
                warnings = (*warnings, warning)
            return TestResolution(status, None, (), context.project_type, frameworks, evidence, warnings, code)

        best_priority = candidates[0].priority
        best = tuple(candidate for candidate in candidates if candidate.priority == best_priority)
        if len(best) > 1:
            return TestResolution(
                TestRunStatus.AMBIGUOUS_TEST_COMMAND,
                None,
                tuple(candidates),
                context.project_type,
                frameworks,
                evidence,
                (*warnings, "Multiple equally ranked test commands remain; no command was selected."),
                TestRunFailureCode.AMBIGUOUS_TEST_COMMAND,
            )

        selected = best[0]
        plan = TestRunPlan(
            selected.argv,
            selected.working_directory,
            selected.framework,
            selected.source,
            selected.evidence,
            selected.confidence,
            False,
            (*selected.warnings, *warnings),
        )
        return TestResolution(TestRunStatus.RESOLVING, plan, tuple(candidates), context.project_type, frameworks, evidence, warnings)

    def _resolve_explicit(
        self,
        request: TestRunRequest,
        context: ProjectContext,
        working_directory: str,
        frameworks: tuple[str, ...],
        evidence: tuple[str, ...],
        warnings: tuple[str, ...],
    ) -> TestResolution:
        argv = _validate_explicit_argv(request.argv)
        plan = TestRunPlan(
            argv,
            working_directory,
            _explicit_framework(argv, frameworks),
            "explicit argv supplied by caller",
            (*evidence, "Explicit argv mode; no automatic test command was invented."),
            "explicit",
            True,
            warnings,
        )
        return TestResolution(TestRunStatus.RESOLVING, plan, (), context.project_type, frameworks, evidence, warnings)


class TestRunner:
    """Run one resolved test command through CommandPolicy and ProcessManager."""

    def __init__(self, *, resolver: TestCommandResolver | None = None, policy: CommandPolicy | None = None) -> None:
        self._resolver = resolver or TestCommandResolver()
        self._policy = policy

    def run(self, request: TestRunRequest) -> TestRunResult:
        try:
            resolution = self._resolver.resolve(request)
        except ToolError as exc:
            status = TestRunStatus.INVALID_WORKING_DIRECTORY if exc.code in {
                ToolErrorCode.PATH_OUTSIDE_ROOT,
                ToolErrorCode.WORKING_DIRECTORY_INVALID,
                ToolErrorCode.PERMISSION_DENIED,
            } else TestRunStatus.RESOLUTION_FAILED
            failure = TestRunFailureCode.INVALID_WORKING_DIRECTORY if status is TestRunStatus.INVALID_WORKING_DIRECTORY else TestRunFailureCode.RESOLUTION_FAILED
            return TestRunResult(status, None, None, None, failure, "unknown", None, (), (), (exc.message,))

        if resolution.plan is None:
            return TestRunResult(
                resolution.status,
                None,
                None,
                None,
                resolution.failure_code,
                resolution.project_type,
                None,
                resolution.frameworks,
                resolution.evidence,
                resolution.warnings,
                resolution.candidates,
            )

        plan = _safe_plan(resolution.plan)
        command_request = CommandRequest(
            resolution.plan.argv,
            request.project_root,
            resolution.plan.working_directory,
            request.environment,
            request.inherit_environment,
            request.timeout_seconds or DEFAULT_COMMAND_TIMEOUT_SECONDS,
            request.max_stdout_bytes,
            request.max_stderr_bytes,
        )
        policy = self._policy_for(plan, request, resolution)
        decision = policy.evaluate(command_request)
        if _contains_shell_syntax(resolution.plan.argv):
            decision = CommandDecision(
                False,
                CommandRiskLevel.DENIED,
                _safe_argv(resolution.plan.argv),
                "shell-syntax",
                "Shell operators and substitutions are not allowed in test argv.",
                (),
                ToolErrorCode.SHELL_BYPASS_ATTEMPT,
            )
        elif _contains_unsafe_argv_path(resolution.plan.argv):
            decision = CommandDecision(
                False,
                CommandRiskLevel.DENIED,
                _safe_argv(resolution.plan.argv),
                "argument-path",
                "Test argv contains an absolute, traversal, or sensitive file path.",
                (),
                ToolErrorCode.UNSAFE_ARGUMENT,
            )
        if not decision.allowed:
            return TestRunResult(
                TestRunStatus.POLICY_DENIED,
                plan,
                None,
                decision,
                TestRunFailureCode.POLICY_DENIED,
                resolution.project_type,
                resolution.plan.framework,
                resolution.frameworks,
                resolution.evidence,
                (*resolution.warnings, decision.reason),
                resolution.candidates,
            )

        try:
            executed = PolicyRunCommandTool(policy).run({
                "argv": list(resolution.plan.argv),
                "project_root": str(request.project_root),
                "working_directory": resolution.plan.working_directory,
                "environment": request.environment,
                "inherit_environment": request.inherit_environment,
                "timeout_seconds": command_request.timeout_seconds,
                "max_stdout_bytes": request.max_stdout_bytes,
                "max_stderr_bytes": request.max_stderr_bytes,
            })
        except ToolError as exc:
            return TestRunResult(
                TestRunStatus.EXECUTION_ERROR,
                plan,
                None,
                decision,
                TestRunFailureCode.EXECUTION_ERROR,
                resolution.project_type,
                resolution.plan.framework,
                resolution.frameworks,
                resolution.evidence,
                (*resolution.warnings, exc.message),
                resolution.candidates,
            )

        command_result = replace(executed.command_result, argv=_safe_argv(executed.command_result.argv))
        status, failure = _status_from_command(command_result)
        return TestRunResult(
            status,
            plan,
            command_result,
            executed.decision,
            failure,
            resolution.project_type,
            resolution.plan.framework,
            resolution.frameworks,
            resolution.evidence,
            (*resolution.warnings, *command_result.warnings),
            resolution.candidates,
        )

    def _policy_for(self, plan: TestRunPlan, request: TestRunRequest, resolution: TestResolution) -> CommandPolicy:
        policy = self._policy or CommandPolicy.default()
        for executable in (sys.executable, shutil.which("node"), shutil.which("npm")):
            if executable:
                policy = policy.with_executable_path(executable)
        if _basename(plan.argv[0]) == "npm":
            # npm is denied globally; this narrow exception is only for an explicit
            # package.json test script and exactly the two argv elements npm/test.
            package_script_evidence = any("package.json: scripts.test" in item for item in resolution.evidence) or "package.json scripts.test" in plan.source
            if package_script_evidence and tuple(plan.argv[1:]) == ("test",):
                policy = replace(
                    policy,
                    allowed_executables=tuple(sorted({*policy.allowed_executables, "npm"})),
                    denied_executables=tuple(item for item in policy.denied_executables if item.casefold() != "npm"),
                )
        return policy.with_exact_argv(plan.argv, name=f"test:{plan.source}")


class RunTestsTool:
    """Opt-in Tool wrapper for one bounded raw test execution."""

    name = "run_tests"
    description = "Resolve and run one bounded test command through evidence, policy, and ProcessManager."
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
                "test_target": {"type": "string"},
                "test_args": {"type": "array", "items": {"type": "string"}},
                "timeout_seconds": {"type": "number"},
                "max_stdout_bytes": {"type": "integer"},
                "max_stderr_bytes": {"type": "integer"},
                "environment": {"type": "object"},
                "inherit_environment": {"type": "boolean"},
            },
        },
    )

    def __init__(self, runner: TestRunner | None = None) -> None:
        self._runner = runner or TestRunner()

    def run(self, arguments: Mapping[str, Any]) -> TestRunResult:
        if not isinstance(arguments, Mapping) or "project_root" not in arguments:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "run_tests requires a project_root object argument.")
        if "command" in arguments:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "Shell command strings are not accepted; provide argv.")
        return self._runner.run(TestRunRequest(
            project_root=arguments["project_root"],
            working_directory=arguments.get("working_directory"),
            argv=arguments.get("argv"),
            test_target=arguments.get("test_target"),
            test_args=arguments.get("test_args"),
            timeout_seconds=arguments.get("timeout_seconds"),
            max_stdout_bytes=arguments.get("max_stdout_bytes", DEFAULT_MAX_STDOUT_BYTES),
            max_stderr_bytes=arguments.get("max_stderr_bytes", DEFAULT_MAX_STDERR_BYTES),
            environment=arguments.get("environment"),
            inherit_environment=arguments.get("inherit_environment", False),
        ))


def run_tests(request: TestRunRequest) -> TestRunResult:
    """Run one bounded test request without automatic retries or parsing."""

    return TestRunner().run(request)


def _read_root_package_json(root: Path, context: ProjectContext) -> dict[str, Any] | None:
    if "package.json" not in {Path(path).as_posix().casefold() for path in context.project_files}:
        return None
    path = root / "package.json"
    try:
        value = json.loads(read_file(root, "package.json", max_bytes=65_536).content)
    except (ToolError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _package_test_script(package: dict[str, Any] | None) -> str | None:
    if not package:
        return None
    scripts = package.get("scripts")
    if not isinstance(scripts, dict):
        return None
    value = scripts.get("test")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _script_framework(frameworks: tuple[str, ...], script: str) -> str:
    selected = tuple(name for name in frameworks if name in {"Jest", "Vitest"})
    if len(selected) == 1:
        return selected[0]
    lowered = script.casefold()
    if "jest" in lowered and "vitest" not in lowered:
        return "Jest"
    if "vitest" in lowered and "jest" not in lowered:
        return "Vitest"
    return "npm test"


def _direct_node_candidate(
    framework: str,
    context: ProjectContext,
    working_directory: str,
    target: str | None,
    test_args: tuple[str, ...],
) -> TestFrameworkCandidate | None:
    # node_modules is normally ignored by bounded discovery. A direct command is
    # therefore allowed only when the executable script itself is visible evidence.
    targets = {
        "Jest": "node_modules/jest/bin/jest.js",
        "Vitest": "node_modules/vitest/vitest.mjs",
    }
    target_path = targets[framework]
    visible = {Path(path).as_posix().casefold() for path in context.project_files}
    if target_path.casefold() not in visible:
        return None
    argv = [shutil.which("node") or "node", target_path]
    argv.extend(_target_and_args(target, test_args))
    detection = next((item for item in context.test_frameworks if item.name == framework), None)
    return TestFrameworkCandidate(
        framework,
        tuple(argv),
        working_directory,
        f"visible {framework} runner script",
        detection.confidence if detection else "medium",
        detection.evidence if detection else (f"{target_path}: runner script present",),
        20,
    )


def _target_and_args(target: str | None, test_args: tuple[str, ...]) -> tuple[str, ...]:
    return ((target,) if target is not None else ()) + test_args


def _explicit_framework(argv: tuple[str, ...], frameworks: tuple[str, ...]) -> str:
    lowered = tuple(item.casefold() for item in argv)
    if "pytest" in lowered:
        return "pytest"
    if "unittest" in lowered:
        return "unittest"
    if "jest" in lowered:
        return "Jest"
    if "vitest" in lowered:
        return "Vitest"
    if _basename(argv[0]) == "npm" and len(argv) == 2 and argv[1] == "test":
        return _script_framework(frameworks, "")
    return "explicit"


def _validate_explicit_argv(value: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "Explicit test argv must be a non-empty sequence of strings, not a command string.")
    argv = tuple(value)
    if any(not isinstance(item, str) or not item or "\x00" in item for item in argv):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "Explicit test argv entries must be non-empty NUL-free strings.")
    return argv


def _validate_test_args(value: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) > 32:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "test_args must be at most 32 argv strings.")
    args = tuple(value)
    if any(not isinstance(item, str) or not item or "\x00" in item for item in args):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "test_args entries must be non-empty NUL-free strings.")
    if _contains_shell_syntax(args):
        raise ToolError(ToolErrorCode.SHELL_BYPASS_ATTEMPT, "Shell operators and substitutions are not accepted in test_args.")
    if any(_unsafe_argument_path(item) for item in args):
        raise ToolError(ToolErrorCode.UNSAFE_ARGUMENT, "test_args cannot contain absolute or traversal paths.")
    return args


def _validate_target(root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or "\x00" in value or _contains_shell_syntax((value,)):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "test_target must be one safe argv string.")
    if _unsafe_argument_path(value):
        raise ToolError(ToolErrorCode.UNSAFE_ARGUMENT, "test_target cannot be absolute or traverse outside project_root.")
    path_like = "/" in value or "\\" in value or Path(value).suffix.casefold() in {".py", ".js", ".ts", ".tsx", ".jsx"}
    if path_like:
        if _sensitive_relative(value):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "test_target cannot refer to a sensitive file path.")
        candidate = (root / Path(value.replace("\\", "/"))).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "test_target is outside project_root.") from exc
        if not candidate.exists() or candidate.is_symlink():
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "test_target must refer to an existing non-symlink path.")
        _reject_symlink_components(root, candidate.relative_to(root).as_posix())
    elif not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", value):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "test_target must be a safe relative path or module name.")
    return value.replace("\\", "/") if path_like else value


def _reject_symlink_components(root: Path, relative: str) -> None:
    current = root
    for part in Path(relative).parts:
        current /= part
        try:
            if current.is_symlink():
                raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "test_target cannot traverse a symlink component.")
        except OSError as exc:
            raise ToolError(ToolErrorCode.PERMISSION_DENIED, "test_target components cannot be inspected.") from exc


def _unsafe_argument_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    pieces = [normalized]
    if "=" in normalized:
        pieces.append(normalized.split("=", 1)[1])
    return any(
        ".." in Path(piece).parts
        or Path(piece).is_absolute()
        or PureWindowsPath(piece).is_absolute()
        or bool(re.match(r"^[A-Za-z]:/", piece))
        for piece in pieces
    )


def _sensitive_relative(value: str) -> bool:
    sensitive_names = {".env"}
    sensitive_parts = ("credential", "secret", "private", "password")
    sensitive_suffixes = (".pem", ".key", ".crt", ".p12", ".pfx")
    path = Path(value.replace("\\", "/"))
    for part in path.parts:
        lowered = part.casefold()
        if lowered in sensitive_names or any(token in lowered for token in sensitive_parts) or lowered.endswith(sensitive_suffixes):
            return True
    return False


def _contains_unsafe_argv_path(argv: Sequence[str]) -> bool:
    return any(_unsafe_argument_path(value) or _sensitive_relative(value) for value in argv[1:])


def _contains_shell_syntax(argv: Sequence[str]) -> bool:
    markers = ("&&", "||", ";", "|", ">", ">>", "<", "$(", "`", "\n", "\r")
    return any(any(marker in item for marker in markers) for item in argv)


def _basename(value: str) -> str:
    return Path(value.replace("\\", "/")).name.casefold()


def _safe_plan(plan: TestRunPlan) -> TestRunPlan:
    return replace(plan, argv=_safe_argv(plan.argv))


def _safe_argv(argv: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    for value in argv:
        lowered = value.casefold()
        if any(marker in lowered for marker in ("password=", "token=", "secret=", "api_key=", "apikey=", "private_key=")):
            output.append("<redacted>")
        else:
            output.append(value if len(value) <= 512 else value[:512] + "…")
    return tuple(output)


def _status_from_command(result: CommandResult) -> tuple[TestRunStatus, TestRunFailureCode | None]:
    if result.timed_out:
        return TestRunStatus.TIMED_OUT, TestRunFailureCode.TIMEOUT
    if result.stdout_truncated or result.stderr_truncated:
        return TestRunStatus.OUTPUT_LIMIT_REACHED, TestRunFailureCode.OUTPUT_LIMIT_REACHED
    if result.start_failed:
        return TestRunStatus.START_FAILED, TestRunFailureCode.START_FAILED
    if result.completed:
        if result.exit_code not in (None, 0):
            return TestRunStatus.COMPLETED, TestRunFailureCode.NONZERO_EXIT
        return TestRunStatus.COMPLETED, None
    return TestRunStatus.EXECUTION_ERROR, TestRunFailureCode.EXECUTION_ERROR


__all__ = [
    "RunTestsTool",
    "TestCommandResolver",
    "TestFrameworkCandidate",
    "TestRunFailureCode",
    "TestRunPlan",
    "TestRunRequest",
    "TestRunResult",
    "TestRunStatus",
    "TestResolution",
    "TestRunner",
    "run_tests",
]
