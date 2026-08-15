"""Deterministic command safety policy above the Phase 5.1 executor."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import ntpath
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.command import CommandRequest, CommandResult, run_command


class CommandRiskLevel(str, Enum):
    SAFE = "SAFE"
    RESTRICTED = "RESTRICTED"
    DANGEROUS = "DANGEROUS"
    DENIED = "DENIED"


@dataclass(frozen=True, slots=True)
class CommandDecision:
    """Immutable policy decision with secret-safe normalized command metadata."""

    allowed: bool
    risk_level: CommandRiskLevel
    normalized_argv: tuple[str, ...]
    matched_rule: str
    reason: str
    warnings: tuple[str, ...] = ()
    error_code: ToolErrorCode | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "risk_level": self.risk_level.value,
            "normalized_argv": list(self.normalized_argv),
            "matched_rule": self.matched_rule,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "error_code": self.error_code.value if self.error_code else None,
        }


@dataclass(frozen=True, slots=True)
class CommandRule:
    """A bounded exact-argv or prefix rule used by a CommandPolicy."""

    name: str
    argv_prefix: tuple[str, ...]
    risk_level: CommandRiskLevel = CommandRiskLevel.SAFE


@dataclass(frozen=True, slots=True)
class CommandPolicy:
    """Conservative deny-by-default policy configuration.

    The default recognizes only version/info commands and read-only Git
    inspection. Explicit exact argv rules can be added for future callers;
    there is no allow-anything switch.
    """

    allowed_executables: tuple[str, ...] = ("python", "python3", "py", "node", "git")
    allowed_executable_paths: tuple[str, ...] = ()
    allowed_exact_argv: tuple[tuple[str, ...], ...] = ()
    allowed_rules: tuple[CommandRule, ...] = ()
    denied_executables: tuple[str, ...] = (
        "bash", "sh", "zsh", "fish", "pwsh", "powershell", "cmd", "cmd.exe",
        "sudo", "doas", "runas", "rm", "rmdir", "del", "erase", "format",
        "shutdown", "reboot", "kill", "pkill", "systemctl", "service", "reg",
        "regedit", "diskpart", "docker", "podman", "curl", "wget", "scp", "ssh",
        "rsync", "nc", "netcat", "ftp", "telnet", "pip", "npm", "pnpm", "yarn",
        "bun", "apt", "apt-get", "brew", "cargo",
    )
    allow_inherited_environment: bool = False
    allowed_environment_names: tuple[str, ...] = (
        "LANG", "LC_ALL", "LC_CTYPE", "PYTHONUNBUFFERED", "PYTHONIOENCODING",
    )
    denied_environment_names: tuple[str, ...] = (
        "PYTHONPATH", "NODE_PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH", "BASH_ENV", "ENV", "SHELLOPTS", "PYTHONSTARTUP", "PYTHONINSPECT",
    )
    allow_project_relative_arguments: bool = False

    @classmethod
    def default(cls) -> "CommandPolicy":
        return cls()

    def with_inherited_environment(self) -> "CommandPolicy":
        return replace(self, allow_inherited_environment=True)

    def with_executable_path(self, executable: str | Path) -> "CommandPolicy":
        return replace(self, allowed_executable_paths=(*self.allowed_executable_paths, str(Path(executable).resolve())) )

    def with_exact_argv(self, argv: tuple[str, ...] | list[str], *, name: str = "explicit-exact-argv") -> "CommandPolicy":
        return replace(self, allowed_exact_argv=(*self.allowed_exact_argv, tuple(argv)), allowed_rules=(*self.allowed_rules, CommandRule(name, tuple(argv))))

    def evaluate(self, request: CommandRequest) -> CommandDecision:
        """Evaluate without spawning a process or mutating the filesystem."""

        normalized = _safe_argv(request.argv)
        if not isinstance(request, CommandRequest):
            return _deny(normalized, "invalid-request", "Policy requires a CommandRequest.", ToolErrorCode.COMMAND_NOT_ALLOWED)
        if not request.argv or any(not isinstance(item, str) or not item or "\x00" in item for item in request.argv):
            return _deny(normalized, "invalid-argv", "argv must be non-empty NUL-free strings.", ToolErrorCode.COMMAND_NOT_ALLOWED)
        executable = request.argv[0]
        basename = Path(executable.replace("\\", "/")).name.lower()
        if _is_shell_bypass(request.argv, basename):
            return _deny(normalized, "shell-bypass", "Interpreter wrapper or shell emulation is not allowed.", ToolErrorCode.SHELL_BYPASS_ATTEMPT)
        if basename in self.denied_executables or basename in {item.lower() for item in self.denied_executables}:
            return _deny(normalized, f"dangerous-executable:{basename}", "The executable belongs to a denied dangerous category.", _category_error(basename))
        if _dangerous_argument_family(basename, request.argv[1:]):
            return _deny(normalized, "dangerous-arguments", "Arguments indicate a destructive, privileged, package, network, or system-management operation.", _argument_error(basename, request.argv[1:]))
        environment_decision = self._evaluate_environment(request, normalized)
        if environment_decision is not None:
            return environment_decision
        try:
            from backend_ai.tools.command import _validate_working_directory
            _validate_working_directory(request.project_root, request.working_directory)
        except ToolError as exc:
            return _deny(normalized, "working-directory", "working_directory violates the explicit project-root policy.", ToolErrorCode.UNSAFE_WORKING_DIRECTORY)
        executable_decision = self._evaluate_executable(request, basename, normalized)
        if executable_decision is not None:
            return executable_decision
        path_decision = _evaluate_argument_paths(request, normalized, self.allow_project_relative_arguments)
        if path_decision is not None:
            return path_decision
        for rule in self.allowed_rules:
            if _argv_matches(rule.argv_prefix, request.argv):
                return CommandDecision(True, rule.risk_level, normalized, rule.name, "Explicit policy rule allowed this argv.")
        for exact in self.allowed_exact_argv:
            if tuple(request.argv) == exact:
                return CommandDecision(True, CommandRiskLevel.RESTRICTED, normalized, "explicit-exact-argv", "Explicit exact argv override allowed this command.")
        if _default_safe_command(basename, request.argv[1:]):
            return CommandDecision(True, CommandRiskLevel.SAFE, normalized, f"default-read-only:{basename}", "Recognized bounded version/info or read-only Git command.")
        return _deny(normalized, "default-deny", "Command is not explicitly recognized as safe by the conservative default policy.", ToolErrorCode.COMMAND_NOT_ALLOWED)

    def _evaluate_executable(self, request: CommandRequest, basename: str, normalized: tuple[str, ...]) -> CommandDecision | None:
        executable = request.argv[0]
        if _looks_like_path(executable):
            try:
                resolved = str(Path(executable).expanduser().resolve())
            except OSError:
                resolved = executable
            if resolved not in {str(Path(item).expanduser().resolve()) for item in self.allowed_executable_paths}:
                return _deny(normalized, "executable-path", "Executable paths must be explicitly approved by policy.", ToolErrorCode.UNSAFE_EXECUTABLE)
        if basename not in {item.lower() for item in self.allowed_executables} and not _is_python_name(basename):
            return _deny(normalized, "unknown-executable", "Executable is not in the conservative policy allowlist.", ToolErrorCode.UNSAFE_EXECUTABLE)
        return None

    def _evaluate_environment(self, request: CommandRequest, normalized: tuple[str, ...]) -> CommandDecision | None:
        if request.inherit_environment and not self.allow_inherited_environment:
            return _deny(normalized, "inherited-environment", "Inherited environment is disabled by the conservative policy.", ToolErrorCode.ENVIRONMENT_NOT_ALLOWED)
        if request.environment is None:
            return None
        if not isinstance(request.environment, Mapping):
            return _deny(normalized, "environment-shape", "Environment must be a mapping of strings.", ToolErrorCode.ENVIRONMENT_NOT_ALLOWED)
        allowed = {name.upper() for name in self.allowed_environment_names}
        denied = {name.upper() for name in self.denied_environment_names}
        for key, value in request.environment.items():
            if not isinstance(key, str) or not isinstance(value, str) or key.upper() in denied or key.upper() not in allowed:
                return _deny(normalized, "environment-variable", "Environment contains a variable outside the policy allowlist.", ToolErrorCode.ENVIRONMENT_NOT_ALLOWED)
        return None


@dataclass(frozen=True, slots=True)
class PolicyCommandResult:
    """Decision plus optional low-level result; denied commands have no process result."""

    decision: CommandDecision
    command_result: CommandResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "command_result": self.command_result.to_dict() if self.command_result else None,
        }


class PolicyRunCommandTool:
    """Opt-in policy wrapper around the existing RunCommand foundation."""

    name = "run_command_with_policy"
    description = "Evaluate an explicit argv command under a conservative policy before execution."
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["argv", "project_root", "working_directory"],
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "project_root": {"type": "string"},
                "working_directory": {"type": "string"},
                "environment": {"type": "object"},
                "inherit_environment": {"type": "boolean"},
                "timeout_seconds": {"type": "number"},
                "max_stdout_bytes": {"type": "integer"},
                "max_stderr_bytes": {"type": "integer"},
            },
        },
    )

    def __init__(self, policy: CommandPolicy | None = None) -> None:
        self.policy = policy or CommandPolicy.default()

    def run(self, arguments: Mapping[str, Any]) -> PolicyCommandResult:
        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.COMMAND_NOT_ALLOWED, "Policy command arguments must be an object.")
        request = CommandRequest(
            tuple(arguments.get("argv", ())),
            arguments.get("project_root"),
            arguments.get("working_directory"),
            arguments.get("environment"),
            arguments.get("inherit_environment", True),
            arguments.get("timeout_seconds", 10.0),
            arguments.get("max_stdout_bytes", 1_048_576),
            arguments.get("max_stderr_bytes", 1_048_576),
        )
        decision = self.policy.evaluate(request)
        if not decision.allowed:
            raise ToolError(decision.error_code or ToolErrorCode.COMMAND_DENIED, decision.reason)
        return PolicyCommandResult(decision, run_command(request))


def _deny(argv: tuple[str, ...], rule: str, reason: str, code: ToolErrorCode) -> CommandDecision:
    return CommandDecision(False, CommandRiskLevel.DENIED, argv, rule, reason, (), code)


def _safe_argv(argv: Any) -> tuple[str, ...]:
    if isinstance(argv, (tuple, list)):
        return tuple(_redact(str(item)) for item in argv)
    return ()


def _redact(value: str) -> str:
    lowered = value.lower()
    sensitive_markers = ("password=", "token=", "secret=", "api_key=", "apikey=", "private_key=")
    if any(marker in lowered for marker in sensitive_markers):
        return "<redacted>"
    return value if len(value) <= 512 else value[:512] + "…"


def _is_python_name(name: str) -> bool:
    return name in {"python", "python3", "python3.11", "python3.12", "py"} or name.startswith("python3.")


def _looks_like_path(value: str) -> bool:
    return "/" in value or "\\" in value or Path(value).is_absolute() or ntpath.isabs(value) or PureWindowsPath(value).is_absolute()


def _is_shell_bypass(argv: tuple[str, ...], basename: str) -> bool:
    shell_names = {"bash", "sh", "zsh", "fish", "pwsh", "powershell", "cmd", "cmd.exe"}
    if basename in shell_names:
        return True
    lowered = [item.lower() for item in argv[1:]]
    return basename in shell_names or (basename in {"command", "command.com"} and any(item in {"-c", "/c", "-command"} for item in lowered))


def _dangerous_argument_family(basename: str, arguments: tuple[str, ...]) -> bool:
    lowered = [item.lower() for item in arguments]
    if basename == "git":
        return bool(lowered and (lowered[0] in {"commit", "merge", "rebase", "cherry-pick", "push", "pull", "fetch", "reset", "clean", "stash", "remote", "branch", "tag"} and not (lowered[0] == "branch" and "--show-current" in lowered)))
    if basename in {"python", "python3", "py"} and any(item in {"-m", "-c"} for item in lowered):
        if "-m" in lowered:
            index = lowered.index("-m")
            if index + 1 < len(lowered) and lowered[index + 1] in {"pip", "ensurepip", "venv", "http.server"}:
                return True
        return "-c" in lowered
    dangerous_flags = {"--force", "-f", "--hard", "-rf", "-fr", "--recursive", "--delete"}
    return any(item in dangerous_flags for item in lowered)


def _category_error(basename: str) -> ToolErrorCode:
    if basename in {"curl", "wget", "scp", "ssh", "rsync", "nc", "netcat", "ftp", "telnet"}:
        return ToolErrorCode.NETWORK_COMMAND_DENIED
    if basename in {"pip", "npm", "pnpm", "yarn", "bun", "apt", "apt-get", "brew", "cargo"}:
        return ToolErrorCode.PACKAGE_OPERATION_DENIED
    if basename == "git":
        return ToolErrorCode.GIT_MUTATION_DENIED
    return ToolErrorCode.COMMAND_DENIED


def _argument_error(basename: str, arguments: tuple[str, ...]) -> ToolErrorCode:
    lowered = [item.lower() for item in arguments]
    if basename == "git":
        return ToolErrorCode.GIT_MUTATION_DENIED
    if basename in {"python", "python3", "py"} and ("-m" in lowered or "-c" in lowered):
        return ToolErrorCode.UNSAFE_ARGUMENT
    if any(item in {"pip", "install", "npm", "pnpm", "yarn", "cargo"} for item in lowered):
        return ToolErrorCode.PACKAGE_OPERATION_DENIED
    return ToolErrorCode.UNSAFE_ARGUMENT


def _default_safe_command(basename: str, arguments: tuple[str, ...]) -> bool:
    lowered = tuple(item.lower() for item in arguments)
    if _is_python_name(basename):
        return lowered in {("--version",), ("-v",), ("-v",), ("-v",)}
    if basename == "node":
        return lowered in {("--version",), ("-v",)}
    if basename == "git" and lowered:
        if lowered[0] in {"status", "diff", "log"}:
            return not any(item in {"--hard", "--force", "--no-index"} for item in lowered)
        if lowered[0] == "branch" and "--show-current" in lowered:
            return True
        if lowered[0] == "rev-parse" and set(lowered[1:]).issubset({"--show-toplevel", "--verify", "head"}):
            return True
    return False


def _argv_matches(prefix: tuple[str, ...], argv: tuple[str, ...]) -> bool:
    return len(argv) >= len(prefix) and tuple(argv[: len(prefix)]) == prefix


def _evaluate_argument_paths(request: CommandRequest, normalized: tuple[str, ...], allow_project_relative: bool) -> CommandDecision | None:
    for argument in request.argv[1:]:
        if argument.startswith("-") and "=" not in argument:
            continue
        if ".." in Path(argument.replace("\\", "/")).parts or ntpath.isabs(argument) or PureWindowsPath(argument).is_absolute():
            return _deny(normalized, "argument-path", "Command argument refers to an unbounded or absolute path.", ToolErrorCode.UNSAFE_ARGUMENT)
    return None


__all__ = [
    "CommandDecision",
    "CommandPolicy",
    "CommandRiskLevel",
    "CommandRule",
    "PolicyCommandResult",
    "PolicyRunCommandTool",
]
