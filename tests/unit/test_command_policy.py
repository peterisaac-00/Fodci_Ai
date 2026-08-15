from __future__ import annotations

from pathlib import Path
import sys

import pytest

from backend_ai.agent.registry import ToolRegistry
from backend_ai.tools import (
    CommandPolicy,
    CommandRequest,
    CommandRiskLevel,
    PolicyRunCommandTool,
    ToolError,
    ToolErrorCode,
)
import backend_ai.tools.command_policy as policy_module


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def _request(root: Path, argv: tuple[str, ...], **kwargs) -> CommandRequest:
    return CommandRequest(argv, root, ".", inherit_environment=False, **kwargs)


def _python_policy() -> CommandPolicy:
    return CommandPolicy.default().with_executable_path(sys.executable)


def test_default_policy_denies_unknown_and_allows_only_explicitly_safe_version_commands(tmp_path: Path) -> None:
    root = _project(tmp_path)
    policy = _python_policy()
    allowed = policy.evaluate(_request(root, (sys.executable, "--version")))
    denied = policy.evaluate(_request(root, ("pytest", "--version")))

    assert allowed.allowed is True
    assert allowed.risk_level is CommandRiskLevel.SAFE
    assert denied.allowed is False
    assert denied.risk_level is CommandRiskLevel.DENIED
    assert denied.error_code is ToolErrorCode.UNSAFE_EXECUTABLE


def test_policy_decisions_are_deterministic_and_secret_safe(tmp_path: Path) -> None:
    root = _project(tmp_path)
    request = _request(root, ("unknown-tool", "token=super-secret"))
    first = CommandPolicy.default().evaluate(request)
    second = CommandPolicy.default().evaluate(request)

    assert first == second
    assert "super-secret" not in str(first.to_dict())
    assert "<redacted>" in first.normalized_argv


def test_shell_interpreters_and_shell_emulation_are_denied(tmp_path: Path) -> None:
    root = _project(tmp_path)
    policy = CommandPolicy.default()
    for argv in (
        ("bash", "-c", "echo unsafe"),
        ("sh", "-c", "echo unsafe"),
        ("powershell", "-Command", "Write-Output unsafe"),
        ("pwsh", "-Command", "Write-Output unsafe"),
        ("cmd.exe", "/c", "echo unsafe"),
        ("command", "-c", "echo unsafe"),
    ):
        decision = policy.evaluate(_request(root, argv))
        assert decision.allowed is False
        assert decision.error_code is ToolErrorCode.SHELL_BYPASS_ATTEMPT


def test_dangerous_categories_are_explicitly_denied(tmp_path: Path) -> None:
    root = _project(tmp_path)
    policy = CommandPolicy.default()
    cases = {
        ("sudo", "id"): ToolErrorCode.COMMAND_DENIED,
        ("rm", "-rf", "project"): ToolErrorCode.COMMAND_DENIED,
        ("curl", "https://example.invalid"): ToolErrorCode.NETWORK_COMMAND_DENIED,
        ("pip", "install", "package"): ToolErrorCode.PACKAGE_OPERATION_DENIED,
        ("git", "commit", "-m", "message"): ToolErrorCode.GIT_MUTATION_DENIED,
        ("git", "push", "--force"): ToolErrorCode.GIT_MUTATION_DENIED,
    }
    for argv, code in cases.items():
        decision = policy.evaluate(_request(root, argv))
        assert decision.allowed is False
        assert decision.error_code is code


def test_read_only_git_inspection_can_be_allowed_without_general_git_allowance(tmp_path: Path) -> None:
    root = _project(tmp_path)
    policy = CommandPolicy.default()
    allowed = policy.evaluate(_request(root, ("git", "status", "--short")))
    denied = policy.evaluate(_request(root, ("git", "reset", "--hard")))

    assert allowed.allowed is True
    assert allowed.risk_level is CommandRiskLevel.SAFE
    assert denied.allowed is False
    assert denied.error_code is ToolErrorCode.GIT_MUTATION_DENIED


def test_python_script_and_interpreter_arguments_are_not_implicitly_allowed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    policy = _python_policy()
    script = policy.evaluate(_request(root, (sys.executable, "-c", "print('unsafe')")))
    package = policy.evaluate(_request(root, (sys.executable, "-m", "pip", "--version")))
    unknown_module = policy.evaluate(_request(root, (sys.executable, "-m", "pytest", "--version")))

    assert script.allowed is False
    assert script.error_code is ToolErrorCode.UNSAFE_ARGUMENT
    assert package.allowed is False
    assert package.error_code is ToolErrorCode.UNSAFE_ARGUMENT
    assert unknown_module.allowed is False
    assert unknown_module.error_code is ToolErrorCode.COMMAND_NOT_ALLOWED


def test_path_arguments_and_working_directory_are_bounded(tmp_path: Path) -> None:
    root = _project(tmp_path)
    policy = _python_policy()
    for argv in (
        (sys.executable, "--version", "../outside"),
        (sys.executable, "--version", r"C:\\Windows\\System32"),
        (sys.executable, "--version", r"\\\\server\\share"),
    ):
        decision = policy.evaluate(_request(root, argv))
        assert decision.allowed is False
        assert decision.error_code is ToolErrorCode.UNSAFE_ARGUMENT

    outside = tmp_path / "outside"
    outside.mkdir()
    bad_cwd = CommandRequest((sys.executable, "--version"), root, outside, inherit_environment=False)
    cwd_decision = policy.evaluate(bad_cwd)
    assert cwd_decision.allowed is False
    assert cwd_decision.error_code is ToolErrorCode.UNSAFE_WORKING_DIRECTORY


def test_environment_is_allowlisted_and_values_never_leak(tmp_path: Path) -> None:
    root = _project(tmp_path)
    policy = _python_policy()
    denied = policy.evaluate(_request(root, (sys.executable, "--version"), environment={"PYTHONPATH": "top-secret"}))
    unknown = policy.evaluate(_request(root, (sys.executable, "--version"), environment={"UNTRUSTED": "secret"}))
    inherited = policy.evaluate(CommandRequest((sys.executable, "--version"), root, "."))

    assert denied.error_code is ToolErrorCode.ENVIRONMENT_NOT_ALLOWED
    assert unknown.error_code is ToolErrorCode.ENVIRONMENT_NOT_ALLOWED
    assert inherited.error_code is ToolErrorCode.ENVIRONMENT_NOT_ALLOWED
    assert "top-secret" not in str(denied.to_dict())
    assert "secret" not in str(unknown.to_dict())


def test_exact_override_is_bounded_and_cannot_enable_shell_or_arbitrary_code(tmp_path: Path) -> None:
    root = _project(tmp_path)
    policy = _python_policy().with_exact_argv((sys.executable, "--version"))
    allowed = policy.evaluate(_request(root, (sys.executable, "--version")))
    shell = policy.evaluate(_request(root, ("bash", "-c", "echo unsafe")))
    arbitrary = policy.evaluate(_request(root, (sys.executable, "-c", "print('unsafe')")))

    assert allowed.allowed is True
    assert shell.allowed is False
    assert arbitrary.allowed is False


def test_policy_wrapper_executes_only_after_allow_and_denied_commands_never_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    policy = _python_policy()
    tool = PolicyRunCommandTool(policy)
    allowed = tool.run({"argv": [sys.executable, "--version"], "project_root": str(root), "working_directory": ".", "inherit_environment": False})
    assert allowed.decision.allowed is True
    assert allowed.command_result is not None and allowed.command_result.succeeded is True

    def unexpected_process(*args, **kwargs):
        raise AssertionError("denied policy must not call run_command")

    monkeypatch.setattr(policy_module, "run_command", unexpected_process)
    with pytest.raises(ToolError) as raised:
        tool.run({"argv": ["rm", "-rf", "x"], "project_root": str(root), "working_directory": ".", "inherit_environment": False})
    assert raised.value.code is ToolErrorCode.COMMAND_DENIED


def test_policy_failure_does_not_mutate_project_and_result_is_not_a_process_result(tmp_path: Path) -> None:
    root = _project(tmp_path)
    marker = root / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")
    tool = PolicyRunCommandTool(CommandPolicy.default())

    with pytest.raises(ToolError) as raised:
        tool.run({"argv": ["unknown-command", "../marker.txt"], "project_root": str(root), "working_directory": ".", "inherit_environment": False})
    assert raised.value.code in {ToolErrorCode.UNSAFE_ARGUMENT, ToolErrorCode.UNSAFE_EXECUTABLE}
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_policy_registry_is_opt_in_and_phase_5_1_registry_remains_available(tmp_path: Path) -> None:
    assert "run_command_with_policy" not in ToolRegistry.default().names()
    assert "run_command_with_policy" in ToolRegistry.with_command_policy().names()
    assert "run_command" in ToolRegistry.with_command_execution().names()
