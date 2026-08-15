"""Manual local smoke for Phase 5.2 command safety policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from backend_ai.tools import CommandPolicy, PolicyRunCommandTool, ToolError, ToolErrorCode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-current-repo", action="store_true")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="fodci-phase52-policy-") as directory:
        root = Path(directory)
        policy = CommandPolicy.default().with_executable_path(sys.executable)
        tool = PolicyRunCommandTool(policy)
        safe = tool.run({"argv": [sys.executable, "--version"], "project_root": str(root), "working_directory": ".", "inherit_environment": False})
        assert safe.decision.allowed is True
        assert safe.command_result is not None and safe.command_result.succeeded is True

        for argv, code in (
            (("bash", "-c", "echo unsafe"), ToolErrorCode.SHELL_BYPASS_ATTEMPT),
            (("pip", "install", "package"), ToolErrorCode.PACKAGE_OPERATION_DENIED),
            (("curl", "https://example.invalid"), ToolErrorCode.NETWORK_COMMAND_DENIED),
            (("git", "reset", "--hard"), ToolErrorCode.GIT_MUTATION_DENIED),
        ):
            try:
                tool.run({"argv": list(argv), "project_root": str(root), "working_directory": ".", "inherit_environment": False})
            except ToolError as exc:
                assert exc.code is code
            else:
                raise AssertionError(f"Expected policy denial for {argv!r}")

    if args.inspect_current_repo:
        root = Path.cwd()
        safe = PolicyRunCommandTool(CommandPolicy.default().with_executable_path(sys.executable)).run({"argv": [sys.executable, "--version"], "project_root": str(root), "working_directory": ".", "inherit_environment": False})
        assert safe.command_result is not None and safe.command_result.succeeded is True
        print("Actual Fodci repository policy smoke passed")

    print("Phase 5.2 command policy smoke passed")


if __name__ == "__main__":
    main()
