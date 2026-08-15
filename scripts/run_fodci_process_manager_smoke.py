"""Manual local smoke for Phase 5.3 process management."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from backend_ai.tools import CommandPolicy, PolicyRunCommandTool, ProcessManager, CommandRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-current-repo", action="store_true")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="fodci-phase53-process-") as directory:
        root = Path(directory)
        manager = ProcessManager(termination_grace_seconds=0.05)

        normal = manager.execute(CommandRequest((sys.executable, "-c", "print('normal')"), root, ".", inherit_environment=False, timeout_seconds=1.0))
        assert normal.succeeded is True and normal.exit_code == 0

        nonzero = manager.execute(CommandRequest((sys.executable, "-c", "import sys; print('stderr', file=sys.stderr); sys.exit(2)"), root, ".", inherit_environment=False, timeout_seconds=1.0))
        assert nonzero.completed is True and nonzero.succeeded is False and nonzero.exit_code == 2

        limited = manager.execute(CommandRequest((sys.executable, "-c", "print('x' * 200000)"), root, ".", inherit_environment=False, timeout_seconds=1.0, max_stdout_bytes=64))
        assert limited.stdout_truncated is True and limited.stdout_bytes <= 64

        timeout = manager.execute(CommandRequest((sys.executable, "-c", "import time; print('partial', flush=True); time.sleep(2)"), root, ".", inherit_environment=False, timeout_seconds=0.05))
        assert timeout.timed_out is True and timeout.completed is False and timeout.termination_attempted is True

        policy = CommandPolicy.default().with_executable_path(sys.executable)
        approved = PolicyRunCommandTool(policy).run({"argv": [sys.executable, "--version"], "project_root": str(root), "working_directory": ".", "inherit_environment": False})
        assert approved.command_result is not None and approved.command_result.succeeded is True

    if args.inspect_current_repo:
        root = Path.cwd()
        policy = CommandPolicy.default().with_executable_path(sys.executable)
        result = PolicyRunCommandTool(policy).run({"argv": [sys.executable, "--version"], "project_root": str(root), "working_directory": ".", "inherit_environment": False})
        assert result.command_result is not None and result.command_result.succeeded is True
        print("Actual Fodci repository process-management smoke passed")

    print("Phase 5.3 process-management smoke passed")


if __name__ == "__main__":
    main()
