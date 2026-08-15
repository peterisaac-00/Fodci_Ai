"""Manual local smoke for Phase 5.4 application runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from backend_ai.tools import ApplicationRunRequest, ApplicationRunStatus, ApplicationRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-current-repo", action="store_true")
    args = parser.parse_args()
    runner = ApplicationRunner()

    with TemporaryDirectory(prefix="fodci-phase54-application-") as directory:
        root = Path(directory)
        python_root = root / "python-app"
        python_root.mkdir()
        (python_root / "main.py").write_text("if __name__ == '__main__':\n    print('python-smoke')\n", encoding="utf-8")
        python_result = runner.run(ApplicationRunRequest(python_root, timeout_seconds=1.0))
        assert python_result.status is ApplicationRunStatus.COMPLETED

        node_root = root / "node-app"
        node_root.mkdir()
        (node_root / "package.json").write_text(json.dumps({"scripts": {"start": "node server.js"}}), encoding="utf-8")
        (node_root / "server.js").write_text("console.log('node-smoke')\n", encoding="utf-8")
        node_result = runner.run(ApplicationRunRequest(node_root, timeout_seconds=1.0))
        assert node_result.status is ApplicationRunStatus.COMPLETED

        ambiguous = root / "ambiguous"
        ambiguous.mkdir()
        (ambiguous / "main.py").write_text("if __name__ == '__main__':\n    print('python')\n", encoding="utf-8")
        (ambiguous / "package.json").write_text(json.dumps({"scripts": {"start": "node server.js"}}), encoding="utf-8")
        (ambiguous / "server.js").write_text("console.log('node')\n", encoding="utf-8")
        ambiguous_result = runner.run(ApplicationRunRequest(ambiguous))
        assert ambiguous_result.status is ApplicationRunStatus.AMBIGUOUS_ENTRYPOINT
        assert ambiguous_result.command_result is None

        unsafe = runner.run(ApplicationRunRequest(root, argv=("bash", "-c", "echo unsafe"), timeout_seconds=1.0))
        assert unsafe.status is ApplicationRunStatus.POLICY_DENIED
        assert unsafe.command_result is None

        timeout_root = root / "timeout"
        timeout_root.mkdir()
        (timeout_root / "main.py").write_text("import time\nif __name__ == '__main__':\n    print('partial', flush=True)\n    time.sleep(2)\n", encoding="utf-8")
        timeout = runner.run(ApplicationRunRequest(timeout_root, timeout_seconds=0.05))
        assert timeout.status is ApplicationRunStatus.TIMED_OUT
        assert timeout.command_result is not None and timeout.command_result.termination_attempted is True

    if args.inspect_current_repo:
        result = runner.run(ApplicationRunRequest(Path.cwd(), argv=(sys.executable, "--version"), timeout_seconds=1.0))
        assert result.status is ApplicationRunStatus.COMPLETED
        print("Actual Fodci repository application-runner smoke passed")

    print("Phase 5.4 application-runner smoke passed")


if __name__ == "__main__":
    main()
