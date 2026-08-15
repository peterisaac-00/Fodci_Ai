"""Manual local smoke for Phase 5.1 command execution foundation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from backend_ai.tools import run_command


def _run(root: Path, argv: tuple[str, ...], **kwargs):
    result = run_command(argv, project_root=root, working_directory=".", **kwargs)
    assert result.started is True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-current-repo", action="store_true")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="fodci-phase51-command-") as directory:
        fixture = Path(directory)
        python_project = fixture / "python-project"
        python_project.mkdir()
        python_result = _run(python_project, (sys.executable, "--version"))
        assert python_result.succeeded is True

        node_project = fixture / "node-express-style-project"
        node_project.mkdir()
        (node_project / "package.json").write_text('{"name":"smoke-express-style"}\n', encoding="utf-8")
        (node_project / "server.js").write_text("console.log('placeholder')\n", encoding="utf-8")
        node_style = _run(node_project, (sys.executable, "-c", "print('node-style-fixture')"))
        assert node_style.stdout == "node-style-fixture\n"

        streams = _run(python_project, (sys.executable, "-c", "import sys; print('stdout'); print('stderr', file=sys.stderr); sys.exit(3)"))
        assert streams.stdout == "stdout\n"
        assert streams.stderr == "stderr\n"
        assert streams.exit_code == 3
        assert streams.succeeded is False

        timeout = _run(python_project, (sys.executable, "-c", "import time; time.sleep(1)"), timeout_seconds=0.05)
        assert timeout.timed_out is True

        shell_text = _run(python_project, (sys.executable, "-c", "import sys; print(sys.argv[1])", "&&"))
        assert shell_text.stdout == "&&\n"

    if args.inspect_current_repo:
        current = _run(Path.cwd(), (sys.executable, "--version"))
        assert current.succeeded is True
        print("Actual Fodci repository safe command smoke passed")

    print("Phase 5.1 command execution smoke passed")


if __name__ == "__main__":
    main()
