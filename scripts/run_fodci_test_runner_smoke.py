"""Manual Phase 5.5 smoke checks using only temporary projects and existing executables."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile

from backend_ai.agent.registry import ToolRegistry
from backend_ai.tools import (
    CommandPolicy,
    TestRunRequest,
    TestRunStatus,
    TestRunner,
)


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run(root: Path, request: TestRunRequest):
    result = TestRunner().run(request)
    assert result.project_type in {"python", "node", "mixed", "empty", "other", "unknown"}
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fodci-phase55-") as raw:
        base = Path(raw)

        pytest_root = base / "pytest-project"
        pytest_root.mkdir()
        _write(pytest_root, "pyproject.toml", "[project]\ndependencies=['pytest']\n")
        _write(pytest_root, "tests/test_basic.py", "def test_basic():\n    assert True\n")
        pytest_result = _run(pytest_root, TestRunRequest(pytest_root, test_args=("-p", "no:cacheprovider"), timeout_seconds=5.0))
        assert pytest_result.status is TestRunStatus.COMPLETED
        assert pytest_result.framework == "pytest"

        unittest_root = base / "unittest-project"
        unittest_root.mkdir()
        _write(unittest_root, "tests/test_basic.py", "import unittest\n\nclass Basic(unittest.TestCase):\n    def test_basic(self):\n        self.assertTrue(True)\n")
        unittest_result = _run(unittest_root, TestRunRequest(unittest_root, timeout_seconds=5.0))
        assert unittest_result.status is TestRunStatus.COMPLETED
        assert unittest_result.framework == "unittest"

        if shutil.which("npm") and shutil.which("node"):
            node_root = base / "node-jest-project"
            node_root.mkdir()
            _write(node_root, "package.json", json.dumps({"scripts": {"test": "node test_runner.js"}, "devDependencies": {"jest": "^1.0.0"}}))
            _write(node_root, "jest.config.js", "module.exports = {};\n")
            _write(node_root, "test_runner.js", "console.log('node-test-ok');\n")
            node_policy = CommandPolicy.default().with_inherited_environment()
            node_result = TestRunner(policy=node_policy).run(TestRunRequest(node_root, inherit_environment=True, timeout_seconds=5.0))
            assert node_result.status is TestRunStatus.COMPLETED
            assert node_result.plan is not None and node_result.plan.source == "package.json scripts.test"
            assert node_result.framework == "Jest"

        explicit_root = base / "explicit-project"
        explicit_root.mkdir()
        _write(explicit_root, "safe_test.py", "print('explicit-ok')\n")
        explicit_result = _run(explicit_root, TestRunRequest(explicit_root, argv=(sys.executable, "safe_test.py"), timeout_seconds=2.0))
        assert explicit_result.status is TestRunStatus.COMPLETED

        denied = _run(explicit_root, TestRunRequest(explicit_root, argv=("bash", "-c", "echo denied"), timeout_seconds=1.0))
        assert denied.status is TestRunStatus.POLICY_DENIED

        _write(explicit_root, "slow.py", "import time\ntime.sleep(2)\n")
        timed = _run(explicit_root, TestRunRequest(explicit_root, argv=(sys.executable, "slow.py"), timeout_seconds=0.05))
        assert timed.status is TestRunStatus.TIMED_OUT

        _write(explicit_root, "failed.py", "import sys\nsys.exit(3)\n")
        failed = _run(explicit_root, TestRunRequest(explicit_root, argv=(sys.executable, "failed.py"), timeout_seconds=2.0))
        assert failed.status is TestRunStatus.COMPLETED and failed.exit_code == 3

        _write(explicit_root, "noisy.py", "print('x' * 10000)\n")
        limited = _run(explicit_root, TestRunRequest(explicit_root, argv=(sys.executable, "noisy.py"), max_stdout_bytes=32, timeout_seconds=2.0))
        assert limited.status is TestRunStatus.OUTPUT_LIMIT_REACHED

        ambiguous_root = base / "ambiguous-project"
        ambiguous_root.mkdir()
        _write(ambiguous_root, "pyproject.toml", "[project]\ndependencies=['pytest']\n")
        _write(ambiguous_root, "tests/test_both.py", "import unittest\ndef test_x():\n    assert True\n")
        ambiguous = _run(ambiguous_root, TestRunRequest(ambiguous_root))
        assert ambiguous.status is TestRunStatus.AMBIGUOUS_TEST_COMMAND

        no_test_root = base / "no-test-project"
        no_test_root.mkdir()
        _write(no_test_root, "README.md", "documentation only\n")
        no_test = _run(no_test_root, TestRunRequest(no_test_root))
        assert no_test.status is TestRunStatus.NO_TEST_COMMAND

    actual_root = Path(__file__).resolve().parents[1]
    actual = TestRunner().run(TestRunRequest(actual_root, argv=(sys.executable, "--version"), timeout_seconds=2.0))
    assert actual.status is TestRunStatus.COMPLETED
    assert "run_tests" not in ToolRegistry.default().names()
    assert "run_tests" in ToolRegistry.with_test_execution().names()
    print("Phase 5.5 test-runner smoke passed")


if __name__ == "__main__":
    main()
