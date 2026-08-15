from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

import pytest

from backend_ai.agent.registry import ToolRegistry
from backend_ai.tools import (
    RunTestsTool,
    TestCommandResolver,
    TestRunFailureCode,
    TestRunRequest,
    TestRunStatus,
    TestRunner,
    CommandPolicy,
    ToolError,
    ToolErrorCode,
)
import backend_ai.tools.command_policy as policy_module


def _root(tmp_path: Path, name: str = "project") -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


def _write_script(root: Path, name: str, source: str) -> Path:
    path = root / name
    path.write_text(source, encoding="utf-8")
    return path


def _python_fixture(root: Path, source: str = "print('fixture-ok')\n") -> Path:
    return _write_script(root, "run_test.py", source)


def test_pytest_project_is_detected_and_executed_from_evidence(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "pyproject.toml").write_text("[project]\ndependencies = ['pytest']\n", encoding="utf-8")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_basic.py").write_text("def test_basic():\n    assert 1 + 1 == 2\n", encoding="utf-8")

    result = TestRunner().run(TestRunRequest(root, test_args=("-p", "no:cacheprovider"), timeout_seconds=5.0))

    assert result.status is TestRunStatus.COMPLETED
    assert result.framework == "pytest"
    assert result.plan is not None and result.plan.argv[1:3] == ("-m", "pytest")
    assert result.command_result is not None and result.command_result.exit_code == 0
    assert result.failure_code is None


def test_unittest_project_is_detected_and_executed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_basic.py").write_text(
        "import unittest\n\nclass BasicTest(unittest.TestCase):\n    def test_basic(self):\n        self.assertEqual(2, 2)\n",
        encoding="utf-8",
    )

    result = TestRunner().run(TestRunRequest(root, timeout_seconds=5.0))

    assert result.status is TestRunStatus.COMPLETED
    assert result.framework == "unittest"
    assert result.plan is not None and result.plan.argv[1:] == ("-m", "unittest", "discover", "-s", "tests")
    assert result.command_result is not None and result.command_result.exit_code == 0


def test_jest_and_vitest_framework_evidence_is_detected_without_guessing(tmp_path: Path) -> None:
    jest = _root(tmp_path, "jest")
    (jest / "package.json").write_text(json.dumps({"devDependencies": {"jest": "^1.0.0"}}), encoding="utf-8")
    (jest / "jest.config.js").write_text("module.exports = {};\n", encoding="utf-8")
    (jest / "tests").mkdir()
    (jest / "tests" / "basic.test.js").write_text("test('basic', () => {});\n", encoding="utf-8")
    jest_context = TestCommandResolver()._context_builder.build(jest)
    assert "Jest" in {item.name for item in jest_context.test_frameworks}
    assert TestRunner().run(TestRunRequest(jest)).status is TestRunStatus.NO_TEST_COMMAND

    vitest = _root(tmp_path, "vitest")
    (vitest / "package.json").write_text(json.dumps({"devDependencies": {"vitest": "^1.0.0"}}), encoding="utf-8")
    (vitest / "vitest.config.ts").write_text("export default {};\n", encoding="utf-8")
    (vitest / "tests").mkdir()
    (vitest / "tests" / "basic.test.ts").write_text("test('basic', () => {});\n", encoding="utf-8")
    vitest_context = TestCommandResolver()._context_builder.build(vitest)
    assert "Vitest" in {item.name for item in vitest_context.test_frameworks}
    assert TestRunner().run(TestRunRequest(vitest)).status is TestRunStatus.NO_TEST_COMMAND


def test_explicit_package_test_script_is_preferred_and_runs_without_shell_argv(tmp_path: Path) -> None:
    if shutil.which("npm") is None:
        pytest.skip("npm is unavailable in this test environment")
    root = _root(tmp_path)
    (root / "package.json").write_text(json.dumps({"scripts": {"test": "node test_runner.js"}}), encoding="utf-8")
    (root / "test_runner.js").write_text("console.log('npm-test-ok');\n", encoding="utf-8")

    policy = CommandPolicy.default().with_inherited_environment()
    result = TestRunner(policy=policy).run(TestRunRequest(root, inherit_environment=True, timeout_seconds=5.0))

    assert result.status is TestRunStatus.COMPLETED
    assert result.plan is not None and result.plan.argv[1:] == ("test",)
    assert result.plan.source == "package.json scripts.test"
    assert result.command_result is not None and "npm-test-ok" in result.stdout


def test_explicit_safe_argv_is_preserved_and_executed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    script = _python_fixture(root)
    result = TestRunner().run(TestRunRequest(root, argv=(sys.executable, script.name), timeout_seconds=2.0))

    assert result.status is TestRunStatus.COMPLETED
    assert result.plan is not None and result.plan.explicit is True
    assert result.plan.argv == (sys.executable, script.name)
    assert result.command_result is not None and result.stdout == "fixture-ok\n"


def test_explicit_unsafe_argv_is_policy_denied_before_process_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    called = False

    def unexpected_process(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("denied test command must not spawn a process")

    monkeypatch.setattr(policy_module.ProcessManager, "execute", unexpected_process)
    result = TestRunner().run(TestRunRequest(root, argv=("bash", "-c", "echo unsafe"), timeout_seconds=1.0))

    assert result.status is TestRunStatus.POLICY_DENIED
    assert result.failure_code is TestRunFailureCode.POLICY_DENIED
    assert called is False
    assert result.decision is not None and result.decision.error_code is ToolErrorCode.SHELL_BYPASS_ATTEMPT


def test_shell_syntax_path_traversal_and_sensitive_paths_are_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path)
    for argv in (
        (sys.executable, "run_test.py", "&&", "echo unsafe"),
        (sys.executable, "../outside.py"),
        (sys.executable, ".env"),
    ):
        result = TestRunner().run(TestRunRequest(root, argv=argv, timeout_seconds=1.0))
        assert result.status is TestRunStatus.POLICY_DENIED
        assert result.failure_code is TestRunFailureCode.POLICY_DENIED


def test_missing_test_command_and_ambiguous_test_command_are_structured(tmp_path: Path) -> None:
    missing = _root(tmp_path, "missing")
    (missing / "README.md").write_text("documentation only\n", encoding="utf-8")
    no_command = TestRunner().run(TestRunRequest(missing))
    assert no_command.status is TestRunStatus.NO_TEST_COMMAND
    assert no_command.plan is None

    ambiguous = _root(tmp_path, "ambiguous")
    (ambiguous / "pyproject.toml").write_text("[project]\ndependencies=['pytest']\n", encoding="utf-8")
    (ambiguous / "tests").mkdir()
    (ambiguous / "tests" / "test_both.py").write_text("import unittest\n\ndef test_x():\n    assert True\n", encoding="utf-8")
    result = TestRunner().run(TestRunRequest(ambiguous))
    assert result.status is TestRunStatus.AMBIGUOUS_TEST_COMMAND
    assert result.plan is None
    assert len(result.candidates) == 2


def test_resolution_is_deterministic_and_does_not_create_parser_or_semantic_states(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "pyproject.toml").write_text("[project]\ndependencies=['pytest']\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_basic.py").write_text("def test_basic():\n    assert True\n", encoding="utf-8")
    resolver = TestCommandResolver()
    first = resolver.resolve(TestRunRequest(root, test_args=("-p", "no:cacheprovider")))
    second = resolver.resolve(TestRunRequest(root, test_args=("-p", "no:cacheprovider")))
    assert first.to_dict() == second.to_dict()
    result = TestRunner().run(TestRunRequest(root, test_args=("-p", "no:cacheprovider"), timeout_seconds=5.0))
    serialized = result.to_dict()
    assert serialized["status"] == "COMPLETED"
    assert all(value not in serialized["status"] for value in ("PASS", "FAIL", "ERROR"))
    assert not (Path(__file__).parents[2] / "src/backend_ai/tools/test_result_parser.py").exists()


def test_success_nonzero_stdout_stderr_and_invalid_utf8_are_raw_execution_facts(tmp_path: Path) -> None:
    root = _root(tmp_path)
    success = _python_fixture(root, "print('out')\nprint('err', file=__import__('sys').stderr)\n")
    ok = TestRunner().run(TestRunRequest(root, argv=(sys.executable, success.name), timeout_seconds=2.0))
    assert ok.status is TestRunStatus.COMPLETED
    assert ok.exit_code == 0 and ok.stdout == "out\n" and ok.stderr == "err\n"

    failed = _write_script(root, "failed.py", "import sys\nprint('failed-output')\nsys.exit(3)\n")
    nonzero = TestRunner().run(TestRunRequest(root, argv=(sys.executable, failed.name), timeout_seconds=2.0))
    assert nonzero.status is TestRunStatus.COMPLETED
    assert nonzero.failure_code is TestRunFailureCode.NONZERO_EXIT
    assert nonzero.exit_code == 3 and "failed-output" in nonzero.stdout

    invalid = _write_script(root, "invalid.py", "import sys\nsys.stdout.buffer.write(b'\\xff')\n")
    invalid_result = TestRunner().run(TestRunRequest(root, argv=(sys.executable, invalid.name), timeout_seconds=2.0))
    assert invalid_result.status is TestRunStatus.COMPLETED
    assert invalid_result.command_result is not None and invalid_result.command_result.stdout_utf8_valid is False


def test_timeout_and_output_limits_preserve_process_metadata(tmp_path: Path) -> None:
    root = _root(tmp_path)
    slow = _write_script(root, "slow.py", "import time\nprint('before-timeout', flush=True)\ntime.sleep(2)\n")
    timed = TestRunner().run(TestRunRequest(root, argv=(sys.executable, slow.name), timeout_seconds=0.05))
    assert timed.status is TestRunStatus.TIMED_OUT
    assert timed.command_result is not None and timed.command_result.timed_out is True
    assert timed.command_result.termination_attempted is True

    noisy = _write_script(root, "noisy.py", "print('x' * 10000)\n")
    limited = TestRunner().run(TestRunRequest(root, argv=(sys.executable, noisy.name), timeout_seconds=2.0, max_stdout_bytes=32))
    assert limited.status is TestRunStatus.OUTPUT_LIMIT_REACHED
    assert limited.failure_code is TestRunFailureCode.OUTPUT_LIMIT_REACHED
    assert limited.command_result is not None and limited.command_result.stdout_truncated is True


def test_invalid_working_directory_and_target_validation_are_bounded(tmp_path: Path) -> None:
    root = _root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    invalid_cwd = TestRunner().run(TestRunRequest(root, working_directory=outside, argv=(sys.executable, "--version")))
    assert invalid_cwd.status is TestRunStatus.INVALID_WORKING_DIRECTORY

    with pytest.raises(ToolError) as traversal:
        TestCommandResolver().resolve(TestRunRequest(root, test_target="../outside.py"))
    assert traversal.value.code in {ToolErrorCode.UNSAFE_ARGUMENT, ToolErrorCode.PATH_OUTSIDE_ROOT}


def test_no_install_network_or_git_mutation_and_no_runner_file_mutation(tmp_path: Path) -> None:
    root = _root(tmp_path)
    script = _python_fixture(root, "print('read-only-run')\n")
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    for argv in (
        (sys.executable, "-m", "pip", "install", "anything"),
        ("npm", "install", "anything"),
        ("curl", "https://example.invalid"),
        ("git", "commit", "-m", "bad"),
    ):
        result = TestRunner().run(TestRunRequest(root, argv=argv, timeout_seconds=1.0))
        assert result.status is TestRunStatus.POLICY_DENIED
    safe = TestRunner().run(TestRunRequest(root, argv=(sys.executable, script.name), timeout_seconds=2.0))
    assert safe.status is TestRunStatus.COMPLETED
    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    assert before == after


def test_environment_values_are_not_exposed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    script = _python_fixture(root)
    result = TestRunner().run(TestRunRequest(root, argv=(sys.executable, script.name), environment={"TOKEN": "secret-value"}, timeout_seconds=1.0))
    assert result.status is TestRunStatus.POLICY_DENIED
    assert "secret-value" not in str(result.to_dict())
    assert "TOKEN" not in str(result.to_dict())


def test_tool_protocol_registry_opt_in_and_agent_default_remain_read_only() -> None:
    default = ToolRegistry.default()
    opt_in = ToolRegistry.with_test_execution()
    assert "run_tests" not in default.names()
    assert "run_tests" in opt_in.names()
    tool = RunTestsTool()
    assert tool.metadata.name == "run_tests"
    assert tool.metadata.input_schema["properties"]["argv"]["type"] == "array"
    with pytest.raises(ToolError):
        tool.run({"project_root": ".", "command": "pytest"})


# These public domain classes are imported into this test module for assertions;
# they are not pytest test classes.
TestCommandResolver.__test__ = False
TestRunFailureCode.__test__ = False
TestRunRequest.__test__ = False
TestRunStatus.__test__ = False
TestRunner.__test__ = False
