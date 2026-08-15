from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from backend_ai.agent.registry import ToolRegistry
from backend_ai.tools import (
    ApplicationRunRequest,
    ApplicationRunStatus,
    ApplicationRunner,
    CommandPolicy,
    RunApplicationTool,
    ToolError,
    ToolErrorCode,
)
import backend_ai.tools.command_policy as policy_module


def _root(tmp_path: Path, name: str = "project") -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


def test_python_application_is_resolved_from_evidence_and_run_through_process_manager(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "main.py").write_text("if __name__ == '__main__':\n    print('python-app')\n", encoding="utf-8")
    result = ApplicationRunner().run(ApplicationRunRequest(root, timeout_seconds=1.0))

    assert result.status is ApplicationRunStatus.COMPLETED
    assert result.plan is not None
    assert result.plan.explicit is False
    assert result.plan.argv[-1] == "main.py"
    assert result.command_result is not None and result.command_result.stdout == "python-app\n"
    assert result.command_result.process_state == "CLEANED_UP"
    assert result.failure_code is None


def test_node_application_requires_existing_explicit_start_script(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "package.json").write_text(json.dumps({"name": "fixture", "scripts": {"start": "node server.js"}}), encoding="utf-8")
    (root / "server.js").write_text("console.log('node-app')\n", encoding="utf-8")
    result = ApplicationRunner().run(ApplicationRunRequest(root, timeout_seconds=1.0))

    assert result.status is ApplicationRunStatus.COMPLETED
    assert result.plan is not None and Path(result.plan.argv[0]).name == "node" and result.plan.argv[1] == "server.js"
    assert result.command_result is not None and result.command_result.stdout == "node-app\n"


def test_missing_entrypoint_is_structured_and_does_not_guess(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = ApplicationRunner().run(ApplicationRunRequest(root))

    assert result.status is ApplicationRunStatus.NO_APPLICATION_ENTRYPOINT
    assert result.failure_code is not None and result.failure_code.value == "NO_APPLICATION_ENTRYPOINT"
    assert result.plan is None


def test_ambiguous_project_returns_candidates_without_arbitrary_execution(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "main.py").write_text("if __name__ == '__main__':\n    print('python')\n", encoding="utf-8")
    (root / "package.json").write_text(json.dumps({"scripts": {"start": "node server.js"}}), encoding="utf-8")
    (root / "server.js").write_text("console.log('node')\n", encoding="utf-8")
    result = ApplicationRunner().run(ApplicationRunRequest(root))

    assert result.status is ApplicationRunStatus.AMBIGUOUS_ENTRYPOINT
    assert len(result.candidates) == 2
    assert result.command_result is None


def test_unsupported_project_is_not_guessed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "README.md").write_text("documentation only\n", encoding="utf-8")
    result = ApplicationRunner().run(ApplicationRunRequest(root))
    assert result.status is ApplicationRunStatus.UNSUPPORTED_PROJECT
    assert result.plan is None


def test_explicit_safe_argv_is_preserved_and_policy_checked(tmp_path: Path) -> None:
    root = _root(tmp_path)
    result = ApplicationRunner().run(ApplicationRunRequest(root, argv=(sys.executable, "--version"), timeout_seconds=1.0))
    assert result.status is ApplicationRunStatus.COMPLETED
    assert result.plan is not None and result.plan.explicit is True
    assert result.plan.argv == (sys.executable, "--version")


def test_explicit_unsafe_argv_is_denied_before_process_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    called = False

    def unexpected_process(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("policy-denied application must not start a process")

    monkeypatch.setattr(policy_module.ProcessManager, "execute", unexpected_process)
    result = ApplicationRunner().run(ApplicationRunRequest(root, argv=("bash", "-c", "echo unsafe"), timeout_seconds=1.0))

    assert result.status is ApplicationRunStatus.POLICY_DENIED
    assert result.failure_code is not None and result.failure_code.value == "POLICY_DENIED"
    assert called is False
    assert result.decision is not None and result.decision.error_code is ToolErrorCode.SHELL_BYPASS_ATTEMPT


def test_invalid_working_directory_is_structured(tmp_path: Path) -> None:
    root = _root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    result = ApplicationRunner().run(ApplicationRunRequest(root, working_directory=outside, argv=(sys.executable, "--version")))
    assert result.status is ApplicationRunStatus.INVALID_WORKING_DIRECTORY
    assert result.failure_code is not None and result.failure_code.value == "INVALID_WORKING_DIRECTORY"


def test_timeout_and_output_limits_preserve_process_manager_metadata(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "main.py").write_text("import time\nif __name__ == '__main__':\n    print('partial', flush=True)\n    time.sleep(2)\n", encoding="utf-8")
    timed = ApplicationRunner().run(ApplicationRunRequest(root, timeout_seconds=0.05))
    assert timed.status is ApplicationRunStatus.TIMED_OUT
    assert timed.command_result is not None and timed.command_result.timed_out is True

    noisy = _root(tmp_path, "noisy")
    (noisy / "main.py").write_text("if __name__ == '__main__':\n    print('x' * 10000)\n", encoding="utf-8")
    limited = ApplicationRunner().run(ApplicationRunRequest(noisy, timeout_seconds=1.0, max_stdout_bytes=32))
    assert limited.status is ApplicationRunStatus.OUTPUT_LIMIT_REACHED
    assert limited.command_result is not None and limited.command_result.stdout_truncated is True


def test_invalid_utf8_and_nonzero_application_exit_are_structured(tmp_path: Path) -> None:
    invalid = _root(tmp_path, "invalid")
    (invalid / "main.py").write_text("import sys\nif __name__ == '__main__':\n    sys.stdout.buffer.write(b'\\xff')\n", encoding="utf-8")
    result = ApplicationRunner().run(ApplicationRunRequest(invalid, timeout_seconds=1.0))
    assert result.status is ApplicationRunStatus.COMPLETED
    assert result.command_result is not None and result.command_result.stdout_utf8_valid is False

    failed = _root(tmp_path, "failed")
    (failed / "main.py").write_text("import sys\nif __name__ == '__main__':\n    sys.exit(3)\n", encoding="utf-8")
    failure = ApplicationRunner().run(ApplicationRunRequest(failed, timeout_seconds=1.0))
    assert failure.status is ApplicationRunStatus.PROCESS_FAILED
    assert failure.command_result is not None and failure.command_result.exit_code == 3


def test_sensitive_files_are_not_used_as_resolution_content_and_result_serialization_is_deterministic(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / ".env").write_text("SECRET=do-not-read\n", encoding="utf-8")
    (root / "main.py").write_text("if __name__ == '__main__':\n    print('safe')\n", encoding="utf-8")
    first = ApplicationRunner().run(ApplicationRunRequest(root, timeout_seconds=1.0))
    second = ApplicationRunner().run(ApplicationRunRequest(root, timeout_seconds=1.0))

    first_dict = first.to_dict()
    second_dict = second.to_dict()
    first_dict["command_result"].pop("duration_seconds", None)
    second_dict["command_result"].pop("duration_seconds", None)
    assert first_dict == second_dict
    assert "do-not-read" not in str(first_dict)
    assert "SECRET" not in str(first_dict)


def test_tool_registry_application_runner_is_opt_in_and_agent_default_unchanged() -> None:
    assert "run_application" not in ToolRegistry.default().names()
    assert "run_application" in ToolRegistry.with_application_execution().names()

    tool = RunApplicationTool()
    assert tool.metadata.name == "run_application"
