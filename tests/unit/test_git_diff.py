from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from backend_ai.agent import AgentLoop, ToolRegistry
from backend_ai.tools import (
    GitDiffResult,
    GitDiffTool,
    ToolError,
    ToolErrorCode,
    git_diff,
)


def _run_git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        text=True,
        capture_output=True,
        check=check,
    )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _run_git(root, "init", "-q")
    _run_git(root, "config", "user.name", "Test User")
    _run_git(root, "config", "user.email", "test@example.invalid")
    return root


def _write(root: Path, relative: str, content: str | bytes) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content, encoding="utf-8")
    return target


def _entry(result: GitDiffResult, relative: str):
    return next(item for item in result.changed_files if item.relative_path == relative)


def test_non_git_directory_returns_structured_empty_result_without_initializing(tmp_path: Path) -> None:
    root = tmp_path / "not-repo"
    root.mkdir()

    result = git_diff(root)

    assert isinstance(result, GitDiffResult)
    assert result.is_git_repository is False
    assert result.changed_files == ()
    assert result.staged_diff == ""
    assert result.unstaged_diff == ""
    assert any("not a Git repository" in warning for warning in result.warnings)
    assert not (root / ".git").exists()


def test_clean_repository_and_tool_protocol_are_read_only(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "app.py", "print('ok')\n")
    _run_git(root, "add", "app.py")
    _run_git(root, "commit", "-qm", "initial")
    before = _run_git(root, "status", "--porcelain=v1").stdout

    result = git_diff(root)

    assert result.is_git_repository is True
    assert result.files_changed == 0
    assert result.changed_files == ()
    assert result.staged_diff == ""
    assert result.unstaged_diff == ""
    assert result.current_branch in {"main", "master", None}
    assert result.head
    assert _run_git(root, "status", "--porcelain=v1").stdout == before
    assert (root / ".git").exists()


def test_unstaged_modified_and_untracked_files_are_structured(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "app.py", "old\n")
    _run_git(root, "add", "app.py")
    _run_git(root, "commit", "-qm", "initial")
    _write(root, "app.py", "new\n")
    _write(root, "notes.txt", "untracked\n")

    result = git_diff(root)

    modified = _entry(result, "app.py")
    untracked = _entry(result, "notes.txt")
    assert result.files_changed == 2
    assert modified.status == "modified"
    assert modified.staged is False
    assert modified.unstaged is True
    assert modified.unstaged_insertions == 1
    assert modified.unstaged_deletions == 1
    assert untracked.status == "untracked"
    assert untracked.untracked is True
    assert untracked.staged is False
    assert untracked.unstaged is False
    assert "old" in result.unstaged_diff
    assert "new" in result.unstaged_diff
    assert "notes.txt" not in result.unstaged_diff


def test_staged_and_unstaged_changes_are_separated(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "app.py", "zero\n")
    _run_git(root, "add", "app.py")
    _run_git(root, "commit", "-qm", "initial")
    _write(root, "app.py", "one\n")
    _run_git(root, "add", "app.py")
    _write(root, "app.py", "two\n")

    result = git_diff(root)
    entry = _entry(result, "app.py")

    assert entry.staged is True
    assert entry.unstaged is True
    assert entry.staged_insertions == 1
    assert entry.unstaged_insertions == 1
    assert "one" in result.staged_diff
    assert "two" in result.unstaged_diff
    assert "one" in result.combined_diff
    assert "two" in result.combined_diff


def test_added_deleted_and_renamed_files_are_reported(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "old.txt", "old\n")
    _write(root, "remove.txt", "remove\n")
    _run_git(root, "add", ".")
    _run_git(root, "commit", "-qm", "initial")
    (root / "remove.txt").unlink()
    (root / "old.txt").rename(root / "new.txt")
    _write(root, "added.txt", "added\n")
    _run_git(root, "add", "-A")

    result = git_diff(root)

    assert _entry(result, "remove.txt").status == "deleted"
    renamed = _entry(result, "new.txt")
    assert renamed.status in {"renamed", "modified", "added"}
    if renamed.status == "renamed":
        assert renamed.old_path == "old.txt"
    assert _entry(result, "added.txt").status == "added"
    assert "remove.txt" in result.staged_diff
    assert "new.txt" in result.staged_diff


def test_binary_changes_are_not_decoded_as_file_content(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "blob.bin", b"\x00\xff\x01\x02")
    _run_git(root, "add", "blob.bin")
    _run_git(root, "commit", "-qm", "initial")
    _write(root, "blob.bin", b"\x00\xfe\x01\x03")

    result = git_diff(root)
    entry = _entry(result, "blob.bin")

    assert entry.is_binary is True
    assert "\ufffd" not in result.unstaged_diff
    assert b"\xff" not in result.unstaged_diff.encode("utf-8")


def test_paths_are_relative_and_deterministic(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "z.txt", "z\n")
    _write(root, "nested/a.txt", "a\n")
    _run_git(root, "add", ".")
    _run_git(root, "commit", "-qm", "initial")
    _write(root, "z.txt", "changed\n")
    _write(root, "nested/new.txt", "new\n")

    first = git_diff(root).to_dict()
    second = git_diff(root).to_dict()

    assert first == second
    for item in first["changed_files"]:
        path = item["relative_path"]
        assert not Path(path).is_absolute()
        assert "\\" not in path
        assert ".." not in Path(path).parts
        assert str(root) not in path


def test_limits_mark_truncation_and_bound_diff_and_changed_files(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _write(root, "app.txt", "base\n")
    _run_git(root, "add", ".")
    _run_git(root, "commit", "-qm", "initial")
    for index in range(5):
        _write(root, f"file-{index}.txt", "x\n")
    _write(root, "app.txt", "changed\n")

    result = git_diff(root, max_diff_bytes=40, max_diff_lines=2, max_changed_files=2)

    assert result.truncated is True
    assert result.truncation_reason
    assert result.files_changed <= 2
    assert len(result.combined_diff.encode("utf-8")) <= 40


def test_invalid_arguments_and_git_tool_schema_are_structured(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    tool = GitDiffTool()

    assert tool.name == "git_diff"
    assert tool.metadata.input_schema["required"] == ["project_root"]
    with pytest.raises(ToolError) as missing:
        tool.run({})
    assert missing.value.code is ToolErrorCode.INVALID_ARGUMENT
    with pytest.raises(ToolError) as non_mapping:
        tool.run([])
    assert non_mapping.value.code is ToolErrorCode.INVALID_ARGUMENT
    with pytest.raises(ToolError) as bad_limit:
        git_diff(root, max_diff_bytes=0)
    assert bad_limit.value.code is ToolErrorCode.INVALID_ARGUMENT


def test_git_unavailable_is_structured_without_mutating_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path)
    git_module = __import__("backend_ai.tools.git_diff", fromlist=["subprocess"])
    real_popen = git_module.subprocess.Popen

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(git_module.subprocess, "Popen", unavailable)
    with pytest.raises(ToolError) as raised:
        git_diff(root)
    assert raised.value.code is ToolErrorCode.GIT_NOT_AVAILABLE
    monkeypatch.setattr(git_module.subprocess, "Popen", real_popen)
    assert not (root / ".git" / "index.lock").exists()


def test_timeout_is_structured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path)
    git_module = __import__("backend_ai.tools.git_diff", fromlist=["_bounded_process_output"])
    monkeypatch.setattr(git_module, "_bounded_process_output", lambda *args, **kwargs: (_ for _ in ()).throw(ToolError(ToolErrorCode.GIT_TIMEOUT, "timeout")))
    with pytest.raises(ToolError) as raised:
        git_diff(root)
    assert raised.value.code is ToolErrorCode.GIT_TIMEOUT


def test_registry_is_opt_in_and_agent_loop_default_remains_without_git_diff(tmp_path: Path) -> None:
    assert "git_diff" not in ToolRegistry.default().names()
    assert "git_diff" in ToolRegistry.with_git_inspection().names()
    assert "git_diff" not in ToolRegistry.with_file_modification().names()

    root = _repo(tmp_path)
    target = _write(root, "app.py", "old\n")
    result = AgentLoop(type("Engine", (), {"tokenizer": type("Tokenizer", (), {"encode": lambda self, value: list(value.encode())})(), "generate": lambda self, prompt: type("Output", (), {"generated_text": "FINAL: inspect"})()})()).run("Inspect", root)
    assert result.final_answer == "inspect"
    assert target.read_text(encoding="utf-8") == "old\n"
