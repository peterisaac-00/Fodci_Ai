from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from backend_ai.agent import AgentLoop, ToolRegistry
from backend_ai.tools import GitStatusResult, GitStatusTool, ToolError, ToolErrorCode, git_status


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(("git", *args), cwd=root, text=True, capture_output=True, check=check)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Status Test")
    _git(root, "config", "user.email", "status@example.invalid")
    return root


def _write(root: Path, relative: str, content: str | bytes) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content, encoding="utf-8")
    return target


def _commit_initial(root: Path, content: str = "one\n") -> None:
    _write(root, "app.py", content)
    _git(root, "add", "app.py")
    _git(root, "commit", "-qm", "initial")


def _find(result: GitStatusResult, relative: str):
    return next(item for item in result.files if item.relative_path == relative)


def test_non_git_directory_returns_structured_result_without_initializing(tmp_path: Path) -> None:
    root = tmp_path / "not-repo"
    root.mkdir()

    result = git_status(root)

    assert result.is_git_repository is False
    assert result.is_clean is True
    assert result.head_state == "unknown"
    assert result.files == ()
    assert result.warnings
    assert not (root / ".git").exists()


def test_clean_repository_reports_branch_head_and_no_changes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit_initial(root)

    result = git_status(root)

    assert result.is_git_repository is True
    assert result.is_clean is True
    assert result.branch in {"main", "master"}
    assert result.head
    assert result.head_state == "branch"
    assert result.upstream is None
    assert result.ahead is None
    assert result.behind is None
    assert result.files == ()
    assert result.to_dict() == git_status(root).to_dict()


def test_unstaged_modified_and_untracked_names_with_spaces_tabs_unicode_and_quotes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit_initial(root)
    _write(root, "app.py", "two\n")
    for name in ("space name.txt", "tab\tname.txt", "ملف اختبار.txt", 'quote"name.txt'):
        _write(root, name, "untracked\n")

    result = git_status(root)

    app = _find(result, "app.py")
    assert app.status == "modified"
    assert app.index_status == " "
    assert app.worktree_status == "M"
    assert app in result.unstaged
    assert len(result.untracked) == 4
    assert result.is_clean is False
    assert [item.relative_path for item in result.files] == sorted(
        [item.relative_path for item in result.files], key=lambda value: (value.casefold(), value)
    )


def test_staged_and_unstaged_are_distinguished(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit_initial(root)
    _write(root, "app.py", "staged\n")
    _git(root, "add", "app.py")
    _write(root, "app.py", "unstaged\n")

    result = git_status(root)
    app = _find(result, "app.py")

    assert app.index_status == "M"
    assert app.worktree_status == "M"
    assert app in result.staged
    assert app in result.unstaged
    assert result.is_clean is False


def test_staged_new_file_and_deletions_are_classified(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit_initial(root)
    _write(root, "staged.py", "new\n")
    _git(root, "add", "staged.py")
    (root / "app.py").unlink()
    _write(root, "unstaged.py", "new\n")

    result = git_status(root)

    staged = _find(result, "staged.py")
    deleted = _find(result, "app.py")
    untracked = _find(result, "unstaged.py")
    assert staged.status == "added"
    assert staged.index_status == "A"
    assert deleted.status == "deleted"
    assert deleted.worktree_status == "D"
    assert untracked.is_untracked is True
    assert untracked.status == "untracked"
    assert deleted in result.deleted
    assert staged in result.added


def test_rename_is_structured_with_old_and_new_paths(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit_initial(root)
    (root / "app.py").rename(root / "renamed.py")
    _git(root, "add", "-A")

    result = git_status(root)
    renamed = _find(result, "renamed.py")

    assert renamed.status == "renamed"
    assert renamed.old_path == "app.py"
    assert renamed.new_path == "renamed.py"
    assert renamed in result.renamed


def test_ignored_files_are_excluded_by_default_and_explicitly_included_when_requested(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit_initial(root)
    _write(root, ".gitignore", "ignored.txt\n")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-qm", "ignore")
    ignored = _write(root, "ignored.txt", "secret-looking\n")

    default = git_status(root)
    included = git_status(root, include_ignored=True)

    assert not any(item.relative_path == "ignored.txt" for item in default.files)
    ignored_item = _find(included, "ignored.txt")
    assert ignored_item.is_ignored is True
    assert ignored_item.status == "ignored"
    assert included.is_clean is True
    assert ignored.read_text(encoding="utf-8") == "secret-looking\n"


def test_detached_head_and_unborn_head_are_explicit(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit_initial(root)
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    _git(root, "checkout", "--detach", "-q", head)
    detached = git_status(root)
    assert detached.branch is None
    assert detached.head == head
    assert detached.head_state == "detached"

    empty = _repo(tmp_path / "empty")
    unborn = git_status(empty)
    assert unborn.is_git_repository is True
    assert unborn.head is None
    assert unborn.head_state == "unborn"
    assert unborn.is_clean is True


def test_upstream_ahead_behind_uses_local_metadata_without_network(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit_initial(root)
    _git(root, "remote", "add", "origin", "https://invalid.example/no-network")
    branch = _git(root, "branch", "--show-current").stdout.strip()
    _git(root, "config", f"branch.{branch}.remote", "origin")
    _git(root, "config", f"branch.{branch}.merge", f"refs/heads/{branch}")

    result = git_status(root)

    assert result.upstream == f"origin/{branch}"
    assert result.ahead is None or result.ahead >= 0
    assert result.behind is None or result.behind >= 0
    assert result.is_git_repository is True


def test_conflict_status_is_structured_without_resolution(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit_initial(root, "base\n")
    branch = _git(root, "branch", "--show-current").stdout.strip()
    _git(root, "checkout", "-qb", "conflict-side")
    _write(root, "app.py", "side\n")
    _git(root, "add", "app.py")
    _git(root, "commit", "-qm", "side")
    _git(root, "checkout", "-q", branch)
    _write(root, "app.py", "main\n")
    _git(root, "add", "app.py")
    _git(root, "commit", "-qm", "main")
    merge = _git(root, "merge", "conflict-side", check=False)
    assert merge.returncode != 0

    result = git_status(root)

    assert result.is_clean is False
    assert result.conflicts
    assert result.conflicts[0].is_conflicted is True
    assert result.conflicts[0].status == "conflicted"
    assert (root / ".git" / "MERGE_HEAD").exists()


def test_limits_mark_truncation_and_omit_long_paths(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _commit_initial(root)
    for index in range(5):
        _write(root, f"file-{index}.txt", "x\n")
    _write(root, "a-very-long-name.txt", "x\n")

    result = git_status(root, max_files=2, max_path_length=8)

    assert result.truncated is True
    assert "max_files" in (result.truncation_reason or "") or result.warnings
    assert len(result.files) <= 2
    assert result.warnings


def test_invalid_arguments_tool_schema_and_no_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _repo(tmp_path)
    tool = GitStatusTool()
    assert tool.name == "git_status"
    assert tool.metadata.input_schema["required"] == ["project_root"]
    with pytest.raises(ToolError) as missing:
        tool.run({})
    assert missing.value.code is ToolErrorCode.INVALID_ARGUMENT
    with pytest.raises(ToolError) as non_mapping:
        tool.run([])
    assert non_mapping.value.code is ToolErrorCode.INVALID_ARGUMENT
    with pytest.raises(ToolError) as bad_limit:
        git_status(root, max_files=0)
    assert bad_limit.value.code is ToolErrorCode.INVALID_ARGUMENT
    git_status(root)
    assert capsys.readouterr().out == ""


def test_default_registry_and_agent_loop_do_not_gain_git_status(tmp_path: Path) -> None:
    assert "git_status" not in ToolRegistry.default().names()
    inspection = ToolRegistry.with_git_inspection()
    assert {"git_diff", "git_status"}.issubset(inspection.names())

    root = _repo(tmp_path)
    target = _write(root, "app.py", "unchanged\n")
    engine = type(
        "Engine",
        (),
        {
            "tokenizer": type("Tokenizer", (), {"encode": lambda self, value: list(value.encode())})(),
            "generate": lambda self, prompt: type("Output", (), {"generated_text": "FINAL: inspect"})(),
        },
    )()
    result = AgentLoop(engine).run("Inspect", root)
    assert result.final_answer == "inspect"
    assert target.read_text(encoding="utf-8") == "unchanged\n"
