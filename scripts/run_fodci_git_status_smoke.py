"""Manual read-only smoke for Phase 4.6 git_status."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from backend_ai.tools import GitStatusResult, git_status


def _git(root: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-current-repo", action="store_true")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="fodci-phase46-git-") as directory:
        root = Path(directory) / "project"
        root.mkdir()
        _git(root, "init", "-q")
        _git(root, "config", "user.name", "Fodci Status Smoke")
        _git(root, "config", "user.email", "status-smoke@example.invalid")
        (root / "app.py").write_text("version = 1\n", encoding="utf-8")
        _git(root, "add", "app.py")
        _git(root, "commit", "-qm", "initial")

        clean = git_status(root)
        assert isinstance(clean, GitStatusResult)
        assert clean.is_git_repository is True
        assert clean.is_clean is True
        assert clean.head_state == "branch"

        (root / "app.py").write_text("version = 2\n", encoding="utf-8")
        _git(root, "add", "app.py")
        (root / "app.py").write_text("version = 3\n", encoding="utf-8")
        (root / "notes.txt").write_text("untracked\n", encoding="utf-8")
        changed = git_status(root)
        app = next(item for item in changed.files if item.relative_path == "app.py")
        notes = next(item for item in changed.files if item.relative_path == "notes.txt")
        assert changed.is_clean is False
        assert app.index_status == "M" and app.worktree_status == "M"
        assert app in changed.staged and app in changed.unstaged
        assert notes.is_untracked is True
        assert notes in changed.untracked
        assert (root / ".git" / "index.lock").exists() is False

    if args.inspect_current_repo:
        current = git_status(Path.cwd())
        assert isinstance(current, GitStatusResult)
        print(
            "Current repository read-only status summary: "
            f"is_git_repository={current.is_git_repository}, "
            f"is_clean={current.is_clean}, files={len(current.files)}, truncated={current.truncated}"
        )

    print("Phase 4.6 git_status smoke passed")


if __name__ == "__main__":
    main()
