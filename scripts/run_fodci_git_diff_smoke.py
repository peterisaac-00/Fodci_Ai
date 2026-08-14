"""Manual read-only smoke for Phase 4.5 git_diff."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from backend_ai.tools import GitDiffResult, git_diff


def _git(root: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-current-repo", action="store_true")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="fodci-phase45-git-") as directory:
        root = Path(directory) / "project"
        root.mkdir()
        _git(root, "init", "-q")
        _git(root, "config", "user.name", "Fodci Smoke")
        _git(root, "config", "user.email", "smoke@example.invalid")
        (root / "app.py").write_text("version = 1\n", encoding="utf-8")
        (root / "config.py").write_text("DEBUG = True\n", encoding="utf-8")
        _git(root, "add", ".")
        _git(root, "commit", "-qm", "initial")

        (root / "app.py").write_text("version = 2\n", encoding="utf-8")
        _git(root, "add", "app.py")
        (root / "app.py").write_text("version = 3\n", encoding="utf-8")
        (root / "notes.txt").write_text("untracked\n", encoding="utf-8")

        result = git_diff(root)
        assert isinstance(result, GitDiffResult)
        assert result.is_git_repository is True
        app = next(item for item in result.changed_files if item.relative_path == "app.py")
        notes = next(item for item in result.changed_files if item.relative_path == "notes.txt")
        assert app.staged is True and app.unstaged is True
        assert notes.untracked is True
        assert "version = 2" in result.staged_diff
        assert "version = 3" in result.unstaged_diff
        assert "notes.txt" not in result.unstaged_diff
        assert (root / ".git").is_dir()

    if args.inspect_current_repo:
        current = git_diff(Path.cwd())
        assert isinstance(current, GitDiffResult)
        print(
            "Current repository read-only summary: "
            f"is_git_repository={current.is_git_repository}, "
            f"files_changed={current.files_changed}, truncated={current.truncated}"
        )

    print("Phase 4.5 git_diff smoke passed")


if __name__ == "__main__":
    main()
