"""Manual local smoke for the Phase 4.3 delete tool."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from backend_ai.tools import ToolError, ToolErrorCode, delete_file, read_file, write_file


def main() -> None:
    with TemporaryDirectory(prefix="fodci-phase43-smoke-") as directory:
        root = Path(directory) / "fixture"
        root.mkdir()
        parent = root / "app"
        parent.mkdir()
        unrelated = root / "unrelated.txt"
        unrelated.write_text("unchanged", encoding="utf-8")

        write_file(root, "app/main.py", "print('delete me')\n")
        assert read_file(root, "app/main.py").content == "print('delete me')\n"
        result = delete_file(root, "app/main.py")
        assert result.deleted is True
        assert not (root / "app" / "main.py").exists()
        assert parent.is_dir()
        assert unrelated.read_text(encoding="utf-8") == "unchanged"

        write_file(root, "app/ambiguous.py", "old\nold\n")
        before = read_file(root, "app/ambiguous.py").content
        try:
            delete_file(root, "app/ambiguous.py")
        except ToolError:
            raise AssertionError("Deletion of a regular file should not depend on content")
        assert not (root / "app" / "ambiguous.py").exists()
        assert before == "old\nold\n"

        directory_target = root / "directory"
        directory_target.mkdir()
        try:
            delete_file(root, "directory")
        except ToolError as error:
            assert error.code is ToolErrorCode.NOT_A_FILE
        else:
            raise AssertionError("Expected directory rejection")

        try:
            delete_file(root, "../outside.txt")
        except ToolError as error:
            assert error.code is ToolErrorCode.PATH_OUTSIDE_ROOT
        else:
            raise AssertionError("Expected traversal rejection")

        outside = Path(directory) / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        link = root / "outside-link"
        link.symlink_to(outside)
        try:
            delete_file(root, "outside-link")
        except ToolError as error:
            assert error.code is ToolErrorCode.PATH_OUTSIDE_ROOT
        else:
            raise AssertionError("Expected symlink rejection")
        assert link.is_symlink()
        assert outside.read_text(encoding="utf-8") == "outside"

    print("Phase 4.3 delete_file smoke passed")


if __name__ == "__main__":
    main()
