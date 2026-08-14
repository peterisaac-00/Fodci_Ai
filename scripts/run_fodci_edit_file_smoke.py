"""Manual local smoke for the Phase 4.2 exact edit tool."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from backend_ai.tools import ToolError, ToolErrorCode, edit_file, read_file, write_file


def main() -> None:
    with TemporaryDirectory(prefix="fodci-phase42-smoke-") as directory:
        root = Path(directory) / "fixture"
        root.mkdir()
        unrelated = root / "unrelated.txt"
        unrelated.write_text("unchanged", encoding="utf-8")

        write_file(root, "app/main.py", "value = 'old'\n")
        assert read_file(root, "app/main.py").content == "value = 'old'\n"

        result = edit_file(root, "app/main.py", "'old'", "'new'")
        assert result.changed is True
        assert read_file(root, "app/main.py").content == "value = 'new'\n"

        before_ambiguous = read_file(root, "app/main.py").content
        edit_file(root, "app/main.py", "value", "value")
        assert read_file(root, "app/main.py").content == before_ambiguous

        try:
            edit_file(root, "app/main.py", "'old'", "'x'")
        except ToolError as error:
            assert error.code is ToolErrorCode.MATCH_NOT_FOUND
        else:
            raise AssertionError("Expected the old content to be absent")

        write_file(root, "app/ambiguous.py", "old\nold\n")
        before_ambiguous = read_file(root, "app/ambiguous.py").content
        try:
            edit_file(root, "app/ambiguous.py", "old", "new")
        except ToolError as error:
            assert error.code is ToolErrorCode.AMBIGUOUS_MATCH
        else:
            raise AssertionError("Expected ambiguous match rejection")
        assert read_file(root, "app/ambiguous.py").content == before_ambiguous

        try:
            edit_file(root, "../outside.py", "old", "new")
        except ToolError as error:
            assert error.code is ToolErrorCode.PATH_OUTSIDE_ROOT
        else:
            raise AssertionError("Expected traversal rejection")

        assert unrelated.read_text(encoding="utf-8") == "unchanged"
    print("Phase 4.2 edit_file smoke passed")


if __name__ == "__main__":
    main()
