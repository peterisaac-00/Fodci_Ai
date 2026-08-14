"""Manual local smoke for Phase 4.4 Safe Editing Infrastructure."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from backend_ai.tools import SafeEditPolicy, SafeEditSession, read_file, write_file


def main() -> None:
    with TemporaryDirectory(prefix="fodci-phase44-smoke-") as directory:
        root = Path(directory) / "project"
        root.mkdir()
        (root / "tests").mkdir()
        write_file(root, "app.py", "value = 'old'\n")
        write_file(root, "config.py", "DEBUG = True\n")
        write_file(root, "tests/test_app.py", "def test_app():\n    assert True\n")
        unrelated_before = read_file(root, "config.py").content

        session = SafeEditSession(
            SafeEditPolicy.for_modification(
                backup_enabled=True,
                retain_backup_on_success=True,
            )
        )
        before = session.snapshot(root, "app.py")
        assert before.exists is True
        assert before.content_hash is not None
        preview = session.diff_for_edit(root, "app.py", "'old'", "'new'")
        assert preview is not None
        assert preview.truncated is False

        edited = session.edit(root, "app.py", "'old'", "'new'")
        assert edited.success is True
        assert edited.verification_passed is True
        assert read_file(root, "app.py").content == "value = 'new'\n"
        assert edited.backup is not None
        assert edited.backup.retained is True
        assert (root / (edited.backup.relative_path or "")).read_text(encoding="utf-8") == "value = 'old'\n"

        deleted = session.delete(root, "tests/test_app.py")
        assert deleted.success is True
        assert deleted.verification_passed is True
        assert not (root / "tests/test_app.py").exists()
        assert (root / "tests").is_dir()
        assert read_file(root, "config.py").content == unrelated_before

    print("Phase 4.4 safe_editing smoke passed")


if __name__ == "__main__":
    main()
