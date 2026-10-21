# test_retire_archive.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests retiring the archive from version control."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.retire_archive import main, tracked_files  # noqa: E402

K_FILES = {
    "docs/archive/design/OLD.md": "# Old\n",
    "docs/archive/agent/2026_01_01_X/SPEC.md": "# X\n",
    "docs/guides/use.md": "# Use\n",
}


def _git(root: Path, *arguments: str) -> str:
    """Runs git in a repository and returns what it printed."""
    result = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *arguments],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return result.stdout


def _repository(folder: str, files: dict[str, str] | None = None) -> Path:
    """Writes the files, commits them in a new repository and returns its root."""
    root = Path(folder) / "alpha"
    for name, text in (K_FILES if files is None else files).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "files")
    return root


def _run(root: Path, *arguments: str) -> tuple[int, str]:
    """Runs the command and returns its exit code with what it printed."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = main([str(root), *arguments])
    return code, printed.getvalue()


class RetireTest(unittest.TestCase):
    """Covers each outcome of the command."""

    def test_a_dry_run_changes_nothing(self) -> None:
        """Reports the plan and leaves the index alone."""
        with tempfile.TemporaryDirectory() as folder:
            root = _repository(folder)
            code, printed = _run(root)
            self.assertEqual(code, 0)
            self.assertIn("2 tracked files", printed)
            self.assertEqual(len(tracked_files(root)), 2)
            self.assertFalse((root / ".gitignore").exists())

    def test_apply_tags_unstages_and_ignores_and_keeps_the_files(self) -> None:
        """Stages the removal and the ignore line, tags the history and leaves the files on disk."""
        with tempfile.TemporaryDirectory() as folder:
            root = _repository(folder)
            self.assertEqual(_run(root, "--apply")[0], 0)
            self.assertIn("docs-archive-", _git(root, "tag"))
            self.assertEqual(_git(root, "ls-files", "--", "docs/archive"), "")
            self.assertIn("docs/archive/", (root / ".gitignore").read_text(encoding="utf-8"))
            self.assertTrue((root / "docs/archive/design/OLD.md").is_file())
            staged = _git(root, "diff", "--cached", "--name-only").splitlines()
            self.assertIn(".gitignore", staged)
            self.assertIn("docs/archive/design/OLD.md", staged)

    def test_refuses_while_something_cites_the_archive(self) -> None:
        """Lists each citation, exits 1 and stages nothing."""
        files = {**K_FILES, "docs/guides/use.md": "See docs/archive/design/OLD.md.\n"}
        with tempfile.TemporaryDirectory() as folder:
            root = _repository(folder, files)
            code, printed = _run(root, "--apply")
            self.assertEqual(code, 1)
            self.assertIn("docs/guides/use.md:1: cites the archive", printed)
            self.assertEqual(len(tracked_files(root)), 2)
            self.assertEqual(_git(root, "tag"), "")

    def test_is_quiet_when_the_archive_is_already_gone(self) -> None:
        """Reports there is nothing to retire when no archive file is tracked."""
        with tempfile.TemporaryDirectory() as folder:
            root = _repository(folder, {"docs/guides/use.md": "# Use\n"})
            code, printed = _run(root, "--apply")
            self.assertEqual(code, 0)
            self.assertIn("nothing to retire", printed)


if __name__ == "__main__":
    unittest.main()
