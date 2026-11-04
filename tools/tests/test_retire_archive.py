# test_retire_archive.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests retiring the archive from version control."""

from __future__ import annotations

import contextlib
import datetime
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.retire_archive import K_EXIT_BLOCKED, K_EXIT_CLEAN, main, plan, tag_name, tracked_files, with_ignore_line  # noqa: E402

K_FILES = {
    "docs/archive/design/OLD.md": "# Old\n",
    "docs/archive/agent/2026_01_01_X/SPEC.md": "# X\n",
    "docs/guides/use.md": "# Use\n",
}

K_DAY = datetime.date(2026, 11, 4)
K_CITED = [("docs/guides/use.md", 1)]


class TagNameTest(unittest.TestCase):
    """Covers the name of the tag that marks the history holding the archive."""

    K_TEST_SIZE = "small"

    def test_names_the_tag_by_the_day(self) -> None:
        """Puts the ISO date after the prefix."""
        self.assertEqual(tag_name(K_DAY), "docs-archive-2026-11-04")


class IgnoreLineTest(unittest.TestCase):
    """Covers the .gitignore text that ignores the archive."""

    K_TEST_SIZE = "small"

    def test_adds_the_line_to_an_empty_file(self) -> None:
        """Writes the line alone when no .gitignore exists yet."""
        self.assertEqual(with_ignore_line(""), "docs/archive/\n")

    def test_appends_the_line_after_the_existing_ones(self) -> None:
        """Keeps every line that is there and adds the archive last."""
        self.assertEqual(with_ignore_line("build/\n*.log\n"), "build/\n*.log\ndocs/archive/\n")

    def test_writes_nothing_when_the_archive_is_already_ignored(self) -> None:
        """Returns None when the line is already there, so the file is left as it is."""
        self.assertIsNone(with_ignore_line("build/\ndocs/archive/\n"))


class PlanTest(unittest.TestCase):
    """Covers each outcome of the command, as lines and an exit code."""

    K_TEST_SIZE = "small"

    def test_reports_nothing_to_retire_when_no_archive_file_is_tracked(self) -> None:
        """Returns the message, exit 0 and no action, even while citations remain."""
        result = plan(0, K_CITED, K_DAY, True)
        self.assertEqual(result.lines, ("the archive holds no tracked file; nothing to retire",))
        self.assertEqual((result.code, result.acts), (K_EXIT_CLEAN, False))

    def test_lists_each_citation_and_blocks(self) -> None:
        """Prints the citation with its line, exits 1 and acts on nothing."""
        result = plan(2, K_CITED, K_DAY, True)
        self.assertEqual(result.lines, ("docs/guides/use.md:1: cites the archive; repoint it to the topic that holds the decision",))
        self.assertEqual((result.code, result.acts), (K_EXIT_BLOCKED, False))

    def test_a_dry_run_names_the_plan_and_acts_on_nothing(self) -> None:
        """Gives the summary with the dry-run mark and exit 0."""
        result = plan(2, [], K_DAY, False)
        self.assertEqual(result.lines, ("2 tracked files; tag docs-archive-2026-11-04, unstage docs/archive, ignore it (dry run)",))
        self.assertEqual((result.code, result.acts, result.tag), (K_EXIT_CLEAN, False, "docs-archive-2026-11-04"))

    def test_apply_acts_and_says_what_was_staged(self) -> None:
        """Asks the shell to tag, unstage and ignore, and prints the staging hint after it."""
        result = plan(2, [], K_DAY, True)
        self.assertEqual(result.lines, ("2 tracked files; tag docs-archive-2026-11-04, unstage docs/archive, ignore it",))
        self.assertEqual(result.after, ("staged; review `git diff --cached --stat` and commit",))
        self.assertEqual((result.code, result.acts), (K_EXIT_CLEAN, True))


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
    """Covers each outcome of the command against a real repository."""

    K_TEST_SIZE = "medium"

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
