# test_changed_lines.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests tools/testing/changed_lines.py: its diff reading on text in memory, its git reading once on a throwaway repository."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.changed_lines import changed_lines, changed_lines_of  # noqa: E402

K_ROOT = Path("repo")
K_EDIT_AND_ADD = (
    "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n"
    "@@ -2 +2 @@\n-two\n+TWO\n"
    "@@ -4,0 +5,2 @@\n+five\n+six\n"
)


class TestChangedLinesParse(unittest.TestCase):
    """Reads a diff text and the line counts of untracked files."""
    K_TEST_SIZE = "small"

    def test_names_the_edited_and_the_added_lines_only(self) -> None:
        self.assertEqual(changed_lines_of(K_ROOT, "base", K_EDIT_AND_ADD, {}),
                         {K_ROOT / "a.txt": frozenset({2, 5, 6})})

    def test_a_pure_deletion_adds_no_line(self) -> None:
        diff = "+++ b/a.txt\n@@ -2,1 +1,0 @@\n-two\n"
        self.assertEqual(changed_lines_of(K_ROOT, "base", diff, {}).get(K_ROOT / "a.txt", frozenset()),
                         frozenset())

    def test_an_untracked_file_counts_every_line(self) -> None:
        self.assertEqual(changed_lines_of(K_ROOT, "base", "", {K_ROOT / "c.txt": 2}),
                         {K_ROOT / "c.txt": frozenset({1, 2})})

    def test_a_deleted_file_adds_no_line(self) -> None:
        diff = "+++ /dev/null\n@@ -1,2 +0,0 @@\n-x\n-y\n"
        self.assertEqual(changed_lines_of(K_ROOT, "base", diff, {}), {})

    def test_an_unknown_base_stops_with_a_message(self) -> None:
        with self.assertRaises(SystemExit):
            changed_lines_of(K_ROOT, "0" * 40, None, {})


class TestChangedLinesShell(unittest.TestCase):
    """Builds a repository with one commit, then a second commit and an untracked file, and reads it with git."""
    K_TEST_SIZE = "medium"

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        self.git("init", "-q")
        self.write("a.txt", "one\ntwo\nthree\nfour\n")
        self.write("b.txt", "keep\n")
        self.git("add", ".")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
        self.base = self.git("rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        self._folder.cleanup()

    def git(self, *arguments: str) -> str:
        """Runs git in the repository and returns its output."""
        return subprocess.run(["git", "-C", str(self.root), *arguments], capture_output=True, text=True,
                              check=True).stdout

    def write(self, name: str, text: str) -> None:
        """Writes a file below the root."""
        (self.root / name).write_text(text, encoding="utf-8")

    def test_a_commit_after_the_base_and_an_untracked_file_are_both_named(self) -> None:
        self.write("b.txt", "keep\nmore\n")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "next")
        self.write("c.txt", "x\ny\n")
        found = changed_lines(self.root, self.base)
        self.assertEqual((found[self.root / "b.txt"], found[self.root / "c.txt"]),
                         (frozenset({2}), frozenset({1, 2})))


if __name__ == "__main__":
    unittest.main()
