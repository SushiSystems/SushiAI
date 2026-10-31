# test_changed_lines.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests tools/testing/changed_lines.py on throwaway git repositories."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.changed_lines import changed_lines  # noqa: E402


class ChangedLinesTest(unittest.TestCase):
    """Builds a repository with one commit, then changes the working tree."""

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

    def test_names_the_edited_and_the_added_lines_only(self) -> None:
        self.write("a.txt", "one\nTWO\nthree\nfour\nfive\nsix\n")
        found = changed_lines(self.root, self.base)
        self.assertEqual(found, {self.root / "a.txt": frozenset({2, 5, 6})})

    def test_a_pure_deletion_adds_no_line(self) -> None:
        self.write("a.txt", "one\nthree\nfour\n")
        self.assertEqual(changed_lines(self.root, self.base).get(self.root / "a.txt", frozenset()), frozenset())

    def test_an_untracked_file_counts_every_line(self) -> None:
        self.write("c.txt", "x\ny\n")
        self.assertEqual(changed_lines(self.root, self.base)[self.root / "c.txt"], frozenset({1, 2}))

    def test_a_commit_after_the_base_is_included(self) -> None:
        self.write("b.txt", "keep\nmore\n")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "next")
        self.assertEqual(changed_lines(self.root, self.base)[self.root / "b.txt"], frozenset({2}))

    def test_an_unknown_base_stops_with_a_message(self) -> None:
        with self.assertRaises(SystemExit):
            changed_lines(self.root, "0" * 40)


if __name__ == "__main__":
    unittest.main()
