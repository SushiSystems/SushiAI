# test_history.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests reading the commits of a range and the text of objects, each in one git call."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from commits.history import blob_texts, commits_since  # noqa: E402
from repo_builder import Repo  # noqa: E402


class HistoryTest(unittest.TestCase):
    """Builds a repository of three commits and reads it back."""

    def setUp(self) -> None:
        self.repo = Repo()
        self.addCleanup(self.repo.close)
        self.repo.write("base.md", "base"+chr(10))
        self.base = self.repo.commit("first")
        self.repo.write("a.md", "one\n")
        self.second = self.repo.commit("second" + chr(10) + chr(10) + "Backlog: none (test)")
        self.repo.write("a.md", "two\n")
        self.third = self.repo.commit("third")

    def test_the_commits_after_a_base_come_oldest_first_with_parent_and_whole_message(self) -> None:
        found = commits_since(self.repo.root, self.base.sha)
        self.assertEqual([c.sha for c in found], [self.second.sha, self.third.sha])
        self.assertEqual([c.parent for c in found], [self.base.sha, self.second.sha])
        self.assertIn("Backlog: none (test)", found[0].message)

    def test_a_blob_is_read_by_its_hash_and_an_unknown_hash_is_left_out(self) -> None:
        blob = self.repo.git("rev-parse", "HEAD:a.md").strip()
        self.assertEqual(blob_texts(self.repo.root, [blob, "0" * 40]), {blob: "two\n"})


if __name__ == "__main__":
    unittest.main()
