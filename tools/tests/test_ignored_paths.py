# test_ignored_paths.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests that a checker skips the paths git ignores."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.ignored_paths import drop_ignored, ignored_folders  # noqa: E402
from documentation.check_source_comments import collect  # noqa: E402


class IgnoredPathsTest(unittest.TestCase):
    """Covers the git ignore filter and the source comment checker that uses it."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        (self.root / "synced").mkdir()
        (self.root / "synced" / "foreign.py").write_text("x = 1\n", encoding="utf-8")
        (self.root / "owned.py").write_text("x = 1\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._folder.cleanup()

    def _make_repository(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / ".gitignore").write_text("synced/\n", encoding="utf-8")

    def test_an_ignored_folder_is_dropped(self) -> None:
        self._make_repository()
        paths = [self.root / "owned.py", self.root / "synced" / "foreign.py"]
        self.assertEqual(drop_ignored(self.root, paths), [self.root / "owned.py"])

    def test_every_path_is_kept_outside_a_repository(self) -> None:
        paths = [self.root / "owned.py", self.root / "synced" / "foreign.py"]
        self.assertEqual(drop_ignored(self.root, paths), paths)

    def test_an_ignored_folder_is_named_once(self) -> None:
        self._make_repository()
        self.assertEqual(ignored_folders(self.root), {(self.root / "synced").resolve()})

    def test_no_folder_is_ignored_outside_a_repository(self) -> None:
        self.assertEqual(ignored_folders(self.root), frozenset())

    def test_the_comment_checker_does_not_read_an_ignored_file(self) -> None:
        self._make_repository()
        names = [source.path.name for source in collect([self.root])]
        self.assertEqual(names, ["owned.py"])


if __name__ == "__main__":
    unittest.main()
