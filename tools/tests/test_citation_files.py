# test_citation_files.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests which files may cite a document, and which lines cite the archive, from values only."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.citation_files import archive_citation_lines, is_citable  # noqa: E402


class IsCitableTest(unittest.TestCase):
    """Covers the names a citation may come from."""

    K_TEST_SIZE = "small"

    def test_accepts_a_manual_page_a_source_file_and_a_top_level_tool_file(self) -> None:
        """Accepts a manual page, a source file and a top-level tool file of a known extension."""
        rows = ("docs/guides/use.md", "modules/core/source/x.cpp", "tools/settings.py", "CMakeLists.txt")
        for name in rows:
            with self.subTest(name=name):
                self.assertTrue(is_citable(name))

    def test_refuses_the_archive_and_the_shared_tools(self) -> None:
        """Refuses a file under docs/archive and a file under a shared tools folder, which name paths as fixtures."""
        rows = ("docs/archive/old.md", "tools/documentation/check_x.py", "tools/tests/test_x.py")
        for name in rows:
            with self.subTest(name=name):
                self.assertFalse(is_citable(name))

    def test_refuses_a_skipped_folder_at_any_depth(self) -> None:
        """Refuses a file under a vendored, build or git folder wherever it sits."""
        rows = ("third_party/lib/x.h", "modules/a/node_modules/b.md", "build/x.txt", "modules/a/.git/notes.md")
        for name in rows:
            with self.subTest(name=name):
                self.assertFalse(is_citable(name))

    def test_refuses_an_extension_no_citation_uses(self) -> None:
        """Refuses an image, and a file with no extension at all."""
        self.assertFalse(is_citable("docs/guides/pic.png"))
        self.assertFalse(is_citable("docs/guides/LICENSE"))

    def test_matches_the_archive_folder_by_name_not_by_prefix(self) -> None:
        """Accepts a folder whose name only begins like the archive's, such as docs/archive2."""
        self.assertTrue(is_citable("docs/archive2/x.md"))


class ArchiveCitationLinesTest(unittest.TestCase):
    """Covers the numbers of the lines that cite the archive."""

    K_TEST_SIZE = "small"

    def test_numbers_each_line_that_cites_the_archive_from_one(self) -> None:
        """Returns the line numbers, counted from 1, of each line with a path or a link into the archive."""
        text = "a\nSee docs/archive/x.md\nplain\n[x](../archive/y.md)\n"
        self.assertEqual(archive_citation_lines(text), [2, 4])

    def test_is_empty_when_no_line_cites_the_archive(self) -> None:
        """Finds nothing in text that names the archive only in passing, or not at all."""
        self.assertEqual(archive_citation_lines("docs/archives/x.md\nthe archive is gone\n"), [])
        self.assertEqual(archive_citation_lines(""), [])


if __name__ == "__main__":
    unittest.main()
