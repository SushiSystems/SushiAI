# test_tests_trailer.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the decision of tools/commits/tests_trailer.py on messages and staged paths held in memory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commits import tests_trailer  # noqa: E402

is_test_path = tests_trailer.is_test_path
find_problems = tests_trailer.tests_problems


class TestsLineTest(unittest.TestCase):
    """Covers a fix or feature with and without a test, the reason line, and the types left alone."""
    K_TEST_SIZE = "small"

    def test_refuses_a_fix_that_stages_no_test(self) -> None:
        self.assertEqual(len(find_problems("fix(a): b\n\nBacklog: X-1\n", ["src/a.py"])), 1)

    def test_accepts_a_feature_that_stages_a_test(self) -> None:
        self.assertEqual(find_problems("feat!: b\n", ["src/a.py", "tools/tests/test_a.py"]), [])

    def test_accepts_a_reason_line(self) -> None:
        self.assertEqual(find_problems("fix: b\n\nTests: none (prose only)\n", ["skills/a/SKILL.md"]), [])

    def test_refuses_an_empty_reason(self) -> None:
        self.assertEqual(len(find_problems("fix: b\n\nTests: none\n", ["src/a.py"])), 1)

    def test_leaves_docs_and_chores_alone(self) -> None:
        self.assertEqual(find_problems("docs: b\n", ["docs/a.md"]), [])


class TestPathTest(unittest.TestCase):
    """Covers the paths read as tests."""
    K_TEST_SIZE = "small"

    def test_reads_test_folders_and_names(self) -> None:
        for path in ("tests/a.cpp", "Assets/Tests/A.cs", "src/a_test.go", "x/FooTests.cs", "test_a.py"):
            self.assertTrue(is_test_path(path), path)

    def test_leaves_other_paths(self) -> None:
        for path in ("src/a.py", "docs/testing.md", "skills/testing/SKILL.md"):
            self.assertFalse(is_test_path(path), path)


if __name__ == "__main__":
    unittest.main()
