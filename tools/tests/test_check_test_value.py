# test_check_test_value.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/testing/check_test_value.py on test trees built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing import check_test_value  # noqa: E402
from testing.check_test_value import (  # noqa: E402
    build,
    collect,
    rule_literal_assertion,
    rule_no_assertion,
    rule_sleep_in_small,
)


class TestValueTest(unittest.TestCase):
    """Builds a repository with a tests/ folder for each test."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def write(self, relative: str, text: str) -> Path:
        """Writes a file below the repository root and returns its path."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def found(self, rule, changed=None) -> list[tuple[str, int]]:
        """Returns the file name and line of each issue a rule yields."""
        return [(issue.path.name, issue.line) for issue in rule(build(self.root, changed))]

    def test_reports_a_cpp_case_without_an_assertion_and_keeps_one_with_a_helper(self) -> None:
        self.write("tests/core/test_shape.cpp",
                   "TEST(Small_Shape, Prints) {\n  Print(f());\n}\n\nTEST(Small_Shape, Helper) {\n  expect_rank(t, 3);\n}\n")
        self.assertEqual(self.found(rule_no_assertion), [("test_shape.cpp", 1)])

    def test_a_benchmark_case_is_left_alone(self) -> None:
        self.write("tests/core/test_wave.cpp", "TEST(Benchmark_Wave, Run) {\n  Run();\n}\n")
        self.assertEqual(self.found(rule_no_assertion), [])

    def test_reports_python_and_typescript_cases_without_an_assertion(self) -> None:
        self.write("tests/test_a.py", "import pytest\n\npytestmark = pytest.mark.small\n\ndef test_x():\n    print(1)\n")
        self.write("tests/web/camera.test.ts", "describe('small: a', () => {\n  it('x', () => {\n    log(1);\n  });\n});\n")
        self.assertEqual(sorted(self.found(rule_no_assertion)), [("camera.test.ts", 2), ("test_a.py", 5)])

    def test_reports_an_assertion_on_a_literal(self) -> None:
        self.write("tests/core/test_shape.cpp", "TEST(Small_A, B) {\n  EXPECT_TRUE(true);\n  EXPECT_EQ(x, 1);\n}\n")
        self.assertEqual(self.found(rule_literal_assertion), [("test_shape.cpp", 2)])

    def test_reports_a_sleep_only_in_a_small_case(self) -> None:
        self.write("tests/core/test_wait.cpp",
                   "TEST(Small_A, B) {\n  sleep_for(1ms);\n  EXPECT_TRUE(f());\n}\n\n"
                   "TEST(Medium_A, C) {\n  sleep_for(1ms);\n  EXPECT_TRUE(f());\n}\n")
        self.assertEqual(self.found(rule_sleep_in_small), [("test_wait.cpp", 2)])

    def test_judges_only_a_case_with_a_changed_line(self) -> None:
        path = self.write("tests/core/test_shape.cpp",
                          "TEST(Small_Old, A) {\n  Print(1);\n}\n\nTEST(Small_New, B) {\n  Print(2);\n}\n")
        self.assertEqual(self.found(rule_no_assertion, {path: frozenset({6})}), [("test_shape.cpp", 5)])
        self.assertEqual(self.found(rule_no_assertion, {}), [])

    def test_checks_nothing_in_a_repository_that_sets_no_base_commit(self) -> None:
        self.write("tests/core/test_shape.cpp", "TEST(Small_A, B) {\n  Print(1);\n}\n")
        with mock.patch.object(check_test_value, "K_SINCE", ""):
            self.assertEqual(list(collect([self.root])), [])


if __name__ == "__main__":
    unittest.main()
