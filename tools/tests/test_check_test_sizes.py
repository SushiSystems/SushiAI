# test_check_test_sizes.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/testing/check_test_sizes.py on test trees built on disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.check_test_sizes import (  # noqa: E402
    discover,
    rule_unsized_cpp_suite,
    rule_unsized_python_test,
    rule_unsized_typescript_suite,
)


class TestSizesTest(unittest.TestCase):
    """Builds a repository with a tests/ folder for each test."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def write(self, relative: str, text: str) -> None:
        """Writes a file below the repository root."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def found(self, rule) -> list[tuple[str, int]]:
        """Returns the file name and line of each issue a rule yields."""
        return [(issue.path.name, issue.line) for issue in rule(discover(self.root))]

    def test_reports_a_cpp_suite_without_a_size_and_skips_a_benchmark(self) -> None:
        self.write("tests/core/test_shape.cpp",
                   "TEST(Small_Shape, Rank) {}\nTEST_F(ShapeFixture, Bad) {}\nTEST_P(Medium_Io, Read) {}\n")
        self.write("tests/core/benchmark_shape.cpp", "TEST(ShapeBench, Run) {}\n")
        self.assertEqual(self.found(rule_unsized_cpp_suite), [("test_shape.cpp", 2)])

    def test_a_python_file_mark_sizes_every_test(self) -> None:
        self.write("tests/test_a.py", "import pytest\n\npytestmark = pytest.mark.small\n\ndef test_x():\n    pass\n")
        self.write("cli/tests/test_b.py", "import pytest\n\npytestmark = [pytest.mark.medium]\n\ndef test_y():\n    pass\n")
        self.assertEqual(self.found(rule_unsized_python_test), [])

    def test_reports_a_python_test_whose_file_and_function_carry_no_size(self) -> None:
        self.write("tests/test_c.py", "import pytest\n\n@pytest.mark.large\ndef test_x():\n    pass\n\n"
                                      "def test_y():\n    pass\n\nclass TestZ:\n    pass\n")
        self.assertEqual(self.found(rule_unsized_python_test), [("test_c.py", 7), ("test_c.py", 10)])

    def test_reports_a_typescript_describe_without_a_size(self) -> None:
        self.write("tests/web/camera.test.ts", "describe('small: camera', () => {});\ndescribe(\"orbit\", () => {});\n")
        self.assertEqual(self.found(rule_unsized_typescript_suite), [("camera.test.ts", 2)])


if __name__ == "__main__":
    unittest.main()
