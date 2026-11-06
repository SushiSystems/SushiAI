# test_check_test_sizes.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests tools/testing/check_test_sizes.py: its rules in memory, its command line once on disk."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import run  # noqa: E402
from checking.source_file import SourceFile  # noqa: E402
from testing.check_test_sizes import (  # noqa: E402
    K_CHECKER,
    rule_unsized_cpp_suite,
    rule_unsized_csharp_fixture,
    rule_unsized_python_test,
    rule_unsized_typescript_suite,
)

K_CSHARP_FIXTURES = (
    "[TestFixture]\n[Category(\"small\")]\n"
    "public class ShapeTests {\n  [Test] public void A() {}\n}\n"
    "[TestFixture]\npublic class LoadTests {\n  [Test] public void B() {}\n}\n"
    "[Category(\"slow\")]\npublic class ScanTests {\n  [UnityTest] public void C() {}\n}\n"
    "public abstract class BaseTests {\n  [Test] public void D() {}\n}\n"
    "public class Helper {\n  public void E() {}\n}\n"
)


def found(rule, files: dict[str, str]) -> list[tuple[str, int]]:
    """Returns the path and line of each issue a rule yields over files given as path and text."""
    sources = [SourceFile(Path(path), text) for path, text in files.items()]
    return [(issue.path.as_posix(), issue.line) for issue in rule(sources)]


class TestSizesRuleTest(unittest.TestCase):
    """Runs each rule on files held in memory."""
    K_TEST_SIZE = "small"

    def test_cpp_rule_reports_an_unsized_suite_in_a_test_file_only(self) -> None:
        files = {
            "tests/test_shape.cpp":
                "TEST(Small_Shape, Rank) {}\nTEST_F(ShapeFixture, Bad) {}\n"
                "TEST_P(Medium_Io, R) {}\n",
            "tests/benchmark_shape.cpp": "TEST(ShapeBench, Run) {}\n",
        }
        self.assertEqual(found(rule_unsized_cpp_suite, files), [("tests/test_shape.cpp", 2)])

    def test_python_rule_accepts_every_test_when_the_file_mark_sizes_it(self) -> None:
        files = {
            "tests/test_a.py": "import pytest\n\npytestmark = pytest.mark.small\n\n"
                               "def test_x():\n    pass\n",
            "tests/test_b.py": "import pytest\n\npytestmark = [pytest.mark.medium]\n\n"
                               "def test_y():\n    pass\n",
        }
        self.assertEqual(found(rule_unsized_python_test, files), [])

    def test_python_rule_accepts_every_test_when_the_size_constant_sizes_it(self) -> None:
        files = {"tests/test_a.py": "K_TEST_SIZE = \"small\"\n\ndef test_x():\n    pass\n"}
        self.assertEqual(found(rule_unsized_python_test, files), [])

    def test_python_rule_reports_an_unsized_testcase_subclass_whatever_its_name(self) -> None:
        files = {"tests/test_a.py": "import unittest\n\nclass RuleTableTest(unittest.TestCase):\n    pass\n"}
        self.assertEqual(found(rule_unsized_python_test, files), [("tests/test_a.py", 3)])

    def test_python_rule_accepts_a_class_size_attribute(self) -> None:
        files = {"tests/test_a.py": "import unittest\n\nclass AlphaTest(unittest.TestCase):\n"
                                    "    K_TEST_SIZE = \"small\"\n\n"
                                    "class BetaTest(unittest.TestCase):\n    K_TEST_SIZE: str = \"large\"\n"}
        self.assertEqual(found(rule_unsized_python_test, files), [])

    def test_python_rule_reports_a_test_when_file_and_test_carry_no_size(self) -> None:
        files = {
            "tests/test_c.py": "import pytest\n\n@pytest.mark.large\ndef test_x():\n    pass\n\n"
                               "def test_y():\n    pass\n\nclass TestZ:\n    pass\n",
            "tests/helper.py": "def test_w():\n    pass\n",
        }
        self.assertEqual(
            found(rule_unsized_python_test, files),
            [("tests/test_c.py", 7), ("tests/test_c.py", 10)],
        )

    def test_typescript_rule_reports_an_unsized_describe_in_a_test_file_only(self) -> None:
        files = {
            "tests/camera.test.ts":
                "describe('small: camera', () => {});\ndescribe(\"orbit\", () => {});\n",
            "tests/camera.ts": "describe('orbit', () => {});\n",
        }
        self.assertEqual(found(rule_unsized_typescript_suite, files), [("tests/camera.test.ts", 2)])

    def test_csharp_rule_reports_a_fixture_without_a_size_category(self) -> None:
        files = {"tests/ShapeTests.cs": K_CSHARP_FIXTURES}
        self.assertEqual(
            found(rule_unsized_csharp_fixture, files),
            [("tests/ShapeTests.cs", 6), ("tests/ShapeTests.cs", 10)],
        )

    def test_a_characterization_suite_names_its_size_after_the_word(self) -> None:
        files = {"tests/test_a.cpp": "TEST(Characterization_Medium_Exit, Pin) {}\nTEST(Characterization_Exit, Pin) {}\n"}
        self.assertEqual(found(rule_unsized_cpp_suite, files), [("tests/test_a.cpp", 2)])

    def test_every_rule_skips_a_benchmark_suite(self) -> None:
        files = {
            "tests/test_a.cpp": "TEST(Benchmark_Wave, Run) {}\n",
            "tests/a.test.ts": "describe('Benchmark_wave', () => {});\n",
            "tests/test_b.py": "class TestBenchmark_Wave:\n    pass\n",
            "tests/B.cs":
                "[TestFixture]\npublic class Benchmark_Wave {\n  [Test] public void A() {}\n}\n",
        }
        rules = (
            rule_unsized_cpp_suite,
            rule_unsized_typescript_suite,
            rule_unsized_python_test,
            rule_unsized_csharp_fixture,
        )
        self.assertEqual([finding for rule in rules for finding in found(rule, files)], [])


class TestSizesShellTest(unittest.TestCase):
    """Runs the checker's command line once on a tree written to a temporary folder."""
    K_TEST_SIZE = "medium"

    def test_command_reads_every_test_folder_at_any_depth_and_nothing_outside(self) -> None:
        files = {
            "tools/tests/test_a.py": "def test_x():\n    pass\n",
            "app/gui/tests/test_b.py": "def test_y():\n    pass\n",
            "web/src/__tests__/camera.test.ts": "describe('orbit', () => {});\n",
            "src/test_c.py": "def test_z():\n    pass\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative, text in files.items():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_text(text, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = run(K_CHECKER, [str(root)])
        self.assertEqual((code, output.getvalue().splitlines()), (1, [
            "app/gui/tests/test_b.py:1: rule_unsized_python_test: "
            "test_y carries no size, and the file sets none",
            "tools/tests/test_a.py:1: rule_unsized_python_test: "
            "test_x carries no size, and the file sets none",
            "web/src/__tests__/camera.test.ts:1: rule_unsized_typescript_suite: "
            "describe 'orbit' opens with none of small:, medium:, large:",
        ]))


if __name__ == "__main__":
    unittest.main()
