# test_case_reader.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests tools/testing/case_reader.py on source text for each language it reads."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.case_reader import cases_of  # noqa: E402

K_CPP = Path("tests/core/test_shape.cpp")
K_PY = Path("tests/test_shape.py")
K_TS = Path("tests/web/camera.test.ts")


def one(path: Path, text: str):
    """Returns the single case a text declares."""
    found = cases_of(path, text)
    assert len(found) == 1, found
    return found[0]


class CppCasesTest(unittest.TestCase):
    """Reads GoogleTest suites."""

    def test_reads_name_size_span_and_assertions(self) -> None:
        case = one(K_CPP, "TEST(Small_Shape, Rank) {\n  int a = 1;\n  EXPECT_EQ(a, 1);\n  ASSERT_TRUE(a > 0);\n}\n")
        self.assertEqual((case.name, case.size, case.first, case.last, case.assertions),
                         ("Small_Shape.Rank", "small", 1, 5, 2))

    def test_a_commented_or_quoted_assertion_does_not_count(self) -> None:
        case = one(K_CPP, 'TEST(Small_A, B) {\n  // EXPECT_TRUE(x);\n  /* ASSERT_EQ(1, 2); */\n'
                          '  Log("EXPECT_TRUE(y)");\n}\n')
        self.assertEqual(case.assertions, 0)

    def test_braces_in_strings_and_raw_strings_do_not_end_the_body(self) -> None:
        case = one(K_CPP, 'TEST(Small_A, B) {\n  auto s = "}";\n  auto r = R"x(})x";\n  EXPECT_TRUE(f(s, r));\n}\n')
        self.assertEqual((case.assertions, case.last), (1, 5))

    def test_a_helper_named_expect_or_assert_counts(self) -> None:
        self.assertEqual(one(K_CPP, "TEST(Small_A, B) {\n  expect_shape(t, 3);\n}\n").assertions, 1)
        self.assertEqual(one(K_CPP, "TEST(Small_A, B) {\n  expect_match<double, float>(t);\n}\n").assertions, 1)
        self.assertEqual(one(K_CPP, "TEST(Small_A, B) {\n  check_shape(t, 3);\n}\n").assertions, 0)

    def test_finds_literal_assertions_and_sleeps_by_line(self) -> None:
        case = one(K_CPP, "TEST(Small_A, B) {\n  EXPECT_TRUE(true);\n  EXPECT_EQ(3, 3);\n  EXPECT_EQ(x, 3);\n"
                          "  std::this_thread::sleep_for(10ms);\n}\n")
        self.assertEqual((case.literal_lines, case.sleep_lines), ((2, 3), (5,)))

    def test_knows_a_benchmark_and_an_unsized_suite(self) -> None:
        self.assertEqual(one(K_CPP, "TEST(Benchmark_Wave, Run) {\n}\n").size, "benchmark")
        self.assertIsNone(one(K_CPP, "TEST_F(ShapeFixture, Run) {\n}\n").size)


class PythonCasesTest(unittest.TestCase):
    """Reads pytest and unittest functions."""

    def test_reads_asserts_raises_and_helpers(self) -> None:
        case = one(K_PY, "def test_a():\n    assert f() == 1\n    with pytest.raises(ValueError):\n        f()\n"
                         "    expect_shape(t)\n    self.assertEqual(1, 1)\n")
        self.assertEqual(case.assertions, 4)

    def test_a_test_that_only_prints_asserts_nothing(self) -> None:
        self.assertEqual(one(K_PY, "def test_a():\n    print(f())\n").assertions, 0)

    def test_finds_a_literal_assert_and_a_sleep(self) -> None:
        case = one(K_PY, "def test_a():\n    assert True\n    time.sleep(1)\n")
        self.assertEqual((case.literal_lines, case.sleep_lines), ((2,), (3,)))

    def test_size_comes_from_the_function_then_the_class_then_the_file(self) -> None:
        text = ("import pytest\n\npytestmark = pytest.mark.medium\n\n@pytest.mark.small\n"
                "def test_a():\n    assert 1\n\n@pytest.mark.large\nclass TestB:\n    def test_c(self):\n        assert 1\n\n"
                "def test_d():\n    assert 1\n")
        self.assertEqual([(case.name, case.size) for case in cases_of(K_PY, text)],
                         [("test_a", "small"), ("test_c", "large"), ("test_d", "medium")])


class TypeScriptCasesTest(unittest.TestCase):
    """Reads describe, it and test."""

    def test_reads_the_size_of_the_describe_above_and_its_assertions(self) -> None:
        text = ("describe('small: camera', () => {\n  it('orbits', () => {\n    expect(a).toBe(1);\n"
                "    expect(b).toBe(2);\n  });\n});\n")
        case = one(K_TS, text)
        self.assertEqual((case.name, case.size, case.assertions, case.first, case.last), ("orbits", "small", 2, 2, 5))

    def test_a_template_string_with_braces_does_not_end_the_body(self) -> None:
        case = one(K_TS, "describe('medium: a', () => {\n  test('x', async () => {\n    const s = `}${y}`;\n"
                         "    expect(s).toBe('}');\n  });\n});\n")
        self.assertEqual(case.assertions, 1)

    def test_finds_a_literal_expectation_and_a_timer(self) -> None:
        case = one(K_TS, "describe('small: a', () => {\n  it('x', () => {\n    expect(true).toBe(true);\n"
                         "    setTimeout(f, 10);\n  });\n});\n")
        self.assertEqual((case.literal_lines, case.sleep_lines), ((3,), (4,)))


class OtherFilesTest(unittest.TestCase):
    """Leaves files of other kinds alone."""

    def test_a_markdown_file_has_no_cases(self) -> None:
        self.assertEqual(cases_of(Path("README.md"), "TEST(Small_A, B) {}"), [])


if __name__ == "__main__":
    unittest.main()
