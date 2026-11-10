# test_check_test_odr.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the rules of tools/testing/check_test_odr.py on test sources held in memory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.source_file import SourceFile  # noqa: E402
from testing.check_test_odr import (  # noqa: E402
    build_sources,
    rule_duplicate_type,
    rule_suite_fixture_mix,
    rule_unsafe_name,
)

K_ROW = "namespace Test {\nstruct Row {\n  int n;\n};\n}\n"
K_ANONYMOUS_ROW = "namespace Test {\nnamespace {\nstruct Row {\n  int n;\n};\n}\n}\n"


def found(rule, files: dict[str, str]) -> list[tuple[str, int]]:
    """Returns the file name and line of each issue a rule yields over files given as path and text."""
    sources = [SourceFile(Path(path), text) for path, text in files.items()]
    return [(issue.path.name, issue.line) for issue in rule(build_sources(sources))]


class TestDuplicateType(unittest.TestCase):
    """Runs the one-definition rule on files held in memory."""
    K_TEST_SIZE = "small"

    def test_reports_a_type_of_one_qualified_name_defined_in_two_files(self) -> None:
        self.assertEqual(found(rule_duplicate_type, {"a/test_a.cpp": K_ROW, "a/test_b.cpp": K_ROW}),
                         [("test_a.cpp", 2), ("test_b.cpp", 2)])

    def test_keeps_a_type_inside_an_anonymous_namespace(self) -> None:
        self.assertEqual(found(rule_duplicate_type, {"a/test_a.cpp": K_ANONYMOUS_ROW, "a/test_b.cpp": K_ANONYMOUS_ROW}), [])

    def test_keeps_a_type_declared_inside_a_function_body(self) -> None:
        body = "void f() {\n  struct Row {\n    int n;\n  };\n}\n"
        self.assertEqual(found(rule_duplicate_type, {"a/test_a.cpp": body, "a/test_b.cpp": body}), [])

    def test_keeps_two_types_of_one_name_in_two_namespaces(self) -> None:
        other = K_ROW.replace("Test", "Other")
        self.assertEqual(found(rule_duplicate_type, {"a/test_a.cpp": K_ROW, "a/test_b.cpp": other}), [])

    def test_keeps_a_forward_declaration_and_an_enum(self) -> None:
        text = "struct Row;\nenum class Mode {\n  A,\n};\n"
        self.assertEqual(found(rule_duplicate_type, {"a/test_a.cpp": text, "a/test_b.cpp": text}), [])

    def test_reports_a_fixture_class_with_a_base_list(self) -> None:
        text = "class Small_Spmv : public ::testing::TestWithParam<int> {\n};\n"
        self.assertEqual(found(rule_duplicate_type, {"a/test_a.cpp": text, "a/test_b.cpp": text}),
                         [("test_a.cpp", 1), ("test_b.cpp", 1)])

    def test_keeps_a_type_named_only_in_a_comment_or_a_string(self) -> None:
        text = '// struct Row {\nconst char* s = "struct Row {";\n'
        self.assertEqual(found(rule_duplicate_type, {"a/test_a.cpp": text, "a/test_b.cpp": text}), [])


class TestSuiteFixtureMix(unittest.TestCase):
    """Runs the suite-name rule on files held in memory."""
    K_TEST_SIZE = "small"

    def test_reports_each_plain_test_of_a_suite_that_a_fixture_test_also_uses(self) -> None:
        files = {"a/test_a.cpp": "TEST(Small_X, A) {\n}\n", "a/test_b.cpp": "TEST_F(Small_X, B) {\n}\n"}
        self.assertEqual(found(rule_suite_fixture_mix, files), [("test_a.cpp", 1)])

    def test_reports_a_suite_used_by_test_and_by_a_parameterised_test(self) -> None:
        text = "TEST_P(Small_Y, A) {\n}\nTEST(Small_Y, B) {\n}\n"
        self.assertEqual(found(rule_suite_fixture_mix, {"a/test_a.cpp": text}), [("test_a.cpp", 3)])

    def test_keeps_a_suite_whose_tests_all_use_one_macro(self) -> None:
        text = "TEST_F(Small_Z, A) {\n}\nTEST_F(Small_Z, B) {\n}\n"
        self.assertEqual(found(rule_suite_fixture_mix, {"a/test_a.cpp": text}), [])


class TestUnsafeName(unittest.TestCase):
    """Runs the test-name rule on files held in memory."""
    K_TEST_SIZE = "small"

    def test_reports_a_row_name_with_a_dash_or_a_space_when_a_generator_reads_the_name(self) -> None:
        text = ('INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i) { return i.param.name; });\n'
                'const Row k[] = {\n  {"a-b", 1},\n  {"a b", 2},\n  {"ab_1", 3},\n};\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}),
                         [("test_a.cpp", 3), ("test_a.cpp", 4)])

    def test_keeps_a_row_table_that_no_generator_names(self) -> None:
        text = 'const Row k[] = {\n  {"a-b", 1},\n};\n'
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [])

    def test_reports_to_string_of_a_signed_parameter_in_an_inline_generator(self) -> None:
        text = ('INSTANTIATE_TEST_SUITE_P(, S, V,\n  [](const ::testing::TestParamInfo<int>& i)\n'
                '  { return "N" + std::to_string(i.param); });\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [("test_a.cpp", 3)])

    def test_keeps_to_string_of_an_unsigned_parameter(self) -> None:
        text = ('INSTANTIATE_TEST_SUITE_P(, S, V,\n  [](const ::testing::TestParamInfo<std::size_t>& i)\n'
                '  { return "N" + std::to_string(i.param); });\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [])

    def test_reports_to_string_in_a_named_generator_function(self) -> None:
        text = ('std::string name(const ::testing::TestParamInfo<long>& i)\n{\n'
                '  return std::to_string(i.param);\n}\nINSTANTIATE_TEST_SUITE_P(, S, V, name);\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [("test_a.cpp", 3)])

    def test_reports_a_dash_literal_written_by_a_generator(self) -> None:
        text = 'INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i) { return "n-" + i.param.tag; });\n'
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [("test_a.cpp", 1)])

    def test_keeps_to_string_when_the_signature_of_a_named_generator_declares_an_unsigned_type(self) -> None:
        text = ('std::string name(const ::testing::TestParamInfo<std::uint32_t>& i)\n{\n'
                '  return std::to_string(i.param);\n}\nINSTANTIATE_TEST_SUITE_P(, S, V, name);\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [])

    def test_keeps_to_string_when_the_fixture_declares_an_unsigned_parameter(self) -> None:
        text = ('class S : public ::testing::TestWithParam<std::size_t>\n{\n};\n'
                'INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i) { return std::to_string(i.param); });\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [])

    def test_keeps_to_string_when_the_fixture_names_its_parameter_through_the_interface(self) -> None:
        text = ('class S : public Base,\n        public ::testing::WithParamInterface<unsigned int>\n{\n};\n'
                'INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i) { return std::to_string(i.param); });\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [])

    def test_reports_to_string_when_the_fixture_declares_a_signed_parameter(self) -> None:
        text = ('class S : public ::testing::TestWithParam<int>\n{\n};\n'
                'INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i) { return std::to_string(i.param); });\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [("test_a.cpp", 4)])

    def test_reports_to_string_when_the_fixture_is_not_in_the_file(self) -> None:
        text = 'INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i) { return std::to_string(i.param); });\n'
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [("test_a.cpp", 1)])

    def test_reports_a_shared_generator_when_one_of_its_fixtures_is_signed(self) -> None:
        text = ('class A : public ::testing::TestWithParam<std::size_t>\n{\n};\n'
                'class B : public ::testing::TestWithParam<int>\n{\n};\n'
                'std::string name(const auto& i)\n{\n  return std::to_string(i.param);\n}\n'
                'INSTANTIATE_TEST_SUITE_P(, A, V, name);\nINSTANTIATE_TEST_SUITE_P(, B, V, name);\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [("test_a.cpp", 9)])

    def test_keeps_to_string_of_an_element_of_an_unsigned_c_array(self) -> None:
        text = ('constexpr std::size_t K_INDICES[] = {0, 1};\n'
                'INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i) { return std::to_string(K_INDICES[i.param]); });\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [])

    def test_keeps_to_string_of_an_element_of_an_unsigned_std_array(self) -> None:
        text = ('constexpr std::array<unsigned, 2> K_SIZES = {1, 2};\n'
                'INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i) { return std::to_string(K_SIZES[i.param]); });\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [])

    def test_reports_to_string_of_an_element_of_a_signed_array(self) -> None:
        text = ('constexpr int K_VALUES[] = {-1, 1};\n'
                'INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i) { return std::to_string(K_VALUES[i.param]); });\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [("test_a.cpp", 2)])

    def test_reports_to_string_of_an_element_of_an_array_declared_elsewhere(self) -> None:
        text = 'INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i) { return std::to_string(K_VALUES[i.param]); });\n'
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [("test_a.cpp", 1)])

    def test_reports_a_computed_value_of_a_signed_parameter_beside_an_unsigned_array(self) -> None:
        text = ('constexpr std::size_t K_INDICES[] = {0, 1};\n'
                'class S : public ::testing::TestWithParam<int>\n{\n};\n'
                'INSTANTIATE_TEST_SUITE_P(, S, V, [](const auto& i)\n'
                '  { return std::to_string(static_cast<int>(std::lround(i.param * 100.0))); });\n')
        self.assertEqual(found(rule_unsafe_name, {"a/test_a.cpp": text}), [("test_a.cpp", 6)])


if __name__ == "__main__":
    unittest.main()
