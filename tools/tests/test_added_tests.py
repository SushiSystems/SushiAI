# test_added_tests.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests tools/testing/added_tests.py on throwaway git repositories."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from testing.added_tests import added_cases, added_programs, main  # noqa: E402
from repo_builder import Repo  # noqa: E402

K_ONE = "TEST(Small_A, One) {\n  EXPECT_TRUE(f());\n}\n"
K_TWO = "TEST(Small_A, Two) {\n  EXPECT_TRUE(f());\n}\n"
K_PROGRAM = "project_add_test_program({name} SOURCES test_a.cpp)\n"


class AddedTestsTest(unittest.TestCase):
    """Builds a repository with one base commit, then commits changes."""

    def setUp(self) -> None:
        self.repo = Repo()
        self.repo.write("tests/core/test_a.cpp", K_ONE)
        self.repo.write("README.md", "x\n")
        self.base = self.repo.commit("base")

    def tearDown(self) -> None:
        self.repo.close()

    def test_names_the_cases_a_commit_adds(self) -> None:
        self.repo.write("tests/core/test_a.cpp", K_ONE + "\n" + K_TWO)
        self.assertEqual(added_cases(self.repo.commit("add")), ["Small_A.Two"])

    def test_a_case_moved_between_files_adds_nothing(self) -> None:
        self.repo.write("tests/core/test_a.cpp", "")
        self.repo.write("tests/core/test_b.cpp", K_ONE)
        self.assertEqual(added_cases(self.repo.commit("move")), [])

    def test_a_renamed_suite_and_an_edited_body_add_nothing_but_a_new_name_with_a_new_body_does(self) -> None:
        renamed = K_ONE.replace("Small_A", "Small_Renamed")
        self.repo.write("tests/core/test_a.cpp", renamed)
        self.assertEqual(added_cases(self.repo.commit("rename")), [])
        edited = renamed.replace("f()", "g()")
        self.repo.write("tests/core/test_a.cpp", edited)
        self.assertEqual(added_cases(self.repo.commit("edit")), [])
        self.repo.write("tests/core/test_a.cpp", edited + "\n" + K_TWO.replace("f()", "h()"))
        self.assertEqual(added_cases(self.repo.commit("add")), ["Small_A.Two"])

    def test_a_copy_under_a_new_name_counts_because_the_original_stays(self) -> None:
        self.repo.write("tests/core/test_a.cpp", K_ONE + "\n" + K_ONE.replace("One", "OneAgain"))
        self.assertEqual(added_cases(self.repo.commit("copy")), ["Small_A.OneAgain"])

    def test_a_benchmark_and_a_file_outside_the_test_folders_add_nothing(self) -> None:
        self.repo.write("tests/core/test_b.cpp", "TEST(Benchmark_A, Run) {\n}\n")
        self.repo.write("tools/test_c.cpp", K_TWO)
        self.assertEqual(added_cases(self.repo.commit("other")), [])

    def test_a_removed_case_is_not_an_addition(self) -> None:
        self.repo.write("tests/core/test_a.cpp", "")
        self.assertEqual(added_cases(self.repo.commit("remove")), [])

    def test_the_first_program_of_a_folder_is_free_and_the_second_is_named(self) -> None:
        self.repo.write("tests/core/CMakeLists.txt", K_PROGRAM.format(name="core_one"))
        self.assertEqual(added_programs(self.repo.commit("first")), [])
        self.repo.write("tests/core/CMakeLists.txt", K_PROGRAM.format(name="core_one") + K_PROGRAM.format(name="core_two"))
        self.assertEqual(added_programs(self.repo.commit("second")), [Path("tests/core/CMakeLists.txt")])

    def test_two_programs_in_a_new_build_file_are_named(self) -> None:
        self.repo.write("tests/new/CMakeLists.txt", K_PROGRAM.format(name="a") + K_PROGRAM.format(name="b"))
        self.assertEqual(added_programs(self.repo.commit("two")), [Path("tests/new/CMakeLists.txt")])

    def test_the_list_prints_each_commit_that_adds_something(self) -> None:
        self.repo.write("tests/core/test_a.cpp", K_ONE + "\n" + K_TWO)
        self.repo.commit("feat: add a case")
        self.repo.write("README.md", "y\n")
        self.repo.commit("docs: change a page")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--root", str(self.repo.root), "--base", self.base.sha])
        lines = out.getvalue().splitlines()
        self.assertEqual((code, len(lines), lines[0].endswith("feat: add a case"), lines[1].strip()),
                         (0, 2, True, "case Small_A.Two"))


if __name__ == "__main__":
    unittest.main()
