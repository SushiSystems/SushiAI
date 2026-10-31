# test_check_test_additions.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of tools/testing/check_test_additions.py on throwaway git repositories."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from testing import check_test_additions  # noqa: E402
from testing.check_test_additions import (  # noqa: E402
    collect,
    rule_second_program_unapproved,
    rule_too_many_cases,
)
from repo_builder import Repo  # noqa: E402


def cases(count: int, first: int = 0) -> str:
    """Returns the source of *count* small C++ cases."""
    return "".join(f"TEST(Small_A, Case{index}) {{\n  EXPECT_TRUE(f());\n}}\n\n" for index in range(first, first + count))


class CheckTestAdditionsTest(unittest.TestCase):
    """Builds a repository with one base commit that holds a test file."""

    def setUp(self) -> None:
        self.repo = Repo()
        self.repo.write("tests/core/test_a.cpp", cases(1))
        self.base = self.repo.commit("chore: base")
        limits = mock.patch.object(check_test_additions, "K_BUDGET", {"cases_per_commit": 3, "cases_per_fix": 1})
        limits.start()
        self.addCleanup(limits.stop)

    def tearDown(self) -> None:
        self.repo.close()

    def issues(self, rule, commit) -> list[str]:
        """Returns the messages a rule yields for a commit."""
        return [issue.message for issue in rule(commit)]

    def test_a_commit_at_its_limit_passes_and_one_over_it_is_named(self) -> None:
        self.repo.write("tests/core/test_a.cpp", cases(4))
        self.assertEqual(self.issues(rule_too_many_cases, self.repo.commit("feat(core): at the limit")), [])
        self.repo.write("tests/core/test_a.cpp", cases(8))
        found = self.issues(rule_too_many_cases, self.repo.commit("feat(core): over the limit"))
        self.assertEqual(len(found), 1)
        self.assertIn("adds 4 test cases, over its limit of 3", found[0])

    def test_a_fix_has_the_smaller_limit(self) -> None:
        self.repo.write("tests/core/test_a.cpp", cases(2))
        self.assertEqual(self.issues(rule_too_many_cases, self.repo.commit("fix(core): one regression test")), [])
        self.repo.write("tests/core/test_a.cpp", cases(4))
        found = self.issues(rule_too_many_cases, self.repo.commit("fix(core): two regression tests"))
        self.assertIn("over its limit of 1", found[0])

    def test_a_second_program_needs_the_owners_line(self) -> None:
        self.repo.write("tests/core/CMakeLists.txt", "add_executable(one a.cpp)\n")
        self.repo.commit("build: first program")
        self.repo.write("tests/core/CMakeLists.txt", "add_executable(one a.cpp)\nadd_executable(two b.cpp)\n")
        found = self.issues(rule_second_program_unapproved, self.repo.commit("build(core): a second program"))
        self.assertEqual(len(found), 1)
        self.assertIn("Test program:", found[0])

    def test_the_owners_line_approves_a_second_program(self) -> None:
        self.repo.write("tests/core/CMakeLists.txt", "add_executable(one a.cpp)\nadd_executable(two b.cpp)\n")
        message = "build(core): two programs\n\nTest program: owner 2026-10-30\n"
        self.assertEqual(self.issues(rule_second_program_unapproved, self.repo.commit(message)), [])

    def test_a_repository_that_sets_no_base_commit_is_not_checked(self) -> None:
        self.repo.write("tests/core/test_a.cpp", cases(9))
        self.repo.commit("feat(core): many")
        with mock.patch.object(check_test_additions, "K_SINCE", ""):
            self.assertEqual(list(collect([self.repo.root])), [])


if __name__ == "__main__":
    unittest.main()
