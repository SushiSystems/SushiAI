# check_test_additions.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks how many test cases and test programs each commit adds, from `testing`.

Usage: python tools/testing/check_test_additions.py [repository root] [--report] [--rule NAME]

K_TEST_ADDITIONS_SINCE in tools/settings.py names the commit after which commits are judged. A
repository that sets none is not checked. `added_tests.py` lists what a range adds.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from checking.checker import Checker, Issue, run  # noqa: E402
from commits.history import Commit, commits_since  # noqa: E402
from testing.added_tests import K_SINCE, added_cases, added_programs  # noqa: E402

K_BUDGET: dict[str, int] = getattr(settings, "K_TEST_BUDGET", {})
K_CASES_PER_COMMIT = 20
K_CASES_PER_FIX = 5
K_FIX = re.compile(r"^fix(?:\([^)]*\))?!?:")
K_APPROVAL = re.compile(r"^Test program:[ \t]*(\S.*?)[ \t]*$", re.MULTILINE)


def collect(paths: list[Path]) -> Iterable[Commit]:
    """Returns each commit after K_TEST_ADDITIONS_SINCE below each given repository root, oldest first."""
    return [commit for root in paths if root.is_dir() and K_SINCE for commit in commits_since(root, K_SINCE)]


def _issue(commit: Commit, text: str) -> Issue:
    """Returns an issue placed on the commit, named by its short hash."""
    return Issue(commit.root / f"commit-{commit.sha[:9]}", 1, text)


def _case_limit(commit: Commit) -> int:
    """Returns how many cases a commit may add: fewer for a `fix`, whose test is the smallest reproduction."""
    if K_FIX.match(commit.message):
        return K_BUDGET.get("cases_per_fix", K_CASES_PER_FIX)
    return K_BUDGET.get("cases_per_commit", K_CASES_PER_COMMIT)


def rule_too_many_cases(commit: Commit) -> Iterator[Issue]:
    """Yields an issue when a commit adds more test cases than its limit."""
    added = added_cases(commit)
    limit = _case_limit(commit)
    if len(added) > limit:
        yield _issue(commit, f"adds {len(added)} test cases, over its limit of {limit}; "
                             f"merge, parametrise or drop some, or split the work")


def rule_second_program_unapproved(commit: Commit) -> Iterator[Issue]:
    """Yields an issue for each build file that gains a second test program with no `Test program:` line."""
    if K_APPROVAL.search(commit.message):
        return
    for path in added_programs(commit):
        yield _issue(commit, f"{path.as_posix()} gains a second test program; the owner decides, "
                             f"so the message needs a `Test program: <owner, date>` line")


K_CHECKER = Checker(
    description="Check how many test cases and test programs each commit adds.",
    collect=collect,
    rules={
        "rule_too_many_cases": rule_too_many_cases,
        "rule_second_program_unapproved": rule_second_program_unapproved,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))
