# check_test_value.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that every test case written after a commit asserts something, from `testing`.

Usage: python tools/testing/check_test_value.py [repository root] [--report] [--rule NAME]

K_TEST_VALUE_SINCE in tools/settings.py names the commit. A repository that sets none is not
checked. Only a case with a line the working tree adds or edits after that commit is judged.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from checking.checker import Checker, Issue, run  # noqa: E402
from testing.case_reader import K_BENCHMARK, Case, cases_of, is_test_file  # noqa: E402
from testing.changed_lines import changed_lines  # noqa: E402
from testing.check_test_sizes import discover  # noqa: E402

K_SINCE: str = getattr(settings, "K_TEST_VALUE_SINCE", "")


@dataclass(frozen=True, slots=True)
class ValueTree:
    """Holds the cases of one repository that the rules judge."""

    cases: tuple[Case, ...]


def _is_touched(case: Case, changed: dict[Path, frozenset[int]] | None) -> bool:
    """Returns whether a case has a line the working tree changed; every case when *changed* is None."""
    if changed is None:
        return True
    lines = changed.get(case.path, frozenset())
    return any(case.first <= line <= case.last for line in lines)


def build(root: Path, changed: dict[Path, frozenset[int]] | None) -> ValueTree:
    """Returns the touched, non-benchmark cases of the test files under a repository's test folders."""
    cases: list[Case] = []
    for path in discover(root).of(is_test_file):
        text = path.read_text(encoding="utf-8", errors="ignore")
        cases.extend(case for case in cases_of(path, text)
                     if case.size != K_BENCHMARK and _is_touched(case, changed))
    return ValueTree(tuple(cases))


def collect(paths: list[Path]) -> Iterable[ValueTree]:
    """Returns the value tree of each given folder, or none when the repository sets no base commit."""
    if not K_SINCE:
        return []
    return [build(path, changed_lines(path, K_SINCE)) for path in paths if path.is_dir()]


def rule_no_assertion(tree: ValueTree) -> Iterator[Issue]:
    """Yields an issue for each case with no assertion in its body."""
    for case in tree.cases:
        if case.assertions == 0:
            yield Issue(case.path, case.first, f"{case.name} asserts nothing; it only runs or prints")


def rule_literal_assertion(tree: ValueTree) -> Iterator[Issue]:
    """Yields an issue for each assertion that compares a literal with itself or holds a constant."""
    for case in tree.cases:
        for line in case.literal_lines:
            yield Issue(case.path, line, f"{case.name} asserts a literal, which cannot fail")


def rule_sleep_in_small(tree: ValueTree) -> Iterator[Issue]:
    """Yields an issue for each sleep in a `small` case."""
    for case in tree.cases:
        if case.size == "small":
            for line in case.sleep_lines:
                yield Issue(case.path, line, f"{case.name} is small and sleeps")


K_CHECKER = Checker(
    description="Checks that every test case written after K_TEST_VALUE_SINCE asserts something.",
    collect=collect,
    rules={
        "rule_no_assertion": rule_no_assertion,
        "rule_literal_assertion": rule_literal_assertion,
        "rule_sleep_in_small": rule_sleep_in_small,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))
