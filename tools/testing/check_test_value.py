# check_test_value.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Checks that every test case written after a commit is none of the forbidden kinds of `testing`.

Usage: python tools/testing/check_test_value.py [repository root] [--report] [--rule NAME]

K_TEST_VALUE_SINCE in tools/settings.py names the commit. A repository that sets none is not
checked. Only a case with a line the working tree adds or edits after that commit is judged.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from checking.checker import Checker, Issue, run  # noqa: E402
from checking.source_file import SourceFile, read_files  # noqa: E402
from checking.tests_tree import discover  # noqa: E402
from testing.case_reader import K_BENCHMARK, Case, cases_of, is_test_file  # noqa: E402
from testing.changed_lines import changed_lines  # noqa: E402

K_SINCE: str = getattr(settings, "K_TEST_VALUE_SINCE", "")
K_SMALL_FORBIDDEN_BASE: tuple[str, ...] = (
    r"\bstd::thread\b", r"\bstd::jthread\b", r"\bstd::async\s*\(",
    r"\bstd::(?:i|o)?fstream\b", r"\bfopen\s*\(", r"\bstd::filesystem::(?:create|remove|copy)\w*\s*\(",
    r"\b(?:steady|system|high_resolution)_clock::now\s*\(", r"\bgetenv\s*\(",
    r"\btmp_path\b", r"\btmpdir\b", r"\bsubprocess\.", r"\bthreading\.", r"\bos\.environ\b",
    r"\btime\.(?:time|monotonic|perf_counter)\s*\(",
)
K_SMALL_FORBIDDEN = re.compile("|".join(
    K_SMALL_FORBIDDEN_BASE + tuple(getattr(settings, "K_SMALL_FORBIDDEN_EXTRA", ()))))


@dataclass(frozen=True, slots=True)
class ValueTree:
    """Holds the cases of one repository that the rules judge, and the files they came from."""

    cases: tuple[Case, ...]
    sources: tuple[SourceFile, ...]


def _is_touched(case: Case, changed: dict[Path, frozenset[int]] | None) -> bool:
    """Returns whether a case has a line the working tree changed; every case when *changed* is None."""
    if changed is None:
        return True
    lines = changed.get(case.path, frozenset())
    return any(case.first <= line <= case.last for line in lines)


def build_tree(sources: Sequence[SourceFile], changed: dict[Path, frozenset[int]] | None) -> ValueTree:
    """Returns the touched, non-benchmark cases of the given test files, with the files they came from."""
    cases = [case for source in sources for case in cases_of(source.path, source.text)
             if case.size != K_BENCHMARK and _is_touched(case, changed)]
    return ValueTree(tuple(cases), tuple(sources))


def rule_no_assertion(tree: ValueTree) -> Iterator[Issue]:
    """Yields an issue for each case with no assertion in its body."""
    for case in tree.cases:
        if case.assertions == 0:
            yield Issue(case.path, case.first, f"{case.name} asserts nothing; it only runs or prints")


def rule_weak_assertion_only(tree: ValueTree) -> Iterator[Issue]:
    """Yields an issue for each case whose only checks are `SUCCEED`, `static_assert` or a no-throw check."""
    for case in tree.cases:
        if case.weak_only:
            yield Issue(case.path, case.first,
                        f"{case.name} only succeeds, checks at compile time or checks that nothing throws; "
                        f"assert the promise")


def rule_literal_assertion(tree: ValueTree) -> Iterator[Issue]:
    """Yields an issue for each assertion that compares a literal with itself or holds a constant."""
    for case in tree.cases:
        for line in case.literal_lines:
            yield Issue(case.path, line, f"{case.name} asserts a literal, which cannot fail")


def rule_sleep(tree: ValueTree) -> Iterator[Issue]:
    """Yields an issue for each sleep in a case of any size."""
    for case in tree.cases:
        for line in case.sleep_lines:
            yield Issue(case.path, line, f"{case.name} sleeps; take a clock and advance it, or wait on an event")


def rule_tolerance_literal(tree: ValueTree) -> Iterator[Issue]:
    """Yields an issue for each floating-point comparison whose tolerance is a typed number."""
    for case in tree.cases:
        for line in case.tolerance_lines:
            yield Issue(case.path, line, f"{case.name} types a tolerance; derive it with the tolerance helper")


def rule_small_reaches_outside(tree: ValueTree) -> Iterator[Issue]:
    """Yields an issue for each line of a `small` case that reaches a thread, a file, the clock or the environment."""
    texts = {source.path: source.text.splitlines() for source in tree.sources}
    for case in tree.cases:
        if case.size != "small":
            continue
        lines = texts.get(case.path, [])
        for number in range(case.first, min(case.last, len(lines)) + 1):
            match = K_SMALL_FORBIDDEN.search(lines[number - 1])
            if match:
                yield Issue(case.path, number,
                            f"{case.name} is small and reaches `{match.group()}`; make it medium or pass the seam in")


def collect(paths: list[Path]) -> Iterable[ValueTree]:
    """Returns the value tree of each given folder, or none when the repository sets no base commit."""
    if not K_SINCE:
        return []
    return [build_tree(read_files(discover(path).files, is_test_file), changed_lines(path, K_SINCE))
            for path in paths if path.is_dir()]


K_CHECKER = Checker(
    description="Checks that every test case written after K_TEST_VALUE_SINCE is none of the forbidden kinds.",
    collect=collect,
    rules={
        "rule_no_assertion": rule_no_assertion,
        "rule_weak_assertion_only": rule_weak_assertion_only,
        "rule_literal_assertion": rule_literal_assertion,
        "rule_sleep": rule_sleep,
        "rule_tolerance_literal": rule_tolerance_literal,
        "rule_small_reaches_outside": rule_small_reaches_outside,
    },
)

if __name__ == "__main__":
    sys.exit(run(K_CHECKER))
