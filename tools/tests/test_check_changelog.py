# test_check_changelog.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests each rule of the changelog checker on values, and its shell on changelogs written to disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.check_changelog import K_CHECKER, Entry, collect, parse_entries  # noqa: E402

K_GOOD = "- 2026-09-15 — matter: Reserved the neighbour list at tick start (`neighbour_list.hpp`)."
K_PATH = Path("docs/reference/CHANGELOG.md")
# Each case: its name, the body lines under the title, and the rules that must fire.
K_CASES = (
    ("a_well_formed_changelog_is_clean", ["## Unreleased", "", K_GOOD, "", "## v0.3.0 — 2026-09-01"], []),
    ("a_heading_of_the_wrong_shape", ["## Version 0.3"], ["rule_entry_shape"]),
    ("an_entry_of_the_wrong_shape", ["## Unreleased", "- matter: reserved the list."], ["rule_entry_shape"]),
    ("a_second_live_release", ["## Unreleased", "## v0.3.0 — 2026-09-01", "## v0.2.0 — 2026-08-01"],
     ["rule_live_releases"]),
    ("an_entry_above_the_length_ceiling",
     ["## Unreleased", K_GOOD.replace("tick start", "tick start " + "and again " * 20)], ["rule_length"]),
    ("an_entry_of_two_sentences",
     ["## Unreleased", K_GOOD.replace(" at tick start", ". It saves time")], ["rule_single_sentence"]),
    ("a_nested_bullet", ["## Unreleased", K_GOOD, "  - a detail."], ["rule_no_nesting"]),
    ("an_entry_citing_six_places",
     ["## Unreleased", K_GOOD.replace("`neighbour_list.hpp`", ", ".join(f"`f{index}.hpp`" for index in range(6)))],
     ["rule_cited_paths"]),
)


def _text(lines: list[str]) -> str:
    """Returns a changelog holding the given lines under its title."""
    return "# Changelog\n\n" + "\n".join(lines) + "\n"


def _rules(entries: list[Entry]) -> list[str]:
    """Returns the name of every rule that fires on the entries, once per issue."""
    return [name for entry in entries for name, rule in K_CHECKER.rules.items() for _ in rule(entry)]


def _case_test(name: str, lines: list[str], expected: list[str]) -> Callable[[unittest.TestCase], None]:
    """Builds the test of one case of the table: the rules it names fire on its text, and no other."""

    def test(self: unittest.TestCase) -> None:
        self.assertEqual(_rules(parse_entries(K_PATH, _text(lines).splitlines())), expected)

    test.__name__ = f"test_{name}"
    return test


class RuleTest(unittest.TestCase):
    """Runs every rule on the entries parsed from changelog text."""

    K_TEST_SIZE = "small"

    def test_a_release_heading_with_trailing_spaces_is_a_release(self) -> None:
        """Counts a release heading with trailing spaces as the release it is, so the older one is flagged."""
        lines = ["## Unreleased", "## v0.3.0 — 2026-09-01  ", "## v0.2.0 — 2026-08-01"]
        entries = parse_entries(K_PATH, _text(lines).splitlines())
        self.assertEqual(_rules(entries), ["rule_live_releases"])

    def test_a_bullet_inside_a_code_fence_is_not_an_entry(self) -> None:
        """Reads no entry from a bullet that sits between two fence lines."""
        self.assertEqual(parse_entries(K_PATH, ["# Changelog", "", "```", "- matter: not an entry.", "```"]), [])

    def test_an_entry_carries_the_line_it_is_on(self) -> None:
        """Numbers each entry by its line in the file, counting from one."""
        entries = parse_entries(K_PATH, ["# Changelog", "", "## Unreleased", K_GOOD])
        self.assertEqual([entry.number for entry in entries], [3, 4])


for _name, _lines, _expected in K_CASES:
    setattr(RuleTest, f"test_{_name}", _case_test(_name, _lines, _expected))


class ShellTest(unittest.TestCase):
    """Runs the shell on changelogs written to disk."""

    K_TEST_SIZE = "medium"

    def test_a_changelog_read_from_disk_fires_the_rule_its_entry_breaks(self) -> None:
        """Reads a changelog from a file, and fires the shape rule for the entry it holds."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "CHANGELOG.md"
            path.write_text(_text(["## Unreleased", "- matter: reserved the list."]), encoding="utf-8")
            self.assertEqual(_rules(collect([path])), ["rule_entry_shape"])


if __name__ == "__main__":
    unittest.main()
