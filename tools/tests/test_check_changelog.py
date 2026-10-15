# test_check_changelog.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of the changelog checker on changelogs written to disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documentation.check_changelog import K_CHECKER, collect  # noqa: E402

K_GOOD = "- 2026-09-15 — matter: Reserved the neighbour list at tick start (`neighbour_list.hpp`)."


class ChangelogTest(unittest.TestCase):
    """Covers each rule of the changelog checker with one failing and one passing line."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.path = Path(self._folder.name) / "CHANGELOG.md"

    def tearDown(self) -> None:
        self._folder.cleanup()

    def _rules(self, *lines: str) -> list[str]:
        """Returns the name of every rule that fires on a changelog of the given lines."""
        self.path.write_text("# Changelog\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
        return [
            name for entry in collect([self.path])
            for name, rule in K_CHECKER.rules.items() for _ in rule(entry)
        ]

    def test_a_well_formed_changelog_is_clean(self) -> None:
        self.assertEqual(self._rules("## Unreleased", "", K_GOOD, "", "## v0.3.0 — 2026-09-01"), [])

    def test_a_heading_of_the_wrong_shape(self) -> None:
        self.assertEqual(self._rules("## Version 0.3"), ["rule_entry_shape"])

    def test_an_entry_of_the_wrong_shape(self) -> None:
        self.assertEqual(self._rules("## Unreleased", "- matter: reserved the list."), ["rule_entry_shape"])

    def test_a_second_live_release(self) -> None:
        rules = self._rules("## Unreleased", "## v0.3.0 — 2026-09-01", "## v0.2.0 — 2026-08-01")
        self.assertEqual(rules, ["rule_live_releases"])

    def test_an_entry_above_the_length_ceiling(self) -> None:
        long = K_GOOD.replace("tick start", "tick start " + "and again " * 20)
        self.assertEqual(self._rules("## Unreleased", long), ["rule_length"])

    def test_an_entry_of_two_sentences(self) -> None:
        two = K_GOOD.replace(" at tick start", ". It saves time")
        self.assertEqual(self._rules("## Unreleased", two), ["rule_single_sentence"])

    def test_a_nested_bullet(self) -> None:
        self.assertIn("rule_no_nesting", self._rules("## Unreleased", K_GOOD, "  - a detail."))

    def test_an_entry_citing_six_places(self) -> None:
        places = ", ".join(f"`f{index}.hpp`" for index in range(6))
        self.assertEqual(self._rules("## Unreleased", K_GOOD.replace("`neighbour_list.hpp`", places)), ["rule_cited_paths"])


if __name__ == "__main__":
    unittest.main()
