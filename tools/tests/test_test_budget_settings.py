# test_test_budget_settings.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests that the budget table of the testing skill and K_TEST_BUDGET in settings.py agree."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402

K_ROOT = Path(__file__).resolve().parents[2]
K_SKILL = K_ROOT / "skills" / "testing" / "SKILL.md"
K_HEADING = "## Budget"
K_ROW = re.compile(r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<value>[^|]+?)\s*\|$")
K_NUMBER = re.compile(r"\d+(?: \d{3})*")
K_LABELS = {
    "A `small` test": "small_test_s",
    "A `medium` test": "medium_test_s",
    "A new `large` test": "large_test_s",
    "The `small` tests of one test program": "small_program_s",
    "The `large` tests of one test program": "large_program_s",
    "The default run (`small` and `medium`) of a repository": "default_run_s",
    "The whole `large` tier of a repository": "large_tier_s",
    "Test cases added by one commit": "cases_per_commit",
    "Test cases added by one `fix` commit": "cases_per_fix",
    "A second test program in one module, added by one commit": "second_program",
}


def read_budget(text: str) -> dict[str, int]:
    """Returns the settings key and number of each row in the Budget table of a skill's text."""
    values: dict[str, int] = {}
    inside = False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line.strip() == K_HEADING
            continue
        match = K_ROW.match(line) if inside else None
        if match is None or match["label"] not in K_LABELS:
            continue
        number = K_NUMBER.match(match["value"])
        if number is not None:
            values[K_LABELS[match["label"]]] = int(number.group().replace(" ", ""))
    return values


def differences(table: dict[str, int], constants: dict[str, int]) -> list[str]:
    """Returns one sentence for each key whose number differs between the table and the constants."""
    keys = sorted(set(table) | set(constants))
    return [f"{key}: the skill says {table.get(key)}, settings.py says {constants.get(key)}"
            for key in keys if table.get(key) != constants.get(key)]


class TestBudgetSettingsTest(unittest.TestCase):
    """Reads the real skill when the repository has one, and sample tables otherwise."""

    def test_the_skill_table_matches_the_settings(self) -> None:
        if not K_SKILL.is_file():
            self.skipTest("this tree holds no skills/ folder")
        self.assertEqual(differences(read_budget(K_SKILL.read_text(encoding="utf-8")), settings.K_TEST_BUDGET), [])

    def test_reads_a_number_with_a_thousands_space_and_a_trailing_note(self) -> None:
        text = ("## Budget\n\n| Limit | Value |\n| --- | --- |\n"
                "| The whole `large` tier of a repository | 1 800 s |\n"
                "| A second test program in one module, added by one commit | 0 without the owner |\n"
                "| A row no key names | 5 |\n\n## Other\n\n| A `small` test | 99 s |\n")
        self.assertEqual(read_budget(text), {"large_tier_s": 1800, "second_program": 0})

    def test_names_each_key_that_differs(self) -> None:
        table = {"small_test_s": 1, "medium_test_s": 10}
        constants = {"small_test_s": 2, "medium_test_s": 10, "large_test_s": 120}
        self.assertEqual(differences(table, constants), [
            "large_test_s: the skill says None, settings.py says 120",
            "small_test_s: the skill says 1, settings.py says 2",
        ])

    def test_every_row_of_the_table_has_a_constant(self) -> None:
        self.assertEqual(set(K_LABELS.values()), set(settings.K_TEST_BUDGET))


if __name__ == "__main__":
    unittest.main()
