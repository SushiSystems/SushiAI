# test_test_times_settings.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests that the Time table of the testing skill and K_TEST_TIMES in settings.py agree."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402

K_ROOT = Path(__file__).resolve().parents[2]
K_SKILL = K_ROOT / "skills" / "testing" / "SKILL.md"
K_HEADING = "## Time"
K_ROW = re.compile(r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<value>[^|]+?)\s*\|$")
K_VALUE = re.compile(r"(?P<number>\d+(?: \d{3})*)\s*(?P<unit>ms|s)\b")
K_LABELS = {
    "A `small` test": "small_test_s",
    "A `medium` test after the shared fixture": "medium_test_s",
    "A repository's pre-merge run (`small` and `medium`), wall": "pre_merge_s",
}


K_TEST_SIZE = "medium"


def read_times(text: str) -> dict[str, float]:
    """Returns the settings key and the value in seconds of each row in the Time table of a skill's text."""
    values: dict[str, float] = {}
    inside = False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line.strip() == K_HEADING
            continue
        match = K_ROW.match(line) if inside else None
        if match is None or match["label"] not in K_LABELS:
            continue
        value = K_VALUE.match(match["value"])
        if value is not None:
            number = float(value["number"].replace(" ", ""))
            values[K_LABELS[match["label"]]] = number / 1000 if value["unit"] == "ms" else number
    return values


def differences(table: dict[str, float], constants: dict[str, float]) -> list[str]:
    """Returns one sentence for each key whose value differs between the table and the constants."""
    keys = sorted(set(table) | set(constants))
    return [f"{key}: the skill says {table.get(key)}, settings.py says {constants.get(key)}"
            for key in keys if table.get(key) != constants.get(key)]


class TestTimesSettingsTest(unittest.TestCase):
    """Reads the real skill when the repository has one, and sample tables otherwise."""

    def test_the_skill_table_matches_the_settings(self) -> None:
        if not K_SKILL.is_file():
            self.skipTest("this tree holds no skills/ folder")
        self.assertEqual(differences(read_times(K_SKILL.read_text(encoding="utf-8")), settings.K_TEST_TIMES), [])

    def test_reads_milliseconds_seconds_and_a_thousands_space_only_under_the_heading(self) -> None:
        text = ("## Time\n\n| Limit | Value |\n| --- | --- |\n"
                "| A `small` test | 100 ms |\n"
                "| A repository's pre-merge run (`small` and `medium`), wall | 1 200 s |\n"
                "| A row no key names | 5 s |\n\n## Other\n\n| The `small` tests of one module | 99 s |\n")
        self.assertEqual(read_times(text), {"small_test_s": 0.1, "pre_merge_s": 1200.0})

    def test_names_each_key_that_differs(self) -> None:
        table = {"small_test_s": 0.1, "medium_test_s": 5.0}
        constants = {"small_test_s": 0.2, "medium_test_s": 5.0, "pre_merge_s": 600.0}
        self.assertEqual(differences(table, constants), [
            "pre_merge_s: the skill says None, settings.py says 600.0",
            "small_test_s: the skill says 0.1, settings.py says 0.2",
        ])

    def test_every_row_of_the_table_has_a_constant(self) -> None:
        self.assertEqual(set(K_LABELS.values()), set(settings.K_TEST_TIMES))


if __name__ == "__main__":
    unittest.main()
