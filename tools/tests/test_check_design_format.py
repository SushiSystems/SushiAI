# test_check_design_format.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rules of the design format checker."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Issue  # noqa: E402
from documentation.check_design_format import (  # noqa: E402
    K_CHECKER,
    Repository,
    rule_no_commit_hashes,
    rule_no_execution_record,
    rule_phase_table,
    rule_sections,
    rule_status_length,
)

K_DOCUMENT = """# Kernels

**Status:** Open — phase K1 is building

## Problem

The kernels wait on a recorder.

## Decision

One recorder serves every kernel.

## Design

```
## Not a section
```

## Phases

| Id | Phase | State |
| --- | --- | --- |
| K0 | The recorder | built |
| K1 | The first kernel | building |

## Open questions

None.
"""

K_PATH = "docs/design/KERNELS.md"


def _repository(folder: str, files: dict[str, str]) -> Repository:
    """Writes the given files under a root and returns its repository."""
    root = Path(folder) / "alpha"
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Repository(root)


def _messages(issues: Iterable[Issue]) -> list[str]:
    """Returns the message of each issue."""
    return [issue.message for issue in issues]


def _check(rule: Callable[[Repository], Iterator[Issue]], text: str, name: str = K_PATH) -> list[str]:
    """Returns the messages a rule gives for one design document."""
    with tempfile.TemporaryDirectory() as folder:
        return _messages(rule(_repository(folder, {name: text})))


class RegistrationTest(unittest.TestCase):
    """Checks the rule table."""

    def test_names_every_rule(self) -> None:
        """Holds the five rules under their names."""
        self.assertEqual(
            sorted(K_CHECKER.rules),
            ["rule_no_commit_hashes", "rule_no_execution_record", "rule_phase_table", "rule_sections", "rule_status_length"],
        )


class ValidDocumentTest(unittest.TestCase):
    """Checks that a document written to the format passes every rule."""

    def test_reports_nothing(self) -> None:
        """Runs every rule over the valid document."""
        for name, rule in K_CHECKER.rules.items():
            self.assertEqual(_check(rule, K_DOCUMENT), [], name)

    def test_leaves_the_indexes_and_phase_files_alone(self) -> None:
        """Skips the design map, the backlog and a phase file of a split topic."""
        with tempfile.TemporaryDirectory() as folder:
            files = {
                "docs/design/README.md": "# Map\n",
                "docs/design/REMAINING_WORK.md": "# Remaining work\n",
                "docs/design/KERNELS/PHASE_ONE.md": "# Phase one\n\n## Anything\n",
            }
            self.assertEqual(_repository(folder, files).documents(), [])

    def test_reads_the_readme_of_a_split_topic(self) -> None:
        """Counts a split topic's README as its document."""
        messages = _check(rule_sections, "# Kernels\n", "docs/design/KERNELS/README.md")
        self.assertIn("the section '## Problem' is missing", messages)


class SectionsTest(unittest.TestCase):
    """Covers the five sections and their order."""

    def test_reports_a_foreign_section_and_a_missing_one(self) -> None:
        """Reports a section outside the five and one of the five that is absent."""
        text = K_DOCUMENT.replace("## Design\n", "## Notes\n")
        self.assertEqual(len(_check(rule_sections, text)), 2)

    def test_reports_the_wrong_order(self) -> None:
        """Reports the five sections in another order."""
        text = K_DOCUMENT.replace("## Decision\n", "## Temp\n").replace("## Design\n", "## Decision\n").replace("## Temp\n", "## Design\n")
        self.assertEqual(len(_check(rule_sections, text)), 1)

    def test_ignores_a_heading_inside_a_fence(self) -> None:
        """Does not read a fenced heading as a section."""
        self.assertEqual(_check(rule_sections, K_DOCUMENT), [])


class StatusLengthTest(unittest.TestCase):
    """Covers the status ceiling."""

    def test_reports_a_long_status(self) -> None:
        """Reports a status line above 160 characters."""
        text = K_DOCUMENT.replace("phase K1 is building", "x" * 160)
        self.assertEqual(len(_check(rule_status_length, text)), 1)

    def test_accepts_a_status_at_the_ceiling(self) -> None:
        """Accepts a status line of exactly 160 characters."""
        line = "**Status:** Open — "
        text = K_DOCUMENT.replace(line + "phase K1 is building", line + "x" * (160 - len(line)))
        self.assertEqual(_check(rule_status_length, text), [])


class PhaseTableTest(unittest.TestCase):
    """Covers the table of phases."""

    def test_reports_other_columns(self) -> None:
        """Reports a header that is not Id, Phase and State."""
        text = K_DOCUMENT.replace("| Id | Phase | State |", "| Id | What | Done |")
        self.assertEqual(len(_check(rule_phase_table, text)), 1)

    def test_reports_an_unknown_state(self) -> None:
        """Reports a state outside the four."""
        self.assertEqual(len(_check(rule_phase_table, K_DOCUMENT.replace("| building |", "| soon |"))), 1)

    def test_reports_a_section_with_neither_table_nor_none(self) -> None:
        """Reports prose where the table goes."""
        text = K_DOCUMENT[:K_DOCUMENT.index("| Id |")] + "Phases follow later.\n\n## Open questions\n\nNone.\n"
        self.assertEqual(len(_check(rule_phase_table, text)), 1)

    def test_accepts_none(self) -> None:
        """Accepts a section that says None."""
        text = K_DOCUMENT[:K_DOCUMENT.index("| Id |")] + "None.\n\n## Open questions\n\nNone.\n"
        self.assertEqual(_check(rule_phase_table, text), [])


class CommitHashTest(unittest.TestCase):
    """Covers the rule against commit hashes."""

    def test_reports_a_hash_and_a_range(self) -> None:
        """Reports a hash and a range of two, each in a code span."""
        text = K_DOCUMENT + "\nBuilt in `24fe81c9`, reviewed in `1587108c..49660a6b`.\n"
        self.assertEqual(len(_check(rule_no_commit_hashes, text)), 2)

    def test_reports_a_bare_hash(self) -> None:
        """Reports a hash written in prose outside a code span."""
        self.assertEqual(len(_check(rule_no_commit_hashes, K_DOCUMENT + "\nBuilt in 24fe81c9 first.\n")), 1)

    def test_leaves_a_word_and_a_number_alone(self) -> None:
        """Does not read a long number or a name in a code span as a hash."""
        text = K_DOCUMENT + "\nThe `1234567` slots and `decade` stay.\n"
        self.assertEqual(_check(rule_no_commit_hashes, text), [])


class ExecutionRecordTest(unittest.TestCase):
    """Covers the rule against a dated record of work done."""

    def test_reports_a_verb_beside_a_date_in_either_order(self) -> None:
        """Reports work dated after its verb and before it."""
        text = K_DOCUMENT + "\nThe cook landed on 2026-09-29.\nOn 2026-09-30 the solver was merged.\n"
        self.assertEqual(len(_check(rule_no_execution_record, text)), 2)

    def test_leaves_a_ruling_and_the_status_alone(self) -> None:
        """Does not read an approval date or the status line as a record of work."""
        text = K_DOCUMENT.replace("phase K1 is building", "built 2026-09-29")
        text += "\nThe owner approved it on 2026-10-04.\n"
        self.assertEqual(_check(rule_no_execution_record, text), [])

    def test_ignores_a_fenced_example(self) -> None:
        """Does not read an example in a fenced block."""
        text = K_DOCUMENT + "\n```\nlanded 2026-09-29\n```\n"
        self.assertEqual(_check(rule_no_execution_record, text), [])


if __name__ == "__main__":
    unittest.main()
