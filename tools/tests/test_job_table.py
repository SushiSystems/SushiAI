# test_job_table.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the reading of a plan's Jobs table and the faults it reports."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routing.job_table import read_jobs  # noqa: E402

K_HEAD = "# Plan\n\n## Jobs\n\n| Job | Files | Waits on | Status |\n| --- | --- | --- | --- |\n"


K_TEST_SIZE = "small"


class JobTableTest(unittest.TestCase):
    """Covers a good table, each fault and the table's position in the document."""

    def test_good_table_yields_its_jobs_in_order(self) -> None:
        jobs, problems = read_jobs(K_HEAD + "| A | `a.py`, `b.py` | - | done |\n| B | c.py | A | todo |\n")
        self.assertEqual(problems, [])
        self.assertEqual([job.id for job in jobs], ["A", "B"])
        self.assertEqual((jobs[0].files, jobs[1].waits_on), (("a.py", "b.py"), ("A",)))

    def test_bad_status_is_reported_with_its_line(self) -> None:
        jobs, problems = read_jobs(K_HEAD + "| A | a.py | - | started |\n")
        self.assertEqual(jobs, [])
        self.assertEqual((problems[0].rule, problems[0].line), ("status", 7))

    def test_repeated_id_is_reported(self) -> None:
        jobs, problems = read_jobs(K_HEAD + "| A | a.py | - | todo |\n| A | b.py | - | todo |\n")
        self.assertEqual(len(jobs), 1)
        self.assertEqual((problems[0].rule, problems[0].line), ("repeated_id", 8))

    def test_row_with_a_missing_cell_is_reported(self) -> None:
        _, problems = read_jobs(K_HEAD + "| A | a.py | todo |\n")
        self.assertEqual(problems[0].rule, "columns")

    def test_missing_column_in_the_header_is_reported(self) -> None:
        text = "## Jobs\n\n| Job | Files | Status |\n| --- | --- | --- |\n| A | a.py | todo |\n"
        jobs, problems = read_jobs(text)
        self.assertEqual((jobs, problems[0].rule), ([], "columns"))

    def test_document_without_the_table_is_reported(self) -> None:
        self.assertEqual(read_jobs("# Plan\n\nNo jobs here.\n")[1][0].rule, "missing_table")

    def test_only_the_table_under_the_heading_is_read(self) -> None:
        text = "| Other | Table |\n| --- | --- |\n| x | y |\n\n" + K_HEAD + "| A | a.py | - | todo |\n\nText.\n| Z | z |\n"
        jobs, problems = read_jobs(text)
        self.assertEqual(([job.id for job in jobs], problems), (["A"], []))


if __name__ == "__main__":
    unittest.main()
