# test_next_jobs.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the exit codes and printed ids of the next-jobs command, pure and on a plan file."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routing.next_jobs import main, plan_report  # noqa: E402

K_HEAD = "## Jobs\n\n| Job | Files | Waits on | Status |\n| --- | --- | --- | --- |\n"


class PlanReportTest(unittest.TestCase):
    """Covers each of the four exit codes, the printed ids and the faults of a plan."""

    K_TEST_SIZE = "small"

    def test_prints_the_ready_jobs_and_exits_zero(self) -> None:
        text = K_HEAD + "| A | a | - | todo |\n| B | b | - | todo |\n"
        self.assertEqual(plan_report("PLAN.md", text, 2), (0, ["A", "B"]))

    def test_exits_one_when_jobs_remain_and_none_is_ready(self) -> None:
        text = K_HEAD + "| A | a | - | running |\n| B | b | A | todo |\n"
        self.assertEqual(plan_report("PLAN.md", text, 1), (1, []))

    def test_exits_two_when_every_job_is_done(self) -> None:
        self.assertEqual(plan_report("PLAN.md", K_HEAD + "| A | a | - | done |\n", 1), (2, []))

    def test_exits_three_on_a_malformed_table_and_names_the_line(self) -> None:
        code, lines = plan_report("PLAN.md", K_HEAD + "| A | a | - | later |\n", 1)
        expected = ["PLAN.md:5: status: `later` is not one of todo, running, done, failed"]
        self.assertEqual((code, lines), (3, expected))

    def test_exits_three_on_a_cycle(self) -> None:
        text = K_HEAD + "| A | a | B | todo |\n| B | b | A | todo |\n"
        code, lines = plan_report("PLAN.md", text, 1)
        self.assertEqual((code, [line.split(": ")[1] for line in lines]), (3, ["cycle", "cycle"]))


class MainTest(unittest.TestCase):
    """Covers the command on a plan file: the file is read and its output and code reach stdout."""

    K_TEST_SIZE = "medium"

    def test_main_reads_the_plan_file_and_prints_its_ids(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            plan = Path(folder) / "PLAN.md"
            text = K_HEAD + "| A | a | - | todo |\n| B | b | - | todo |\n"
            plan.write_text(text, encoding="utf-8")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = main([str(plan), "--free", "2"])
        self.assertEqual((code, captured.getvalue()), (0, "A\nB\n"))


if __name__ == "__main__":
    unittest.main()
