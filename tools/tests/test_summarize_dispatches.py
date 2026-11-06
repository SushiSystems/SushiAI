# test_summarize_dispatches.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the reading and the totals of the dispatch summary, pure, and its folders on disk."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routing.summarize_dispatches import Seat, main, parse, summarize, summary_lines  # noqa: E402

K_LINES = (
    "# Report",
    "dispatch: rename the pass — haiku medium, 4 turns, passed",
    "dispatch: split the loader — haiku high, 9 turns, failed, rung 2",
    "dispatch: split the loader — sonnet medium, 6 turns, passed, rung 3",
    "dispatch: fix the docs — haiku medium, 2 turns, passed",
    "dispatch is mentioned here but this is prose",
)


class SummarizeDispatchesTest(unittest.TestCase):
    """Covers the line format, the per-seat totals and the summary table."""

    K_TEST_SIZE = "small"

    def test_reads_only_dispatch_lines(self) -> None:
        self.assertEqual(len(list(parse(K_LINES))), 4)

    def test_a_line_without_a_rung_is_on_the_first(self) -> None:
        first = next(parse(K_LINES))
        seen = (first.model, first.effort, first.rung, first.passed)
        self.assertEqual(seen, ("haiku", "medium", 1, True))

    def test_totals_per_model_and_effort(self) -> None:
        self.assertEqual(summarize(parse(K_LINES)), [
            Seat("haiku", "high", 1, 0, 9, 1),
            Seat("haiku", "medium", 2, 2, 6, 0),
            Seat("sonnet", "medium", 1, 1, 6, 1),
        ])

    def test_table_names_the_columns_then_one_row_per_seat(self) -> None:
        lines = summary_lines([("\n".join(K_LINES))])
        columns = ["model", "effort", "runs", "pass", "turns", "escalated"]
        self.assertEqual(lines[0].split(), columns)
        self.assertEqual([line.split() for line in lines[1:]], [
            ["haiku", "high", "1", "0%", "9.0", "1"],
            ["haiku", "medium", "2", "100%", "3.0", "0"],
            ["sonnet", "medium", "1", "100%", "6.0", "1"],
        ])

    def test_no_dispatch_line_in_any_report_says_so(self) -> None:
        self.assertEqual(summary_lines(["# Report\n", ""]), ["no dispatch lines found"])


class ReportsOnDiskTest(unittest.TestCase):
    """Covers which report files the command reads: REPORT.md under agent and archive only."""

    K_TEST_SIZE = "medium"

    def test_main_sums_only_the_reports_under_agent_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {
                "docs/agent/work/REPORT.md": "dispatch: a — haiku medium, 3 turns, passed\n",
                "docs/archive/old/REPORT.md": "dispatch: b — sonnet medium, 5 turns, failed\n",
                "docs/agent/work/PLAN.md": "dispatch: c — haiku medium, 1 turns, passed\n",
                "docs/other/REPORT.md": "dispatch: d — opus high, 1 turns, passed\n",
            }
            for name, text in files.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(text, encoding="utf-8")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = main([str(root)])
        rows = [line.split() for line in captured.getvalue().splitlines()[1:]]
        self.assertEqual((code, rows), (0, [["haiku", "medium", "1", "100%", "3.0", "0"],
                                            ["sonnet", "medium", "1", "0%", "5.0", "0"]]))


if __name__ == "__main__":
    unittest.main()
