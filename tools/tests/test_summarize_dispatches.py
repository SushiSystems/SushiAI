# test_summarize_dispatches.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the reading and the totals of the dispatch summary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routing.summarize_dispatches import Seat, parse, summarize  # noqa: E402

K_LINES = (
    "# Report",
    "dispatch: rename the pass — haiku medium, 4 turns, passed",
    "dispatch: split the loader — haiku high, 9 turns, failed, rung 2",
    "dispatch: split the loader — sonnet medium, 6 turns, passed, rung 3",
    "dispatch: fix the docs — haiku medium, 2 turns, passed",
    "dispatch is mentioned here but this is prose",
)


class SummarizeDispatchesTest(unittest.TestCase):
    """Covers the line format and the per-seat totals."""

    def test_reads_only_dispatch_lines(self) -> None:
        self.assertEqual(len(list(parse(K_LINES))), 4)

    def test_a_line_without_a_rung_is_on_the_first(self) -> None:
        first = next(parse(K_LINES))
        self.assertEqual((first.model, first.effort, first.rung, first.passed), ("haiku", "medium", 1, True))

    def test_totals_per_model_and_effort(self) -> None:
        self.assertEqual(summarize(parse(K_LINES)), [
            Seat("haiku", "high", 1, 0, 9, 1),
            Seat("haiku", "medium", 2, 2, 6, 0),
            Seat("sonnet", "medium", 1, 1, 6, 1),
        ])


if __name__ == "__main__":
    unittest.main()
