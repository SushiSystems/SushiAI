# test_checker.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the command line every checker shares, from tools/checking/checker.py."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Checker, run  # noqa: E402

K_CHECKER = Checker(description="Checks nothing.", collect=lambda paths: [],
                    rules={"rule_first": lambda subject: iter(()), "rule_second": lambda subject: iter(())})


class CheckerTest(unittest.TestCase):
    """Runs a checker with two rules and no subjects."""

    def test_lists_its_rules_one_a_line(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = run(K_CHECKER, ["--list-rules"])
        self.assertEqual((code, output.getvalue()), (0, "rule_first\nrule_second\n"))


if __name__ == "__main__":
    unittest.main()
