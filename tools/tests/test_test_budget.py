# test_test_budget.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests tools/testing/test_budget.py on JUnit files built in a temporary folder."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing import test_budget  # noqa: E402
from testing.test_budget import Timing, evaluate, main, read_exemptions, read_junit, suite_and_size  # noqa: E402

K_LIMITS = {"small_test_s": 1, "medium_test_s": 10, "large_test_s": 120, "small_program_s": 10,
            "large_program_s": 600, "default_run_s": 300, "large_tier_s": 1800}


def junit(cases: dict[str, float], skipped: tuple[str, ...] = ()) -> str:
    """Returns the text of a JUnit file with one test case per entry."""
    rows = "".join(f'<testcase name="{name}" classname="{name}" time="{seconds}" status="run"/>'
                   for name, seconds in cases.items())
    rows += "".join(f'<testcase name="{name}" classname="{name}" time="0" status="run"><skipped/></testcase>'
                    for name in skipped)
    return f'<?xml version="1.0"?><testsuite name="(empty)" tests="{len(cases)}">{rows}</testsuite>'


def run(timings: dict[str, float], **overrides):
    """Evaluates a run with the test limits and default arguments, changed by *overrides*."""
    arguments = dict(programs={}, exemptions={}, rerun=None, scope="partial", wall=None, factor=1.25,
                     added=None, base="", rebuilt=True, open_backlog=None)
    arguments.update(overrides)
    with mock.patch.object(test_budget, "K_BUDGET", K_LIMITS):
        return evaluate([Timing(name, seconds) for name, seconds in timings.items()], **arguments)


class ReadingTest(unittest.TestCase):
    """Reads the JUnit file, the names and the exemption file."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def test_reads_times_and_leaves_out_a_skipped_test(self) -> None:
        path = self.folder / "a.xml"
        path.write_text(junit({"Small_A.One": 0.25}, skipped=("Large_B.Gpu",)), encoding="utf-8")
        self.assertEqual(read_junit(path), [Timing("Small_A.One", 0.25)])

    def test_refuses_a_file_with_a_doctype_or_an_entity(self) -> None:
        path = self.folder / "b.xml"
        path.write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><testsuite/>', encoding="utf-8")
        with self.assertRaises(SystemExit):
            read_junit(path)

    def test_finds_the_suite_and_size_of_a_plain_and_a_parametrised_name(self) -> None:
        self.assertEqual(suite_and_size("Small_Shape.Rank"), ("Small_Shape", "small"))
        self.assertEqual(suite_and_size("Rows/Medium_Io.Read/3"), ("Medium_Io", "medium"))
        self.assertEqual(suite_and_size("Benchmark_Wave.Run"), (None, None))

    def test_reads_the_exemption_table(self) -> None:
        path = self.folder / "e.toml"
        path.write_text('[exempt]\n"Large_Matter.Crush" = "RW-40"\n', encoding="utf-8")
        self.assertEqual(read_exemptions(path), {"Large_Matter.Crush": "RW-40"})


class EvaluateTest(unittest.TestCase):
    """Applies the limits."""

    def test_a_run_inside_every_limit_has_no_finding(self) -> None:
        report = run({"Small_A.One": 0.2, "Medium_B.Two": 3.0}, programs={"Small_A": "core", "Medium_B": "io"},
                     scope="default")
        self.assertEqual(report.findings, ())
        self.assertEqual(report.line, "budget: small 0.2 s of 10 s (core), default 3 s sum of 300 s")

    def test_a_test_over_its_limit_is_named_for_a_rerun_and_judged_by_the_rerun(self) -> None:
        first = run({"Small_A.Slow": 1.5})
        self.assertEqual(first.rerun, ("Small_A.Slow",))
        self.assertIn("1.5 s, over the small limit of 1 s (not rerun)", first.findings[0])
        quiet = run({"Small_A.Slow": 1.5}, rerun=[Timing("Small_A.Slow", 0.4)])
        self.assertEqual(quiet.findings, ())
        slow = run({"Small_A.Slow": 1.5}, rerun=[Timing("Small_A.Slow", 1.4)])
        self.assertIn("1.4 s alone, over the small limit of 1 s", slow.findings[0])

    def test_a_program_total_fails_only_above_the_limit_times_the_factor(self) -> None:
        small = {f"Small_A.T{index}": 1.0 for index in range(12)}
        ok = run(small, programs={"Small_A": "core"}, factor=1.25)
        self.assertEqual([f for f in ok.findings if "core" in f], [])
        bad = run({f"Small_A.T{index}": 1.0 for index in range(14)}, programs={"Small_A": "core"}, factor=1.25)
        self.assertIn("core: its small tests take 14.0 s, over 10 s x 1.25", bad.findings)

    def test_the_default_run_is_judged_by_wall_time_when_given(self) -> None:
        timings = {"Medium_B.One": 9.0, "Medium_C.Two": 9.0}
        self.assertEqual(run(timings, scope="default", wall=500.0).findings[-1][:12], "default run:")
        self.assertEqual(run(timings, scope="default", wall=100.0).findings, ())

    def test_a_partial_run_judges_no_whole_run_limit(self) -> None:
        report = run({"Medium_B.One": 9.0}, scope="partial", wall=9999.0)
        self.assertEqual(report.findings, ())
        self.assertIn("whole run not measured", report.line)

    def test_an_exempt_test_is_left_out_and_a_stale_exemption_is_a_finding(self) -> None:
        exempt = {"Large_M.Crush": "RW-40", "Small_A.Fine": "RW-41"}
        report = run({"Large_M.Crush": 700.0, "Small_A.Fine": 0.1}, exemptions=exempt, scope="large",
                     open_backlog=frozenset({"RW-40", "RW-41"}))
        self.assertEqual(len(report.findings), 1)
        self.assertIn("Small_A.Fine: exempt as RW-41 but within its limit", report.findings[0])
        self.assertIn("2 exempt, 700 s left out", report.line)

    def test_an_exemption_needs_an_open_backlog_id(self) -> None:
        report = run({"Large_M.Crush": 700.0}, exemptions={"Large_M.Crush": "RW-40"}, open_backlog=frozenset())
        self.assertIn("'RW-40' is not an open backlog ID", report.findings[0])

    def test_a_run_that_did_not_rebuild_says_so_and_a_count_is_printed(self) -> None:
        report = run({"Small_A.One": 0.1}, rebuilt=False, added=7, base="origin/main")
        self.assertTrue(report.line.endswith("7 cases added since origin/main (not rebuilt)"))


class MainTest(unittest.TestCase):
    """Runs the command."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.path = Path(self._folder.name) / "run.xml"
        self.path.write_text(junit({"Small_A.Slow": 5.0}), encoding="utf-8")
        patch = mock.patch.object(test_budget, "K_BUDGET", K_LIMITS)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def call(self, *extra: str) -> tuple[int, list[str]]:
        """Returns the exit code and the printed lines of one command."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([str(self.path), *extra])
        return code, out.getvalue().splitlines()

    def test_prints_the_rerun_line_and_exits_clean_without_gate(self) -> None:
        code, lines = self.call()
        self.assertEqual(code, 0)
        self.assertEqual(lines[1], "rerun: ^(Small_A\\.Slow)$")

    def test_gate_exits_one_on_a_finding_and_a_run_that_did_not_rebuild_never_gates(self) -> None:
        self.assertEqual(self.call("--gate")[0], 1)
        self.assertEqual(self.call("--gate", "--not-rebuilt")[0], 0)

    def test_reads_the_program_map(self) -> None:
        programs = Path(self._folder.name) / "map.json"
        programs.write_text(json.dumps({"Small_A": "core"}), encoding="utf-8")
        self.assertIn("(core)", self.call("--programs", str(programs))[1][0])


if __name__ == "__main__":
    unittest.main()
