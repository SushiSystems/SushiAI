# test_test_budget.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests tools/testing/test_budget.py: its rules on values in memory, its command line once on files on disk."""

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
from testing.test_budget import (  # noqa: E402
    Report,
    Timing,
    evaluate,
    exit_code,
    main,
    parse_exemptions,
    parse_junit,
    parse_programs,
    render,
    suite_and_size,
)

K_LIMITS = {"small_test_s": 1, "medium_test_s": 10, "pre_merge_s": 300}


def junit(cases: dict[str, float], skipped: tuple[str, ...] = ()) -> str:
    """Returns the text of a JUnit file with one test case per entry."""
    rows = "".join(f'<testcase name="{name}" classname="{name}" time="{seconds}" status="run"/>'
                   for name, seconds in cases.items())
    rows += "".join(f'<testcase name="{name}" classname="{name}" time="0" status="run"><skipped/></testcase>'
                    for name in skipped)
    return f'<?xml version="1.0"?><testsuite name="(empty)" tests="{len(cases)}">{rows}</testsuite>'


def judge(timings: list[Timing], **overrides) -> Report:
    """Evaluates a run of the given timings with the test limits and default arguments, changed by *overrides*."""
    arguments = dict(programs={}, exemptions={}, rerun=None, scope="partial", wall=None, factor=1.25,
                     added=None, base="", rebuilt=True, open_backlog=None)
    arguments.update(overrides)
    with mock.patch.object(test_budget, "K_BUDGET", K_LIMITS):
        return evaluate(timings, **arguments)


def run(timings: dict[str, float], **overrides) -> Report:
    """Evaluates a run whose tests are given as name and seconds."""
    return judge([Timing(name, seconds) for name, seconds in timings.items()], **overrides)


class TestReadingAndNames(unittest.TestCase):
    """Reads the JUnit text, the names, the exemption table and the program map."""
    K_TEST_SIZE = "small"

    def test_reads_times_and_leaves_out_a_skipped_test(self) -> None:
        self.assertEqual(parse_junit(junit({"Small_A.One": 0.25}, skipped=("Large_B.Gpu",)).encode(), "a.xml"),
                         [Timing("Small_A.One", 0.25)])

    def test_refuses_a_file_with_a_doctype_or_an_entity(self) -> None:
        with self.assertRaises(SystemExit):
            parse_junit(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><testsuite/>', "b.xml")

    def test_finds_the_suite_and_size_of_a_plain_and_a_parametrised_name(self) -> None:
        self.assertEqual(suite_and_size("Small_Shape.Rank"), ("Small_Shape", "small"))
        self.assertEqual(suite_and_size("Rows/Medium_Io.Read/3"), ("Medium_Io", "medium"))
        self.assertEqual(suite_and_size("Benchmark_Wave.Run"), (None, None))

    def test_reads_the_exemption_table(self) -> None:
        self.assertEqual(parse_exemptions('[exempt]\n"Large_Matter.Crush" = "RW-40"\n'),
                         {"Large_Matter.Crush": "RW-40"})

    def test_reads_the_program_map(self) -> None:
        self.assertEqual(parse_programs('{"Small_A": "core"}'), {"Small_A": "core"})

    def test_the_shape_counts_a_case_by_its_name_or_its_size_property(self) -> None:
        text = ('<?xml version="1.0"?><testsuite name="(empty)">'
                '<testcase name="Small_A.One" classname="Small_A.One" time="0.1" status="run"/>'
                '<testcase name="Large_B.Two" classname="Large_B.Two" time="1" status="run"/>'
                '<testcase name="Benchmark_C.Three" classname="Benchmark_C.Three" time="1" status="run"/>'
                '<testcase name="test_pytest_case" classname="tests.test_x" time="0.2" status="run">'
                '<properties><property name="size" value="large"/></properties></testcase>'
                '<testcase name="test_unsized" classname="tests.test_x" time="0.2" status="run"/>'
                '</testsuite>')
        report = judge(parse_junit(text.encode(), "run.xml"))
        self.assertIn("shape 33% small, 0% medium, 67% large", report.line)


class TestEvaluate(unittest.TestCase):
    """Applies the limits."""
    K_TEST_SIZE = "small"

    def test_a_run_inside_every_limit_has_no_finding(self) -> None:
        report = run({"Small_A.One": 0.2, "Medium_B.Two": 3.0}, programs={"Small_A": "core", "Medium_B": "io"},
                     scope="default")
        self.assertEqual(report.findings, ())
        self.assertEqual(report.line,
                         "budget: small 0.2 s (core), pre-merge 3 s sum of 300 s, shape 50% small, 50% medium, 0% large")

    def test_a_large_test_has_no_time_limit_and_counts_in_the_shape(self) -> None:
        report = run({"Large_M.Crush": 5000.0, "Small_A.One": 0.1, "Small_A.Two": 0.1, "Small_A.Three": 0.1},
                     scope="large", wall=5001.0)
        self.assertEqual(report.findings, ())
        self.assertEqual(report.rerun, ())
        self.assertIn("large tier 5000 s sum, 5001 s wall", report.line)
        self.assertIn("shape 75% small, 0% medium, 25% large", report.line)

    def test_a_test_over_its_limit_is_named_for_a_rerun_and_judged_by_the_rerun(self) -> None:
        first = run({"Small_A.Slow": 1.5})
        self.assertEqual(first.rerun, ("Small_A.Slow",))
        self.assertIn("1.5 s, over the small limit of 1 s (not rerun)", first.findings[0])
        quiet = run({"Small_A.Slow": 1.5}, rerun=[Timing("Small_A.Slow", 0.4)])
        self.assertEqual(quiet.findings, ())
        slow = run({"Small_A.Slow": 1.5}, rerun=[Timing("Small_A.Slow", 1.4)])
        self.assertIn("1.4 s alone, over the small limit of 1 s", slow.findings[0])

    def test_a_module_small_total_is_printed_and_never_a_finding(self) -> None:
        report = run({f"Small_A.T{index}": 0.05 for index in range(2000)}, programs={"Small_A": "core"})
        self.assertEqual(report.findings, ())
        self.assertIn("small 100.0 s (core)", report.line)

    def test_the_default_run_is_judged_by_wall_time_when_given(self) -> None:
        timings = {"Medium_B.One": 9.0, "Medium_C.Two": 9.0}
        self.assertEqual(run(timings, scope="default", wall=500.0).findings[-1][:14], "pre-merge run:")
        self.assertEqual(run(timings, scope="default", wall=100.0).findings, ())

    def test_a_partial_run_judges_no_whole_run_limit(self) -> None:
        report = run({"Medium_B.One": 9.0}, scope="partial", wall=9999.0)
        self.assertEqual(report.findings, ())
        self.assertIn("whole run not measured", report.line)

    def test_an_exempt_test_is_left_out_and_a_stale_exemption_is_a_finding(self) -> None:
        exempt = {"Medium_M.Crush": "RW-40", "Small_A.Fine": "RW-41"}
        report = run({"Medium_M.Crush": 700.0, "Small_A.Fine": 0.1}, exemptions=exempt, scope="default",
                     open_backlog=frozenset({"RW-40", "RW-41"}))
        self.assertEqual(len(report.findings), 1)
        self.assertIn("Small_A.Fine: exempt as RW-41 but within its limit", report.findings[0])
        self.assertIn("2 exempt, 700 s left out", report.line)

    def test_an_exemption_needs_an_open_backlog_id(self) -> None:
        report = run({"Medium_M.Crush": 700.0}, exemptions={"Medium_M.Crush": "RW-40"}, open_backlog=frozenset())
        self.assertIn("'RW-40' is not an open backlog ID", report.findings[0])

    def test_a_run_that_did_not_rebuild_says_so_and_a_count_is_printed(self) -> None:
        report = run({"Small_A.One": 0.1}, rebuilt=False, added=7, base="origin/main")
        self.assertTrue(report.line.startswith("budget: "))
        self.assertTrue(report.line.endswith("7 cases added since origin/main (not rebuilt)"))


class TestRendering(unittest.TestCase):
    """Prints the lines of a run and decides its exit code."""
    K_TEST_SIZE = "small"

    def test_prints_the_rerun_line_and_exits_clean_without_gate(self) -> None:
        report = run({"Small_A.Slow": 5.0})
        self.assertEqual((render(report, rerun_given=False)[1], exit_code(report, gate=False, rebuilt=True)),
                         ("rerun: ^(Small_A\\.Slow)$", 0))

    def test_gate_exits_one_on_a_finding_and_a_run_that_did_not_rebuild_never_gates(self) -> None:
        report = run({"Small_A.Slow": 5.0})
        self.assertEqual((exit_code(report, gate=True, rebuilt=True), exit_code(report, gate=True, rebuilt=False)),
                         (1, 0))


class TestMainCommand(unittest.TestCase):
    """Runs the command on a JUnit file and a program map written to a temporary folder."""
    K_TEST_SIZE = "medium"

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.junit = Path(self._folder.name) / "run.xml"
        self.junit.write_text(junit({"Small_A.Slow": 5.0}), encoding="utf-8")
        self.programs = Path(self._folder.name) / "map.json"
        self.programs.write_text(json.dumps({"Small_A": "core"}), encoding="utf-8")
        patch = mock.patch.object(test_budget, "K_BUDGET", K_LIMITS)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def test_reads_the_junit_and_the_program_map_and_gates_the_run(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([str(self.junit), "--programs", str(self.programs), "--gate"])
        self.assertEqual((code, out.getvalue().splitlines()), (1, [
            "budget: small 5.0 s (core), whole run not measured, shape 100% small, 0% medium, 0% large",
            "rerun: ^(Small_A\\.Slow)$",
            "Small_A.Slow: 5.0 s, over the small limit of 1 s (not rerun)",
        ]))


if __name__ == "__main__":
    unittest.main()
