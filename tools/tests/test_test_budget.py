# test_test_budget.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests tools/testing/test_budget.py: its rules on values in memory, its command line once on files on disk."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.test_budget import (  # noqa: E402
    Report,
    Timing,
    evaluate,
    exit_code,
    main,
    missing_limits,
    parse_exemptions,
    parse_gtest_xml,
    parse_junit,
    parse_programs,
    render,
    suite_and_size,
)

K_LIMITS = {"small_test_s": 1, "medium_test_s": 10, "large_test_s": 60, "pre_merge_s": 300}


def junit(cases: dict[str, float], skipped: tuple[str, ...] = ()) -> str:
    """Returns the text of a JUnit file with one test case per entry."""
    rows = "".join(f'<testcase name="{name}" classname="{name}" time="{seconds}" status="run"/>'
                   for name, seconds in cases.items())
    rows += "".join(f'<testcase name="{name}" classname="{name}" time="0" status="run"><skipped/></testcase>'
                    for name in skipped)
    return f'<?xml version="1.0"?><testsuite name="(empty)" tests="{len(cases)}">{rows}</testsuite>'


def gtest_xml(cases: dict[tuple[str, str], float], skipped: tuple[tuple[str, str], ...] = ()) -> str:
    """Returns the text of a GoogleTest XML file with one case per (suite, case) entry."""
    rows = "".join(f'<testcase name="{case}" status="run" result="completed" time="{seconds}" classname="{suite}"/>'
                   for (suite, case), seconds in cases.items())
    rows += "".join(f'<testcase name="{case}" status="run" result="skipped" time="0" classname="{suite}"/>'
                    for suite, case in skipped)
    return f'<?xml version="1.0"?><testsuites tests="{len(cases)}"><testsuite name="S">{rows}</testsuite></testsuites>'


def judge(timings: list[Timing], **overrides) -> Report:
    """Evaluates a run of the given timings with the test limits and default arguments, changed by *overrides*."""
    arguments = dict(programs={}, exemptions={}, rerun=None, scope="partial", wall=None, factor=1.25,
                     added=None, base="", rebuilt=True, open_backlog=None)
    arguments.update(overrides)
    return evaluate(timings, budget=K_LIMITS, **arguments)


def small_timings(count: int, seconds: float) -> dict[str, float]:
    """Returns *count* small tests of one suite, each taking *seconds*."""
    return {f"Small_A.T{index}": seconds for index in range(count)}


def run(timings: dict[str, float], **overrides) -> Report:
    """Evaluates a run whose tests are given as name and seconds."""
    return judge([Timing(name, seconds) for name, seconds in timings.items()], **overrides)


class TestReadingAndNames(unittest.TestCase):
    """Reads the JUnit text, the names, the exemption table and the program map."""
    K_TEST_SIZE = "small"

    def test_reads_times_and_leaves_out_a_skipped_test(self) -> None:
        self.assertEqual(parse_junit(junit({"Small_A.One": 0.25}, skipped=("Large_B.Gpu",)).encode(), "a.xml"),
                         [Timing("Small_A.One", 0.25)])

    def test_reads_a_gtest_case_as_suite_dot_case_and_leaves_out_a_skipped_one(self) -> None:
        text = gtest_xml({("Small_A", "One"): 0.002, ("Rows/Medium_B", "Two/3"): 0.5}, skipped=(("Large_C", "Gpu"),))
        self.assertEqual(parse_gtest_xml(text.encode(), "s.xml"),
                         [Timing("Small_A.One", 0.002), Timing("Rows/Medium_B.Two/3", 0.5)])

    def test_refuses_a_file_with_a_doctype_or_an_entity(self) -> None:
        with self.assertRaises(SystemExit):
            parse_junit(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><testsuite/>', "b.xml")

    def test_finds_the_suite_and_size_of_a_plain_and_a_parametrised_name(self) -> None:
        self.assertEqual(suite_and_size("Small_Shape.Rank"), ("Small_Shape", "small"))
        self.assertEqual(suite_and_size("Rows/Medium_Io.Read/3"), ("Medium_Io", "medium"))
        self.assertEqual(suite_and_size("Benchmark_Wave.Run"), (None, None))

    def test_finds_the_size_of_a_characterization_suite(self) -> None:
        self.assertEqual(suite_and_size("Characterization_Small_Io.Read"), ("Characterization_Small_Io", "small"))

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

    def test_a_large_test_within_its_limit_counts_in_the_shape_and_the_large_total(self) -> None:
        report = run({"Large_M.Scene": 50.0, "Small_A.One": 0.1, "Small_A.Two": 0.1, "Small_A.Three": 0.1},
                     scope="large", wall=51.0)
        self.assertEqual((report.findings, report.rerun), ((), ()))
        self.assertIn("large tier 50 s sum, 51 s wall", report.line)
        self.assertIn("shape 75% small, 0% medium, 25% large", report.line)

    def test_a_large_test_over_its_limit_is_named_for_a_rerun_and_judged_alone(self) -> None:
        first = run({"Large_M.Crush": 61.0}, scope="large")
        self.assertEqual(first.rerun, ("Large_M.Crush",))
        alone = run({"Large_M.Crush": 61.0}, scope="large", rerun=[Timing("Large_M.Crush", 62.0)])
        self.assertEqual(alone.findings, ("Large_M.Crush: 62.0 s alone, over the large limit of 60 s",))

    def test_a_budget_without_every_limit_names_the_missing_ones(self) -> None:
        self.assertEqual(missing_limits({"small_test_s": 1, "pre_merge_s": 300}), ["medium_test_s", "large_test_s"])
        self.assertEqual(missing_limits(K_LIMITS), [])

    def test_a_test_over_its_limit_is_named_for_a_rerun_and_judged_by_the_rerun(self) -> None:
        first = run({"Small_A.Slow": 1.5})
        self.assertEqual(first.rerun, ("Small_A.Slow",))
        self.assertIn("1.5 s, over the small limit of 1 s (not rerun)", first.findings[0])
        quiet = run({"Small_A.Slow": 1.5}, rerun=[Timing("Small_A.Slow", 0.4)])
        self.assertEqual(quiet.findings, ())
        slow = run({"Small_A.Slow": 1.5}, rerun=[Timing("Small_A.Slow", 1.4)])
        self.assertIn("1.4 s alone, over the small limit of 1 s", slow.findings[0])

    def test_a_module_small_total_is_printed_and_never_a_finding(self) -> None:
        report = run(small_timings(2000, 0.05), programs={"Small_A": "core"})
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

    def tearDown(self) -> None:
        self._folder.cleanup()

    def test_reads_the_junit_and_the_program_map_and_gates_the_run(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([str(self.junit), "--programs", str(self.programs), "--gate"], budget=K_LIMITS)
        self.assertEqual((code, out.getvalue().splitlines()), (1, [
            "budget: small 5.0 s (core), whole run not measured, shape 100% small, 0% medium, 0% large",
            "rerun: ^(Small_A\\.Slow)$",
            "Small_A.Slow: 5.0 s, over the small limit of 1 s (not rerun)",
        ]))

    def test_adds_the_per_case_times_of_the_shard_files_and_ignores_the_shard_entry_of_ctest(self) -> None:
        cases = Path(self._folder.name) / "cases"
        cases.mkdir()
        (cases / "shard_1.xml").write_text(gtest_xml({("Small_A", "Slow"): 5.0}), encoding="utf-8")
        (cases / "shard_2.xml").write_text(gtest_xml({("Small_A", "Fast"): 0.01}), encoding="utf-8")
        self.junit.write_text(junit({"core_tests.small.shard_1of2": 5.0}), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([str(self.junit), "--cases", str(cases), "--programs", str(self.programs), "--gate"],
                        budget=K_LIMITS)
        self.assertEqual((code, out.getvalue().splitlines()), (1, [
            "budget: small 5.0 s (core), whole run not measured, shape 100% small, 0% medium, 0% large",
            "rerun: ^(Small_A\\.Slow)$",
            "Small_A.Slow: 5.0 s, over the small limit of 1 s (not rerun)",
        ]))


    def test_refuses_a_budget_without_the_large_limit(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main([str(self.junit)], budget={"small_test_s": 1, "medium_test_s": 10, "pre_merge_s": 300})
        self.assertEqual((code, err.getvalue().strip()), (2, "K_TEST_TIMES in tools/settings.py lacks large_test_s"))

    def test_list_over_writes_the_tests_over_a_limit_and_prints_nothing(self) -> None:
        listed = Path(self._folder.name) / "over.txt"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([str(self.junit), "--gate", "--list-over", str(listed)], budget=K_LIMITS)
        self.assertEqual((code, out.getvalue(), listed.read_text(encoding="utf-8")), (0, "", "Small_A.Slow\n"))

    def test_joins_the_times_of_every_rerun_file(self) -> None:
        first = Path(self._folder.name) / "first.xml"
        second = Path(self._folder.name) / "second.xml"
        self.junit.write_text(junit({"Small_A.Slow": 5.0, "Small_A.Late": 5.0}), encoding="utf-8")
        first.write_text(junit({"Small_A.Slow": 0.2}), encoding="utf-8")
        second.write_text(junit({"Small_A.Late": 3.0}), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([str(self.junit), "--rerun", str(first), "--rerun", str(second), "--gate"], budget=K_LIMITS)
        self.assertEqual((code, out.getvalue().splitlines()[1:]), (1, ["Small_A.Late: 3.0 s alone, over the small limit of 1 s"]))


if __name__ == "__main__":
    unittest.main()
