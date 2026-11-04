# test_known_findings.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests how a checker's output is judged against known findings, pure and on disk."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_all  # noqa: E402
from checking import known_findings  # noqa: E402
from checking.known_findings import (  # noqa: E402
    Entry,
    Finding,
    KnownFindingsError,
    apply_baseline,
    count_groups,
    entry_problems,
    orphan_lines,
    parse_entries,
    parse_output,
    read_baseline,
    render_baseline,
    summary,
    write_baseline,
)

K_PATTERN = r"[A-Z]{2}-[0-9]+"
K_CHECKER = "testing/check_test_mirror.py"
K_A = "a.cs:3: rule_missing_test: no test for A"
K_A2 = "a.cs:9: rule_missing_test: no test for B"
K_B = "b.cs:1: rule_missing_test: no test for C"


def entry(path: str = "a.cs", count: int = 2, rule: str = "rule_missing_test",
          backlog: str = "RW-1", checker: str = K_CHECKER) -> Entry:
    """Returns an entry with the fields a test does not set left at their usual value."""
    return Entry(checker, rule, path, count, backlog)


def judge(output: str, entries: list[Entry], code: int = 1,
          open_backlog: frozenset[str] | None = None) -> tuple[int, str]:
    """Returns the verdict of apply_baseline for the test mirror checker."""
    return apply_baseline(K_CHECKER, code, output, entries, K_PATTERN, open_backlog)


class ParseOutputTest(unittest.TestCase):
    """Covers the split of a checker's output into finding lines and other lines."""

    K_TEST_SIZE = "small"

    def test_reads_path_rule_and_keeps_the_line(self) -> None:
        parsed = parse_output(f"{K_A}\n")
        self.assertEqual(parsed.findings, (Finding(K_A, "rule_missing_test", "a.cs"),))
        self.assertEqual(parsed.others, ())

    def test_a_traceback_and_a_usage_line_are_other_lines(self) -> None:
        text = 'Traceback (most recent call last):\n  File "x.py", line 3, in f\nusage: x [-h]\n\n'
        parsed = parse_output(text)
        self.assertEqual(parsed.findings, ())
        self.assertEqual(len(parsed.others), 3)

    def test_a_path_with_folders_and_a_message_with_colons(self) -> None:
        parsed = parse_output("Assets/Core/Foo.cs:12: rule_x: bad: really: 3\n")
        self.assertEqual((parsed.findings[0].path, parsed.findings[0].rule),
                         ("Assets/Core/Foo.cs", "rule_x"))

    def test_an_indented_line_is_no_finding(self) -> None:
        self.assertEqual(parse_output("  a.cs:1: rule_x: m").findings, ())


class CountGroupsTest(unittest.TestCase):
    """Covers the grouping by rule and path."""

    K_TEST_SIZE = "small"

    def test_counts_per_rule_and_path(self) -> None:
        found = parse_output(f"{K_A}\n{K_A2}\n{K_B}\n").findings
        self.assertEqual(count_groups(found),
                         {("rule_missing_test", "a.cs"): 2, ("rule_missing_test", "b.cs"): 1})


class ApplyBaselineTest(unittest.TestCase):
    """Covers every verdict the baseline gives a checker."""

    K_TEST_SIZE = "small"

    def test_no_entry_for_the_checker_returns_it_unchanged(self) -> None:
        other = entry(checker="testing/check_probes.py")
        self.assertEqual(judge(f"{K_A}\n", [other]), (1, f"{K_A}\n"))

    def test_exactly_covered_findings_pass_although_the_checker_exited_one(self) -> None:
        self.assertEqual(judge(f"{K_A}\n{K_A2}\n", [entry(count=2)]), (0, ""))

    def test_more_than_allowed_prints_every_line_of_the_group(self) -> None:
        output = f"{K_A}\n{K_A2}\n{K_B}\n"
        code, text = judge(output, [entry(count=1), entry(path="b.cs", count=1)])
        self.assertEqual((code, text), (1, f"{K_A}\n{K_A2}\n"))

    def test_a_group_with_no_entry_prints_its_lines_and_the_covered_group_stays_quiet(self) -> None:
        self.assertEqual(judge(f"{K_A}\n{K_A2}\n{K_B}\n", [entry(count=2)]), (1, f"{K_B}\n"))

    def test_fewer_than_allowed_says_the_finding_shrank(self) -> None:
        code, text = judge(f"{K_A}\n", [entry(count=3)])
        self.assertEqual((code, text), (1, f"known finding shrank: {K_CHECKER} rule_missing_test a.cs: "
                                           "found 1, allowed 3; lower the count\n"))

    def test_an_entry_found_never_says_gone_even_when_the_checker_exited_zero(self) -> None:
        code, text = judge("", [entry()], code=0)
        self.assertEqual((code, text), (1, f"known finding gone: {K_CHECKER} rule_missing_test a.cs: "
                                           "allowed 2, found 0; remove the entry\n"))

    def test_another_line_makes_nothing_covered(self) -> None:
        output = f"{K_A}\n{K_A2}\nTraceback (most recent call last):\n"
        self.assertEqual(judge(output, [entry(count=2)]), (1, output))

    def test_an_exit_code_of_two_is_never_covered(self) -> None:
        self.assertEqual(judge(f"{K_A}\n{K_A2}\n", [entry(count=2)], code=2), (2, f"{K_A}\n{K_A2}\n"))

    def test_a_backlog_id_that_does_not_match_fails_a_covered_checker(self) -> None:
        code, text = judge(f"{K_A}\n{K_A2}\n", [entry(backlog="oops")])
        self.assertEqual(code, 1)
        self.assertIn("known finding backlog invalid", text)

    def test_a_bad_backlog_id_fails_a_checker_that_exited_zero_with_other_lines(self) -> None:
        code, text = judge("note\n", [entry(backlog="oops")], code=0)
        self.assertEqual(code, 1)
        self.assertTrue(text.startswith("note\nknown finding backlog invalid"))


class EntryProblemsTest(unittest.TestCase):
    """Covers the check of an entry's backlog ID against the pattern and the open IDs."""

    K_TEST_SIZE = "small"

    def test_a_matching_id_with_no_open_list_passes(self) -> None:
        self.assertEqual(entry_problems(entry(), K_PATTERN, None), [])

    def test_an_id_outside_the_open_list_is_closed(self) -> None:
        problems = entry_problems(entry(), K_PATTERN, frozenset({"RW-2"}))
        self.assertEqual(len(problems), 1)
        self.assertIn("is not an open backlog ID", problems[0])

    def test_an_open_id_passes(self) -> None:
        self.assertEqual(entry_problems(entry(), K_PATTERN, frozenset({"RW-1"})), [])

    def test_the_pattern_must_match_the_whole_id(self) -> None:
        self.assertEqual(len(entry_problems(entry(backlog="RW-1x"), K_PATTERN, None)), 1)


class SummaryTest(unittest.TestCase):
    """Covers the tail of the clean line and the entries no checker owns."""

    K_TEST_SIZE = "small"

    def test_no_entries_add_nothing(self) -> None:
        self.assertEqual(summary([]), "")

    def test_counts_findings_and_entries(self) -> None:
        self.assertEqual(summary([entry(count=3), entry(path="b", count=2)]),
                         "; 5 known findings in 2 entries")

    def test_one_of_each_is_singular(self) -> None:
        self.assertEqual(summary([entry(count=1)]), "; 1 known finding in 1 entry")

    def test_names_the_entry_whose_checker_is_not_planned(self) -> None:
        self.assertEqual(orphan_lines([entry(), entry(checker="x/check_y.py")], [K_CHECKER]),
                         ["known finding names no checker: x/check_y.py rule_missing_test a.cs"])


class ParseEntriesTest(unittest.TestCase):
    """Covers the reading of a parsed baseline table."""

    K_TEST_SIZE = "small"

    K_TABLE = {"checker": K_CHECKER, "rule": "r", "path": "p", "count": 3, "backlog": "RW-1"}

    def test_reads_each_field(self) -> None:
        self.assertEqual(parse_entries({"finding": [self.K_TABLE]}), [Entry(K_CHECKER, "r", "p", 3, "RW-1")])

    def test_no_table_is_no_entry(self) -> None:
        self.assertEqual(parse_entries({}), [])

    def test_a_missing_field_is_an_error(self) -> None:
        with self.assertRaises(KnownFindingsError):
            parse_entries({"finding": [{**self.K_TABLE, "backlog": ""}]})

    def test_a_count_below_one_or_not_an_integer_is_an_error(self) -> None:
        for count in (0, "3", True):
            with self.subTest(count=count), self.assertRaises(KnownFindingsError):
                parse_entries({"finding": [{**self.K_TABLE, "count": count}]})

    def test_a_table_that_is_no_table_is_an_error(self) -> None:
        with self.assertRaises(KnownFindingsError):
            parse_entries({"finding": ["x"]})

    def test_finding_must_be_a_list(self) -> None:
        with self.assertRaises(KnownFindingsError):
            parse_entries({"finding": {"a": 1}})

    def test_a_repeated_group_is_an_error(self) -> None:
        with self.assertRaises(KnownFindingsError):
            parse_entries({"finding": [self.K_TABLE, self.K_TABLE]})


class RenderBaselineTest(unittest.TestCase):
    """Covers the text the writer produces."""

    K_TEST_SIZE = "small"

    def test_one_sorted_entry_per_group_that_reads_back(self) -> None:
        found = {"z/check_z.py": parse_output(f"{K_B}\n").findings,
                 K_CHECKER: parse_output(f"{K_B}\n{K_A}\n{K_A2}\n").findings}
        text = render_baseline(found, "RW-9")
        entries = parse_entries(tomllib.loads(text))
        self.assertEqual([(e.checker, e.path, e.count, e.backlog) for e in entries],
                         [(K_CHECKER, "a.cs", 2, "RW-9"), (K_CHECKER, "b.cs", 1, "RW-9"),
                          ("z/check_z.py", "b.cs", 1, "RW-9")])

    def test_quotes_and_backslashes_in_a_path_survive(self) -> None:
        findings = (Finding("x", "r", 'we"ird\\path'),)
        entries = parse_entries(tomllib.loads(render_baseline({K_CHECKER: findings}, "RW-1")))
        self.assertEqual(entries[0].path, 'we"ird\\path')

    def test_a_non_ascii_path_is_written_as_it_is(self) -> None:
        text = render_baseline({K_CHECKER: (Finding("x", "r", "Varlık/ü.cs"),)}, "RW-1")
        self.assertIn('path = "Varlık/ü.cs"', text)

    def test_no_findings_render_empty(self) -> None:
        self.assertEqual(render_baseline({}, "RW-1"), "")


class ShellTest(unittest.TestCase):
    """Covers the baseline file on disk and the writer's command."""

    K_TEST_SIZE = "medium"

    def test_an_absent_file_is_an_empty_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(read_baseline(Path(folder)), [])

    def test_write_refuses_an_existing_file_unless_replace_and_writes_lf(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tools").mkdir()
            path = write_baseline(root, "a\nb\n", False)
            self.assertEqual(path.read_bytes(), b"a\nb\n")
            with self.assertRaises(KnownFindingsError):
                write_baseline(root, "c\n", False)
            write_baseline(root, "c\n", True)
            self.assertEqual(path.read_bytes(), b"c\n")

    def test_a_file_that_is_not_toml_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tools").mkdir()
            (root / known_findings.K_BASELINE_FILE).write_text("[[finding", encoding="utf-8")
            with self.assertRaises(KnownFindingsError):
                read_baseline(root)

    def test_the_writer_writes_what_the_checkers_print_and_reads_back(self) -> None:
        checks = [check_all.Check(K_CHECKER, (), False), check_all.Check("tests", (), True)]
        results = [(1, f"{K_A}\n{K_A2}\n"), (1, "FAIL: suite\n")]
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(check_all, "run_plan", return_value=(checks, results)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(folder)
            (root / "tools").mkdir()
            command = [str(root), "--write", "--backlog", "RW-1"]
            self.assertEqual(known_findings.main(command), 0)
            self.assertEqual(read_baseline(root), [entry(count=2)])
            self.assertEqual(known_findings.main(command), 2)
            self.assertEqual(known_findings.main([*command, "--replace"]), 0)

    def test_the_writer_reports_a_checker_that_prints_more_than_findings(self) -> None:
        checks = [check_all.Check(K_CHECKER, (), False)]
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(check_all, "run_plan", return_value=(checks, [(1, "Traceback\n")])), \
                contextlib.redirect_stdout(out):
            (Path(folder) / "tools").mkdir()
            self.assertEqual(known_findings.main([folder, "--write", "--backlog", "RW-1"]), 1)
        self.assertIn(f"not covered, prints more than findings: {K_CHECKER}", out.getvalue())

    def test_the_writer_reports_a_note_after_a_zero_exit_and_a_silent_failure(self) -> None:
        checks = [check_all.Check("a/check_a.py", (), False), check_all.Check("b/check_b.py", (), False)]
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(check_all, "run_plan",
                                  return_value=(checks, [(0, "a note\n"), (1, "")])), \
                contextlib.redirect_stdout(out):
            (Path(folder) / "tools").mkdir()
            known_findings.main([folder, "--write", "--backlog", "RW-1"])
        self.assertEqual(out.getvalue().count("not covered"), 2)

    def test_the_command_needs_both_write_and_backlog(self) -> None:
        for command in (["x", "--backlog", "RW-1"], ["x", "--write"]):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()),                     self.assertRaises(SystemExit):
                known_findings.main(command)

    def test_the_writer_rejects_a_backlog_id_that_does_not_match(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(known_findings.main(["x", "--write", "--backlog", "nope"]), 2)


if __name__ == "__main__":
    unittest.main()
