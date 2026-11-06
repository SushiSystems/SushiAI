# test_check_all.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests which checks tools/check_all.py plans and how it reports them, pure and on disk."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import tomllib
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_all  # noqa: E402
from checking.known_findings import parse_entries  # noqa: E402
from check_all import (  # noqa: E402
    K_CHECKERS,
    Check,
    _read_found,
    _read_suite_folders,
    _run_all,
    _judged,
    _run_check,
    own_checker_names,
    plan_checks,
    report,
)

K_BASELINE_TEXT = (
    "[[finding]]\n"
    'checker = "documentation/check_source_comments.py"\n'
    'rule = "rule_x"\n'
    'path = "a.cs"\n'
    "count = 1\n"
    'backlog = "RW-1"\n'
)
K_ROOT = Path("repo")
K_TOOLS = K_ROOT / "tools"


def first_check_fails(output: str) -> Callable[[list[Check]], list[tuple[int, str]]]:
    """Returns a runner whose first check really fails with *output* and whose others pass."""
    def run(checks: list[Check]) -> list[tuple[int, str]]:
        failure = _run_check([sys.executable, "-c", f"print({output!r}); raise SystemExit(1)"])
        return [failure, *[(0, "")] * (len(checks) - 1)]
    return run


def every_check_passes(checks: list[Check]) -> list[tuple[int, str]]:
    """Returns a passing result for each check, without starting a process."""
    return [(0, "")] * len(checks)


class OwnCheckerNamesTest(unittest.TestCase):
    """Covers which of the found paths are own checkers that the shared list does not name."""

    K_TEST_SIZE = "small"

    def test_names_a_checker_the_shared_list_does_not_name(self) -> None:
        found = [*K_CHECKERS, "layering/check_package_layers.py"]
        self.assertEqual(own_checker_names(found), ("layering/check_package_layers.py",))

    def test_drops_a_test_helper_and_a_checker_under_tests(self) -> None:
        found = ["layering/helper.py", "tests/check_fake.py", "layering/check_package_layers.py"]
        self.assertEqual(own_checker_names(found), ("layering/check_package_layers.py",))


class PlanChecksTest(unittest.TestCase):
    """Covers the order of the checks, their commands and the names of the test suites."""

    K_TEST_SIZE = "small"

    def test_shared_checkers_run_first_then_own_then_suites(self) -> None:
        plan = plan_checks(K_ROOT, K_TOOLS, "py", ["layering/check_x.py"], [K_TOOLS / "tests"])
        names = [check.name for check in plan]
        self.assertEqual(names, [*K_CHECKERS, "layering/check_x.py", "tests"])

    def test_each_checker_runs_with_the_interpreter_and_the_root(self) -> None:
        plan = plan_checks(K_ROOT, K_TOOLS, "py", [], [])
        self.assertEqual(plan[0].command, ("py", str(K_TOOLS / K_CHECKERS[0]), str(K_ROOT)))

    def test_suite_runs_unittest_discover_on_its_folder(self) -> None:
        plan = plan_checks(K_ROOT, K_TOOLS, "py", [], [K_ROOT / "gui" / "tests"])
        self.assertEqual(plan[-1], Check("tests: gui/tests",
                                         ("py", "-m", "unittest", "discover", "-q", "-s",
                                          str(K_ROOT / "gui" / "tests")), True))


class ReportTest(unittest.TestCase):
    """Covers the lines and the exit code for a clean run, a failed run and several failures."""

    K_TEST_SIZE = "small"

    def test_clean_run_counts_checkers_not_suites(self) -> None:
        checks = [Check("a", (), False), Check("b", (), False), Check("tests", (), True)]
        lines, _ = report(checks, [(0, ""), (0, ""), (0, "")])
        self.assertEqual(lines, ["clean: 2 checkers, tests"])

    def test_clean_run_exits_zero(self) -> None:
        checks = [Check("a", (), False), Check("tests", (), True)]
        _, code = report(checks, [(0, ""), (0, "")])
        self.assertEqual(code, 0)

    def test_failed_check_prints_its_trimmed_output_and_is_named(self) -> None:
        checks = [Check("a", (), False), Check("b", (), False)]
        self.assertEqual(report(checks, [(0, "ok\n"), (1, "finding\n\n")]),
                         (["== b\nfinding", "failed: b"], 1))

    def test_several_failures_keep_the_order_of_the_checks(self) -> None:
        checks = [Check("a", (), False), Check("b", (), False), Check("c", (), False)]
        lines, code = report(checks, [(2, "one"), (0, ""), (1, "three")])
        self.assertEqual((lines, code), (["== a\none", "== c\nthree", "failed: a, c"], 1))

    def test_known_findings_end_the_clean_line(self) -> None:
        checks = [Check("a", (), False), Check("tests", (), True)]
        lines, code = report(checks, [(0, ""), (0, "")], "; 3 known findings in 1 entry")
        self.assertEqual((lines, code), (["clean: 1 checkers, tests; 3 known findings in 1 entry"], 0))

    def test_orphans_fail_under_the_baseline_file_name(self) -> None:
        checks = [Check("a", (), False)]
        lines, code = report(checks, [(0, "")], "; 1 known finding in 1 entry", ["names no checker"])
        self.assertEqual((lines, code), (["== tools/known_findings.toml\nnames no checker",
                                          "failed: tools/known_findings.toml"], 1))

    def test_orphans_follow_the_failed_checks(self) -> None:
        checks = [Check("a", (), False)]
        lines, _ = report(checks, [(1, "bad")], "", ["orphan"])
        self.assertEqual(lines[-1], "failed: a, tools/known_findings.toml")


class JudgedTest(unittest.TestCase):
    """Covers which results the baseline may change."""

    K_TEST_SIZE = "small"

    def test_no_entries_return_the_results_as_they_are(self) -> None:
        checks = [Check("a", (), False)]
        self.assertEqual(_judged(K_ROOT, checks, [(1, "x")], []), [(1, "x")])


class EnvironmentTest(unittest.TestCase):
    """Covers which variables a test suite's environment drops."""

    K_TEST_SIZE = "small"

    def test_drops_every_variable_that_points_git_at_a_repository(self) -> None:
        names = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_PREFIX",
                 "GIT_NAMESPACE")
        environment = {"PATH": "/bin", **{name: "x" for name in names}}
        self.assertEqual(check_all.suite_environment(environment), {"PATH": "/bin"})


class ShellTest(unittest.TestCase):
    """Covers the disk and process edges: own checkers found on disk, and a check's output."""

    K_TEST_SIZE = "medium"

    def test_finds_a_checker_the_shared_list_does_not_name(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tools = Path(folder)
            for name in (*K_CHECKERS, "layering/check_package_layers.py", "tests/check_fake.py",
                         "layering/helper.py"):
                (tools / name).parent.mkdir(parents=True, exist_ok=True)
                (tools / name).write_text("\n", encoding="utf-8")
            own = own_checker_names(_read_found(tools))
            self.assertEqual(own, ("layering/check_package_layers.py",))


    def test_runs_each_command_and_returns_its_exit_code_in_order(self) -> None:
        checks = [Check("passes", (sys.executable, "-c", "raise SystemExit(0)"), False),
                  Check("fails", (sys.executable, "-c", "raise SystemExit(1)"), False)]
        self.assertEqual([code for code, _ in _run_all(checks)], [0, 1])

    def test_lists_only_the_test_folders_that_exist(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "present").mkdir()
            with mock.patch.object(check_all, "K_EXTRA_TEST_FOLDERS", ("present", "absent")):
                folders = _read_suite_folders(root)
            self.assertEqual(folders, [check_all.K_TOOLS / "tests", root / "present"])

    def test_main_returns_one_when_a_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_all.main([folder], first_check_fails("finding")), 1)

    def test_a_baseline_makes_a_red_checker_pass(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            self.write_baseline(folder, K_BASELINE_TEXT)
            with contextlib.redirect_stdout(io.StringIO()):
                code = check_all.main([folder], first_check_fails("a.cs:1: rule_x: bad"))
        self.assertEqual(code, 0)

    def test_the_clean_line_counts_the_known_findings(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            self.write_baseline(folder, K_BASELINE_TEXT)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                check_all.main([folder], first_check_fails("a.cs:1: rule_x: bad"))
        self.assertEqual(out.getvalue().splitlines()[-1].split(";", 1)[1],
                         " 1 known finding in 1 entry")

    def test_a_baseline_entry_for_no_checker_fails(self) -> None:
        text = K_BASELINE_TEXT.replace("documentation/check_source_comments.py", "nope/check_x.py")
        with tempfile.TemporaryDirectory() as folder:
            self.write_baseline(folder, text)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = check_all.main([folder], every_check_passes)
        self.assertEqual((code, "known finding names no checker" in out.getvalue()), (1, True))

    @staticmethod
    def write_baseline(folder: str, text: str) -> None:
        """Writes *text* as the baseline file of the repository at *folder*."""
        (Path(folder) / "tools").mkdir()
        (Path(folder) / "tools" / "known_findings.toml").write_text(text, encoding="utf-8")

    def test_a_broken_baseline_file_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "tools").mkdir()
            (Path(folder) / "tools" / "known_findings.toml").write_text("[[finding", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check_all.main([folder]), 1)

    def test_a_closed_backlog_id_fails_when_the_root_lists_its_open_ones(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs" / "backlog").mkdir(parents=True)
            (root / "docs" / "backlog" / "OPEN.md").write_text("| RW-2 | P1 | x |\n", encoding="utf-8")
            checks = [Check("documentation/check_source_comments.py", (), False)]
            entries = parse_entries(tomllib.loads(K_BASELINE_TEXT))
            judged = _judged(root, checks, [(1, "a.cs:1: rule_x: bad\n")], entries)
        self.assertEqual(judged[0][0], 1)
        self.assertIn("is not an open backlog ID", judged[0][1])

    def test_reads_a_non_ascii_finding(self) -> None:
        """Reads an em dash a checker prints, whatever the console's code page."""
        script = "print('Open — phase 1'); raise SystemExit(1)"
        code, output = _run_check([sys.executable, "-c", script])
        self.assertEqual((code, output.strip()), (1, "Open — phase 1"))


if __name__ == "__main__":
    unittest.main()
