# test_check_backlog_trailer.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the commit trailer checker.

The judgement of a message and the IDs it may name are checked on values in memory; the history of
a git repository is checked once, in the one medium class of this file. The hook that shares the
rules is tested in test_commit_msg.py.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commits import check_backlog_trailer  # noqa: E402
from commits.check_backlog_trailer import (  # noqa: E402
    K_CHECKER,
    TrailerCheck,
    collect,
    ids_in,
    open_ids,
    rule_trailer,
    trailer_problems,
)

K_AREA = "docs/backlog/GENERAL.md"
K_TABLE = "| ID | Pri |\n| --- | --- |\n| AB-1 | P1 |\n| AB-22 | P2 |\n"
K_KNOWN = frozenset({"AB-1", "AB-22"})
K_SHA = "0123456789abcdef"
K_PATTERN = r"[A-Z]{2}-[0-9]+"


def _check(message: str, known: frozenset[str] = K_KNOWN, id_pattern: str = K_PATTERN) -> TrailerCheck:
    """Returns a commit in memory, with the given message, the IDs open for it and the shape of an ID."""
    return TrailerCheck(Path("repo"), K_SHA, message, known, id_pattern)


class TrailerProblemsTest(unittest.TestCase):
    """Covers the judgement of one message against a set of open entries."""
    K_TEST_SIZE = "small"

    def test_a_message_without_the_line_is_refused(self) -> None:
        self.assertIn("no `Backlog:` line", trailer_problems("fix: a thing\n", K_KNOWN, K_PATTERN)[0])

    def test_an_open_id_passes(self) -> None:
        self.assertEqual(trailer_problems("fix: a thing\n\nBacklog: AB-1\n", K_KNOWN, K_PATTERN), [])

    def test_several_ids_pass_when_each_is_open(self) -> None:
        self.assertEqual(trailer_problems("fix: a thing\n\nBacklog: AB-1, AB-22\n", K_KNOWN, K_PATTERN), [])

    def test_an_id_that_is_not_open_is_refused(self) -> None:
        problems = trailer_problems("fix: a thing\n\nBacklog: AB-9\n", K_KNOWN, K_PATTERN)
        self.assertEqual(len(problems), 1)
        self.assertIn("AB-9", problems[0])

    def test_text_that_is_no_id_is_refused(self) -> None:
        self.assertIn("not a backlog ID", trailer_problems("x\n\nBacklog: later\n", K_KNOWN, K_PATTERN)[0])

    def test_none_needs_a_reason(self) -> None:
        self.assertEqual(trailer_problems("x\n\nBacklog: none (typo in a comment)\n", K_KNOWN, K_PATTERN), [])
        self.assertIn("reason", trailer_problems("x\n\nBacklog: none\n", K_KNOWN, K_PATTERN)[0])
        self.assertIn("reason", trailer_problems("x\n\nBacklog: none ()\n", K_KNOWN, K_PATTERN)[0])

    def test_the_shape_of_an_id_is_the_pattern_given(self) -> None:
        known = frozenset({"TASK-7"})
        self.assertEqual(trailer_problems("x\n\nBacklog: TASK-7\n", known, r"TASK-[0-9]+"), [])
        self.assertIn("not a backlog ID", trailer_problems("x\n\nBacklog: AB-1\n", K_KNOWN, r"TASK-[0-9]+")[0])

    def test_the_pattern_must_match_the_whole_item(self) -> None:
        self.assertIn("not a backlog ID", trailer_problems("x\n\nBacklog: AB-1x\n", K_KNOWN, K_PATTERN)[0])

    def test_the_last_line_counts(self) -> None:
        message = "x\n\nBacklog: AB-9\n\nBacklog: AB-1\n"
        self.assertEqual(trailer_problems(message, K_KNOWN, K_PATTERN), [])


class TrailerRuleTest(unittest.TestCase):
    """Covers the rule over one commit held in memory, with the IDs open for it."""
    K_TEST_SIZE = "small"

    def test_reads_the_rows_of_a_backlog_table(self) -> None:
        self.assertEqual(ids_in(K_TABLE), K_KNOWN)

    def test_an_id_open_at_the_parent_or_in_the_commit_passes(self) -> None:
        self.assertEqual(list(rule_trailer(_check("fix: close AB-1\n\nBacklog: AB-1\n"))), [])

    def test_a_commit_without_the_line_is_reported_on_its_short_hash(self) -> None:
        issues = list(rule_trailer(_check("fix: a thing\n")))
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].path.name, f"commit-{K_SHA[:9]}")

    def test_an_id_closed_before_the_commit_is_refused(self) -> None:
        issues = list(rule_trailer(_check("fix: again\n\nBacklog: AB-9\n", frozenset({"AB-22"}))))
        self.assertEqual(len(issues), 1)

    def test_the_rule_judges_the_id_by_the_pattern_of_the_commit(self) -> None:
        check = _check("fix: a thing\n\nBacklog: TASK-7\n", frozenset({"TASK-7"}), r"TASK-[0-9]+")
        self.assertEqual(list(rule_trailer(check)), [])

    def test_nothing_is_checked_without_a_base(self) -> None:
        with mock.patch.object(check_backlog_trailer, "K_SINCE", ""):
            self.assertEqual(list(collect([Path("unused")])), [])

    def test_registers_one_rule(self) -> None:
        self.assertEqual(list(K_CHECKER.rules), ["rule_trailer"])


class RepositoryTest(unittest.TestCase):
    """Builds a git repository with a backlog for each test and runs the checker on it."""
    K_TEST_SIZE = "medium"

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        self._git("init", "-q")
        self._write(K_AREA, K_TABLE)
        self._commit("chore: start\n\nBacklog: none (the first commit)")
        self.base = self._git("rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        self._folder.cleanup()

    def _git(self, *arguments: str) -> str:
        identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]
        done = subprocess.run(["git", "-C", str(self.root), *identity, *arguments],
                              check=True, capture_output=True, text=True)
        return done.stdout

    def _write(self, name: str, text: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _commit(self, message: str) -> None:
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", message)

    def _findings(self) -> list[str]:
        with mock.patch.object(check_backlog_trailer, "K_SINCE", self.base):
            return [issue.message for check in collect([self.root]) for issue in rule_trailer(check)]

    def test_reads_the_rows_on_disk_and_at_a_revision(self) -> None:
        self.assertEqual(open_ids(self.root, None), K_KNOWN)
        self.assertEqual(open_ids(self.root, "HEAD"), K_KNOWN)

    def test_a_folder_without_a_backlog_holds_no_ids(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(open_ids(Path(empty), None), frozenset())

    def test_a_commit_that_names_an_entry_closed_earlier_is_refused(self) -> None:
        self._write(K_AREA, "| ID | Pri |\n| --- | --- |\n| AB-22 | P2 |\n")
        self._commit("fix: close AB-1\n\nBacklog: AB-1")
        self._commit("fix: again\n\nBacklog: AB-1")
        self.assertEqual(len(self._findings()), 1)

    def test_a_commit_that_closes_its_entry_and_one_that_adds_an_entry_both_pass(self) -> None:
        self._write(K_AREA, "| ID | Pri |\n| --- | --- |\n| AB-22 | P2 |\n")
        self._commit("fix: close AB-1\n\nBacklog: AB-1")
        self._write(K_AREA, "| ID | Pri |\n| --- | --- |\n| AB-22 | P2 |\n| AB-23 | P1 |\n")
        self._commit("fix: add AB-23\n\nBacklog: AB-23")
        self.assertEqual(self._findings(), [])

    def test_a_commit_without_the_line_is_reported_with_its_hash(self) -> None:
        self._commit("fix: a thing")
        with mock.patch.object(check_backlog_trailer, "K_SINCE", self.base):
            check = next(iter(collect([self.root])))
            issue = next(iter(rule_trailer(check)))
        self.assertEqual(len(self._findings()), 1)
        self.assertIn(check.sha[:9], issue.path.name)

    def test_a_history_that_cannot_be_read_stops_the_check(self) -> None:
        with mock.patch.object(check_backlog_trailer, "K_SINCE", "0" * 40):
            with self.assertRaises(SystemExit):
                collect([self.root])


if __name__ == "__main__":
    unittest.main()
