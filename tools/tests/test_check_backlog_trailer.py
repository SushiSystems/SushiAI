# test_check_backlog_trailer.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the commit trailer checker, the commit-msg hook that shares its rules and the hook installer."""

from __future__ import annotations

import contextlib
import io
import os
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
    collect,
    open_ids,
    rule_trailer,
    trailer_problems,
)
from documentation import docs_follow  # noqa: E402
from hooks import commit_msg, install_hooks  # noqa: E402

K_AREA = "docs/backlog/GENERAL.md"
K_TABLE = "| ID | Pri |\n| --- | --- |\n| AB-1 | P1 |\n| AB-22 | P2 |\n"
K_KNOWN = frozenset({"AB-1", "AB-22"})


class TrailerProblemsTest(unittest.TestCase):
    """Covers the judgement of one message against a set of open entries."""

    def test_a_message_without_the_line_is_refused(self) -> None:
        self.assertIn("no `Backlog:` line", trailer_problems("fix: a thing\n", K_KNOWN)[0])

    def test_an_open_id_passes(self) -> None:
        self.assertEqual(trailer_problems("fix: a thing\n\nBacklog: AB-1\n", K_KNOWN), [])

    def test_several_ids_pass_when_each_is_open(self) -> None:
        self.assertEqual(trailer_problems("fix: a thing\n\nBacklog: AB-1, AB-22\n", K_KNOWN), [])

    def test_an_id_that_is_not_open_is_refused(self) -> None:
        problems = trailer_problems("fix: a thing\n\nBacklog: AB-9\n", K_KNOWN)
        self.assertEqual(len(problems), 1)
        self.assertIn("AB-9", problems[0])

    def test_text_that_is_no_id_is_refused(self) -> None:
        self.assertIn("not a backlog ID", trailer_problems("x\n\nBacklog: later\n", K_KNOWN)[0])

    def test_none_needs_a_reason(self) -> None:
        self.assertEqual(trailer_problems("x\n\nBacklog: none (typo in a comment)\n", K_KNOWN), [])
        self.assertIn("reason", trailer_problems("x\n\nBacklog: none\n", K_KNOWN)[0])
        self.assertIn("reason", trailer_problems("x\n\nBacklog: none ()\n", K_KNOWN)[0])

    def test_the_last_line_counts(self) -> None:
        message = "x\n\nBacklog: AB-9\n\nBacklog: AB-1\n"
        self.assertEqual(trailer_problems(message, K_KNOWN), [])


class RepositoryTest(unittest.TestCase):
    """Builds a git repository with a backlog for each test."""

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
            return [issue.message for commit in collect([self.root]) for issue in rule_trailer(commit)]


class OpenIdsTest(RepositoryTest):
    """Covers reading the open entries from disk and from a revision."""

    def test_reads_the_rows_on_disk_and_at_a_revision(self) -> None:
        self.assertEqual(open_ids(self.root, None), K_KNOWN)
        self.assertEqual(open_ids(self.root, "HEAD"), K_KNOWN)

    def test_a_folder_without_a_backlog_holds_no_ids(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(open_ids(Path(empty), None), frozenset())


class CommitsTest(RepositoryTest):
    """Covers the commits after the base."""

    def test_nothing_is_checked_without_a_base(self) -> None:
        self._commit("fix: no line at all")
        with mock.patch.object(check_backlog_trailer, "K_SINCE", ""):
            self.assertEqual(list(collect([self.root])), [])

    def test_a_commit_with_an_open_id_passes(self) -> None:
        self._commit("fix: a thing\n\nBacklog: AB-1")
        self.assertEqual(self._findings(), [])

    def test_a_commit_without_the_line_is_reported_with_its_hash(self) -> None:
        self._commit("fix: a thing")
        findings = self._findings()
        self.assertEqual(len(findings), 1)
        with mock.patch.object(check_backlog_trailer, "K_SINCE", self.base):
            commit = next(iter(collect([self.root])))
            issue = next(iter(rule_trailer(commit)))
        self.assertIn(commit.sha[:9], issue.path.name)

    def test_a_commit_that_closes_its_entry_still_names_it(self) -> None:
        self._write(K_AREA, "| ID | Pri |\n| --- | --- |\n| AB-22 | P2 |\n")
        self._commit("fix: close AB-1\n\nBacklog: AB-1")
        self.assertEqual(self._findings(), [])

    def test_a_commit_that_names_an_entry_closed_earlier_is_refused(self) -> None:
        self._write(K_AREA, "| ID | Pri |\n| --- | --- |\n| AB-22 | P2 |\n")
        self._commit("fix: close AB-1\n\nBacklog: AB-1")
        self._commit("fix: again\n\nBacklog: AB-1")
        self.assertEqual(len(self._findings()), 1)

    def test_a_commit_that_adds_and_names_a_new_entry_passes(self) -> None:
        self._write(K_AREA, K_TABLE + "| AB-23 | P1 |\n")
        self._commit("fix: add AB-23\n\nBacklog: AB-23")
        self.assertEqual(self._findings(), [])

    def test_a_history_that_cannot_be_read_stops_the_check(self) -> None:
        with mock.patch.object(check_backlog_trailer, "K_SINCE", "0" * 40):
            with self.assertRaises(SystemExit):
                collect([self.root])

    def test_registers_one_rule(self) -> None:
        self.assertEqual(list(K_CHECKER.rules), ["rule_trailer"])


class HookTest(RepositoryTest):
    """Covers the commit-msg hook against the repository it runs in."""

    def _run(self, message: str) -> tuple[int, str]:
        path = self.root / "MESSAGE"
        path.write_text(message, encoding="utf-8")
        errors = io.StringIO()
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with (mock.patch.object(check_backlog_trailer, "K_SINCE", self.base),
                  mock.patch.object(docs_follow, "K_SINCE", ""), contextlib.redirect_stderr(errors)):
                code = commit_msg.main([str(path)])
        finally:
            os.chdir(previous)
        return code, errors.getvalue()

    def test_refuses_a_message_without_the_line(self) -> None:
        code, errors = self._run("fix: a thing\n")
        self.assertEqual(code, 1)
        self.assertIn("commit refused", errors)

    def test_accepts_an_open_id_and_ignores_git_comment_lines(self) -> None:
        self.assertEqual(self._run("fix: a thing\n\nBacklog: AB-1\n# Please enter the message\n")[0], 0)

    def test_accepts_an_entry_added_but_not_yet_committed(self) -> None:
        self._write(K_AREA, K_TABLE + "| AB-30 | P1 |\n")
        self.assertEqual(self._run("fix: a thing\n\nBacklog: AB-30\n")[0], 0)

    def test_leaves_a_merge_alone(self) -> None:
        self.assertEqual(self._run("Merge branch 'x'\n")[0], 0)

    def test_checks_nothing_without_a_base(self) -> None:
        path = self.root / "MESSAGE"
        path.write_text("fix: a thing\n", encoding="utf-8")
        with (mock.patch.object(check_backlog_trailer, "K_SINCE", ""),
              mock.patch.object(docs_follow, "K_SINCE", "")):
            self.assertEqual(commit_msg.main([str(path)]), 0)

    def test_wants_exactly_one_argument(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(commit_msg.main([]), 1)


class InstallTest(RepositoryTest):
    """Covers pointing git at the tracked hooks."""

    def test_sets_the_hooks_path(self) -> None:
        self.assertTrue(install_hooks.install(self.root))
        self.assertEqual(self._git("config", "core.hooksPath").strip(), "tools/hooks")

    def test_reports_a_folder_that_is_no_repository(self) -> None:
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(install_hooks.main([folder]), 1)


if __name__ == "__main__":
    unittest.main()
