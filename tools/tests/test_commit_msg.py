# test_commit_msg.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the commit-msg hook: its decision on values in memory, and the hook itself once on a real repository."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from commits import check_backlog_trailer  # noqa: E402
from documentation import docs_follow  # noqa: E402
from documentation.docs_follow import Change, Follow, PageFacts  # noqa: E402
from hooks import commit_msg  # noqa: E402
from hooks.commit_msg import commit_problems, is_merge, refusal_lines, strip_comments  # noqa: E402
from repo_builder import Repo  # noqa: E402

K_PATTERN = r"[A-Z]{2}-[0-9]+"
K_KNOWN = frozenset({"AB-1", "AB-22"})
K_GOOD = "fix: a thing\n\nBacklog: AB-1\n"
K_PAGE = "# Guide\n\n**Describes:** `src/a.py`\n\nText.\n"


def _follow(message: str, page_changed: bool = False) -> Follow:
    """Returns a staged change of src/a.py, with the page that describes it staged too when asked."""
    changes = [Change("M", "src/a.py")]
    if page_changed:
        changes.append(Change("M", "docs/GUIDE.md"))
    return Follow(
        changes=tuple(changes),
        message=message,
        ignored=(),
        pages=(docs_follow.page_facts("docs/GUIDE.md", K_PAGE),),
        versions=(),
        tree_before=frozenset({"src/a.py", "docs/GUIDE.md"}),
        tree_after=frozenset({"src/a.py", "docs/GUIDE.md"}),
    )


class MessageTest(unittest.TestCase):
    """Covers reading the text of a message file."""

    K_TEST_SIZE = "small"

    def test_git_comment_lines_are_removed(self) -> None:
        self.assertEqual(strip_comments("fix: a\n# comment\n\nBacklog: AB-1\n#last\n"), "fix: a\n\nBacklog: AB-1")

    def test_a_hash_inside_a_line_stays(self) -> None:
        self.assertEqual(strip_comments("fix: a # b\n"), "fix: a # b")

    def test_a_message_that_opens_with_merge_is_a_merge(self) -> None:
        self.assertTrue(is_merge("Merge branch 'x'"))
        self.assertFalse(is_merge("fix: merge two lists"))
        self.assertFalse(is_merge("Merged"))


class UsageTest(unittest.TestCase):
    """Covers the arguments the hook takes."""

    K_TEST_SIZE = "small"

    def test_wants_exactly_one_argument(self) -> None:
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(commit_msg.main([]), 1)
            self.assertEqual(commit_msg.main(["a", "b"]), 1)
        self.assertEqual(errors.getvalue(), "usage: commit_msg.py MESSAGE_FILE\n" * 2)


class CommitProblemsTest(unittest.TestCase):
    """Covers the decision on a message, the IDs open for it and the staged change."""

    K_TEST_SIZE = "small"

    def test_a_commit_with_an_open_id_and_nothing_staged_that_a_page_describes_passes(self) -> None:
        self.assertEqual(commit_problems(K_GOOD, K_KNOWN, K_PATTERN, None), [])

    def test_a_message_without_the_line_is_refused(self) -> None:
        problems = commit_problems("fix: a thing\n", K_KNOWN, K_PATTERN, None)
        self.assertEqual(len(problems), 1)
        self.assertIn("no `Backlog:` line", problems[0])

    def test_an_id_that_is_not_open_is_refused(self) -> None:
        problems = commit_problems("fix: a\n\nBacklog: AB-9\n", K_KNOWN, K_PATTERN, None)
        self.assertEqual(len(problems), 1)
        self.assertIn("AB-9", problems[0])

    def test_the_shape_of_an_id_is_the_pattern_given(self) -> None:
        self.assertEqual(commit_problems("x\n\nBacklog: TASK-7\n", frozenset({"TASK-7"}), r"TASK-[0-9]+", None), [])
        self.assertIn("not a backlog ID", commit_problems(K_GOOD, K_KNOWN, r"TASK-[0-9]+", None)[0])

    def test_nothing_is_required_of_the_line_when_no_base_is_set(self) -> None:
        self.assertEqual(commit_problems("fix: a thing\n", None, K_PATTERN, None), [])

    def test_a_staged_change_without_its_page_is_refused(self) -> None:
        problems = commit_problems(K_GOOD, K_KNOWN, K_PATTERN, _follow(K_GOOD))
        self.assertEqual(len(problems), 1)
        self.assertIn("docs/GUIDE.md describes src/a.py", problems[0])

    def test_the_page_staged_with_the_change_passes(self) -> None:
        self.assertEqual(commit_problems(K_GOOD, K_KNOWN, K_PATTERN, _follow(K_GOOD, page_changed=True)), [])

    def test_a_docs_line_waives_the_page(self) -> None:
        message = K_GOOD + "Docs: docs/GUIDE.md unchanged (the change is internal)\n"
        self.assertEqual(commit_problems(message, K_KNOWN, K_PATTERN, _follow(message)), [])

    def test_the_page_is_judged_even_when_no_line_is_required(self) -> None:
        problems = commit_problems("fix: a\n", None, K_PATTERN, _follow("fix: a\n"))
        self.assertEqual(len(problems), 1)
        self.assertIn("docs/GUIDE.md describes", problems[0])

    def test_the_faults_of_the_line_come_before_the_faults_of_the_pages(self) -> None:
        problems = commit_problems("fix: a\n", K_KNOWN, K_PATTERN, _follow("fix: a\n"))
        self.assertEqual(len(problems), 2)
        self.assertIn("no `Backlog:` line", problems[0])
        self.assertIn("docs/GUIDE.md describes", problems[1])

    def test_each_problem_is_printed_on_a_line_of_its_own(self) -> None:
        self.assertEqual(refusal_lines(["one", "two"]), ["commit refused: one", "commit refused: two"])
        self.assertEqual(refusal_lines([]), [])


class HookTest(unittest.TestCase):
    """Runs the hook on a real repository: the index, the backlog on disk and at HEAD, and the exit code."""

    K_TEST_SIZE = "medium"

    def setUp(self) -> None:
        self.repo = Repo()
        self.addCleanup(self.repo.close)
        self.repo.write("docs/backlog/GENERAL.md", "| ID | Pri |\n| --- | --- |\n| AB-1 | P1 |\n")
        self.repo.write("docs/GUIDE.md", K_PAGE)
        self.repo.write("src/a.py", "x = 1\n")
        self.base = self.repo.commit("first").sha

    def _run(self, message: str) -> tuple[int, str]:
        path = self.repo.root / "MESSAGE"
        path.write_text(message, encoding="utf-8")
        errors = io.StringIO()
        previous = Path.cwd()
        os.chdir(self.repo.root)
        try:
            with (mock.patch.object(check_backlog_trailer, "K_SINCE", self.base),
                  mock.patch.object(docs_follow, "K_SINCE", self.base),
                  contextlib.redirect_stderr(errors)):
                code = commit_msg.main([str(path)])
        finally:
            os.chdir(previous)
        return code, errors.getvalue()

    def test_refuses_and_accepts_by_what_is_staged_and_what_the_backlog_holds(self) -> None:
        self.repo.write("src/a.py", "x = 2\n")
        self.repo.git("add", "-A")
        code, errors = self._run("fix: change a\n\nBacklog: AB-1\n")
        self.assertEqual(code, 1)
        self.assertIn("commit refused: docs/GUIDE.md describes src/a.py", errors)
        self.repo.write("docs/GUIDE.md", K_PAGE + "More.\n")
        self.repo.write("docs/backlog/GENERAL.md", "| ID | Pri |\n| --- | --- |\n| AB-1 | P1 |\n| AB-30 | P1 |\n")
        self.repo.git("add", "-A")
        self.assertEqual(self._run("fix: change a\n\nBacklog: AB-30\n# Please enter the message\n"), (0, ""))
        self.assertEqual(self._run("fix: change a\n")[0], 1)
        self.assertEqual(self._run("Merge branch 'x'\n"), (0, ""))


if __name__ == "__main__":
    unittest.main()
