# test_docs_follow.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the rule that a change updates the pages describing it, in the checker and in the commit hook."""

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
from commits.history import commits_since  # noqa: E402
from documentation import check_docs_follow, docs_follow  # noqa: E402
from documentation.docs_follow import Change, acknowledged, describes_of, parse_changes  # noqa: E402
from hooks import commit_msg  # noqa: E402

K_PAGE = "# Guide\n\n**Describes:** `src/a.py`, `core`\n\nText.\n"


class ParsingTest(unittest.TestCase):
    """Covers reading the describes line, the status listing and the acknowledgements."""

    def test_the_describes_line_lists_its_code_spans(self) -> None:
        self.assertEqual(describes_of("# T\n\n**Describes:** `a/b.py`, `c/`\n"), ("a/b.py", "c"))

    def test_a_describes_line_below_the_head_of_the_page_is_ignored(self) -> None:
        self.assertEqual(describes_of("# T\n" + "x\n" * 20 + "**Describes:** `a`\n"), ())

    def test_the_status_listing_names_renames_with_both_paths(self) -> None:
        listing = "M\ta.py\nA\tb.py\nD\tc.py\nR100\told.py\tnew.py\n"
        self.assertEqual(parse_changes(listing), [Change("M", "a.py"), Change("A", "b.py"), Change("D", "c.py"),
                                                  Change("R", "new.py", "old.py")])

    def test_a_page_is_acknowledged_with_a_reason_only(self) -> None:
        message = "x\n\nDocs: docs/A.md unchanged (wording only)\nDocs: docs/B.md unchanged ()\n"
        self.assertEqual(acknowledged(message), (False, {"docs/A.md"}))

    def test_none_with_a_reason_waives_every_page(self) -> None:
        self.assertEqual(acknowledged("x\n\nDocs: none (a comment fix)\n")[0], True)
        self.assertEqual(acknowledged("x\n\nDocs: none\n")[0], False)


class RepositoryTest(unittest.TestCase):
    """Builds a git repository with a guide, a library folder and a README for each test."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)
        self._git("init", "-q")
        self._write("docs/GUIDE.md", K_PAGE)
        self._write("src/a.py", "x = 1\n")
        self._write("lib/README.md", "# Lib\n")
        self._write("lib/old.py", "y = 1\n")
        self._write("lib/sub/inner.py", "z = 1\n")
        self._write("core/deep/x.py", "c = 1\n")
        self._commit("chore: start")
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

    def _problems(self) -> list[str]:
        """Returns the problems of the last commit made after the base."""
        with mock.patch.object(docs_follow, "K_SINCE", self.base):
            return [issue.message for commit in check_docs_follow.collect([self.root])
                    for issue in check_docs_follow.rule_pages_follow(commit)]


class ExplicitTest(RepositoryTest):
    """Covers a page that names its files."""

    def test_a_changed_described_file_without_its_page_is_a_problem(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._commit("fix: change a")
        problems = self._problems()
        self.assertEqual(len(problems), 1)
        self.assertIn("docs/GUIDE.md describes src/a.py", problems[0])

    def test_the_page_in_the_same_commit_satisfies_the_rule(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._write("docs/GUIDE.md", K_PAGE + "More.\n")
        self._commit("fix: change a")
        self.assertEqual(self._problems(), [])

    def test_a_docs_line_with_a_reason_satisfies_the_rule(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._commit("fix: change a\n\nDocs: docs/GUIDE.md unchanged (the page names no value)")
        self.assertEqual(self._problems(), [])

    def test_a_waiver_with_a_reason_satisfies_every_page(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._commit("fix: change a\n\nDocs: none (a comment fix)")
        self.assertEqual(self._problems(), [])

    def test_a_described_folder_covers_the_files_below_it(self) -> None:
        self._write("core/deep/x.py", "c = 2\n")
        self._commit("fix: change x")
        self.assertIn("docs/GUIDE.md describes core", self._problems()[0])

    def test_a_renamed_described_file_is_a_problem(self) -> None:
        self._git("mv", "src/a.py", "src/b.py")
        self._commit("refactor: rename a")
        self.assertIn("docs/GUIDE.md describes src/a.py", self._problems()[0])

    def test_a_file_nobody_describes_is_free(self) -> None:
        self._write("other/x.py", "q = 1\n")
        self._commit("feat: add x")
        self.assertEqual(self._problems(), [])


class ReadmeTest(RepositoryTest):
    """Covers a README that lists the entries of its folder."""

    def test_a_new_entry_needs_the_readme(self) -> None:
        self._write("lib/new.py", "n = 1\n")
        self._commit("feat: add new")
        self.assertIn("lib/README.md lists its folder, and this commit adds lib/new.py", self._problems()[0])

    def test_a_new_folder_is_one_new_entry(self) -> None:
        self._write("lib/fresh/x.py", "n = 1\n")
        self._commit("feat: add fresh")
        self.assertIn("adds lib/fresh", self._problems()[0])

    def test_a_file_inside_an_existing_entry_is_free(self) -> None:
        self._write("lib/sub/more.py", "n = 1\n")
        self._commit("feat: add more")
        self.assertEqual(self._problems(), [])

    def test_an_edit_of_an_existing_file_is_free(self) -> None:
        self._write("lib/old.py", "y = 2\n")
        self._commit("fix: edit old")
        self.assertEqual(self._problems(), [])

    def test_removing_an_entry_needs_the_readme(self) -> None:
        self._git("rm", "-q", "lib/old.py")
        self._commit("refactor: drop old")
        self.assertIn("removes lib/old.py", self._problems()[0])

    def test_removing_one_file_of_a_folder_that_stays_is_free(self) -> None:
        self._write("lib/sub/two.py", "n = 1\n")
        self._commit("feat: add two\n\nDocs: lib/README.md unchanged (inside an entry)")
        self._git("rm", "-q", "lib/sub/two.py")
        self._commit("refactor: drop two")
        self.assertEqual(self._problems(), [])

    def test_the_readme_in_the_same_commit_satisfies_the_rule(self) -> None:
        self._write("lib/new.py", "n = 1\n")
        self._write("lib/README.md", "# Lib\n\nnew.py\n")
        self._commit("feat: add new")
        self.assertEqual(self._problems(), [])

    def test_work_folders_and_the_backlog_are_not_followed(self) -> None:
        self._write("docs/README.md", "# Docs\n")
        self._commit("docs: add index\n\nDocs: none (the index itself)")
        self._write("docs/agent/2026_01_01_WORK/SPEC.md", "# S\n")
        self._commit("docs: open work")
        self.assertEqual(self._problems(), [])


class CheckerTest(RepositoryTest):
    """Covers which commits the checker reads."""

    def test_nothing_is_read_without_a_base(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._commit("fix: change a")
        with mock.patch.object(docs_follow, "K_SINCE", ""):
            self.assertEqual(list(check_docs_follow.collect([self.root])), [])

    def test_only_the_commits_after_the_base_are_read(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._commit("fix: change a")
        self.assertEqual(len(commits_since(self.root, self.base)), 1)


class HookTest(RepositoryTest):
    """Covers the commit-msg hook seeing the staged change."""

    def _run(self, message: str) -> tuple[int, str]:
        path = self.root / "MESSAGE"
        path.write_text(message, encoding="utf-8")
        errors = io.StringIO()
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with (mock.patch.object(docs_follow, "K_SINCE", self.base),
                  mock.patch.object(check_backlog_trailer, "K_SINCE", ""),
                  contextlib.redirect_stderr(errors)):
                code = commit_msg.main([str(path)])
        finally:
            os.chdir(previous)
        return code, errors.getvalue()

    def test_refuses_a_staged_change_that_leaves_its_page_behind(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._git("add", "-A")
        code, errors = self._run("fix: change a\n")
        self.assertEqual(code, 1)
        self.assertIn("docs/GUIDE.md describes src/a.py", errors)

    def test_accepts_the_page_staged_with_it(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._write("docs/GUIDE.md", K_PAGE + "More.\n")
        self._git("add", "-A")
        self.assertEqual(self._run("fix: change a\n")[0], 0)

    def test_accepts_a_docs_line(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._git("add", "-A")
        self.assertEqual(self._run("fix: change a\n\nDocs: docs/GUIDE.md unchanged (no value named)\n")[0], 0)

    def test_a_new_entry_is_seen_in_the_index(self) -> None:
        self._write("lib/new.py", "n = 1\n")
        self._git("add", "-A")
        code, errors = self._run("feat: add new\n")
        self.assertEqual(code, 1)
        self.assertIn("lib/README.md lists its folder", errors)


if __name__ == "__main__":
    unittest.main()
