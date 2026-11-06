# test_docs_follow.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the rule that a change updates the pages describing it.

The rules are tested on the values of one commit; the checker and the commit hook are tested on
a temporary git repository.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commits import check_backlog_trailer  # noqa: E402
from documentation import check_docs_follow, docs_follow  # noqa: E402
from documentation.check_docs_follow import CommitFollow  # noqa: E402
from documentation.docs_follow import (  # noqa: E402
    Change,
    Follow,
    PageFacts,
    Versions,
    acknowledged,
    describes_of,
    is_followed_page,
    page_facts,
    parse_changes,
)
from hooks import commit_msg  # noqa: E402

K_PAGE = "# Guide\n\n**Describes:** `src/a.py`, `core`\n\nText.\n"
K_GUIDE = page_facts("docs/GUIDE.md", K_PAGE)
K_LIB_README = page_facts("lib/README.md", "# Lib\n")
K_DOCS_README = page_facts("docs/README.md", "# Docs\n")
K_IGNORED = ("docs/agent", "docs/design", "docs/backlog", "docs/archive", "third_party")
K_TREE = frozenset({
    "core/deep/x.py", "docs/GUIDE.md", "docs/README.md", "lib/README.md", "lib/old.py",
    "lib/sub/inner.py", "other/x.py", "src/a.py",
})
K_FACT_OLD = "<!-- fact:begin ship-date -->\n2026-01-01\n<!-- fact:end ship-date -->\n"
K_FACT_NEW = "<!-- fact:begin ship-date -->\n2026-02-01\n<!-- fact:end ship-date -->\n"
K_FOLLOWER = page_facts("docs/GUIDE.md",
                        "# Guide\n\n<!-- follows: ship-date -->\n\nText.\n")
K_FACTS_FILE = "docs/FACTS.md"
K_ROOT = Path("repo")
K_SHA = "0123456789abcdef"


def _follow(
    changes: Iterable[Change] = (),
    pages: Iterable[PageFacts] = (),
    message: str = "",
    versions: Iterable[Versions] = (),
    before: Iterable[str] = K_TREE,
    after: Iterable[str] = K_TREE,
) -> Follow:
    """Returns the values one commit presents to the rules, with the checker's ignored folders."""
    return Follow(tuple(changes), message, K_IGNORED, tuple(pages), tuple(versions),
                  frozenset(before), frozenset(after))


def _problems(follow: Follow) -> list[str]:
    """Returns the message of each issue the checker's rule yields for the values of a commit."""
    subject = CommitFollow(K_ROOT, K_SHA, follow)
    return [issue.message for issue in check_docs_follow.rule_pages_follow(subject)]


class TestParsing(unittest.TestCase):
    """Covers reading the describes line, the status listing and the acknowledgements."""
    K_TEST_SIZE = "small"

    def test_the_describes_line_lists_its_code_spans(self) -> None:
        self.assertEqual(describes_of("# T\n\n**Describes:** `a/b.py`, `c/`\n"), ("a/b.py", "c"))

    def test_a_describes_line_below_the_head_of_the_page_is_ignored(self) -> None:
        self.assertEqual(describes_of("# T\n" + "x\n" * 20 + "**Describes:** `a`\n"), ())

    def test_the_status_listing_names_renames_with_both_paths(self) -> None:
        listing = "M\ta.py\nA\tb.py\nD\tc.py\nR100\told.py\tnew.py\n"
        self.assertEqual(parse_changes(listing), [
            Change("M", "a.py"), Change("A", "b.py"), Change("D", "c.py"),
            Change("R", "new.py", "old.py"),
        ])

    def test_a_page_is_acknowledged_with_a_reason_only(self) -> None:
        message = "x\n\nDocs: docs/A.md unchanged (wording only)\nDocs: docs/B.md unchanged ()\n"
        self.assertEqual(acknowledged(message), (False, {"docs/A.md"}))

    def test_none_with_a_reason_waives_every_page(self) -> None:
        self.assertEqual(acknowledged("x\n\nDocs: none (a comment fix)\n")[0], True)
        self.assertEqual(acknowledged("x\n\nDocs: none\n")[0], False)


class TestExplicitRule(unittest.TestCase):
    """Covers a page that names its files, judged on the values of one commit."""
    K_TEST_SIZE = "small"

    def test_a_changed_described_file_without_its_page_is_a_problem(self) -> None:
        problems = _problems(_follow([Change("M", "src/a.py")], [K_GUIDE]))
        self.assertEqual(len(problems), 1)
        self.assertIn("docs/GUIDE.md describes src/a.py", problems[0])

    def test_the_page_in_the_same_commit_satisfies_the_rule(self) -> None:
        changes = [Change("M", "src/a.py"), Change("M", "docs/GUIDE.md")]
        self.assertEqual(_problems(_follow(changes, [K_GUIDE])), [])

    def test_a_docs_line_with_a_reason_satisfies_the_rule(self) -> None:
        message = "fix: change a\n\nDocs: docs/GUIDE.md unchanged (the page names no value)"
        self.assertEqual(_problems(_follow([Change("M", "src/a.py")], [K_GUIDE], message)), [])

    def test_a_waiver_with_a_reason_satisfies_every_page(self) -> None:
        message = "fix: change a\n\nDocs: none (a comment fix)"
        self.assertEqual(_problems(_follow([Change("M", "src/a.py")], [K_GUIDE], message)), [])

    def test_a_docs_line_names_one_page_and_leaves_the_others_reported(self) -> None:
        other = page_facts("docs/OTHER.md", "# Other\n\n**Describes:** `src/a.py`\n")
        message = "fix: change a\n\nDocs: docs/GUIDE.md unchanged (the page names no value)"
        problems = _problems(_follow([Change("M", "src/a.py")], [K_GUIDE, other], message))
        self.assertEqual(len(problems), 1)
        self.assertIn("docs/OTHER.md describes src/a.py", problems[0])

    def test_a_described_folder_covers_the_files_below_it(self) -> None:
        problems = _problems(_follow([Change("M", "core/deep/x.py")], [K_GUIDE]))
        self.assertEqual(len(problems), 1)
        self.assertIn("docs/GUIDE.md describes core", problems[0])

    def test_a_renamed_described_file_is_a_problem(self) -> None:
        after = (K_TREE - {"src/a.py"}) | {"src/b.py"}
        problems = _problems(_follow([Change("R", "src/b.py", "src/a.py")], [K_GUIDE], after=after))
        self.assertEqual(len(problems), 1)
        self.assertIn("docs/GUIDE.md describes src/a.py", problems[0])

    def test_a_removed_described_file_is_a_problem(self) -> None:
        problems = _problems(_follow([Change("D", "src/a.py")], [K_GUIDE],
                                     after=K_TREE - {"src/a.py"}))
        self.assertEqual(len(problems), 1)
        self.assertIn("docs/GUIDE.md describes src/a.py", problems[0])

    def test_a_file_nobody_describes_is_free(self) -> None:
        problems = _problems(_follow([Change("A", "other/x.py")], [K_GUIDE],
                                     after=K_TREE | {"other/x.py"}))
        self.assertEqual(problems, [])


class TestReadmeRule(unittest.TestCase):
    """Covers a README that lists the entries of its folder, judged on the trees either side."""
    K_TEST_SIZE = "small"

    def test_a_new_entry_needs_the_readme(self) -> None:
        problems = _problems(_follow([Change("A", "lib/new.py")], [K_LIB_README],
                                     after=K_TREE | {"lib/new.py"}))
        self.assertEqual(len(problems), 1)
        self.assertIn("lib/README.md lists its folder, and this commit adds lib/new.py",
                      problems[0])

    def test_a_new_folder_is_one_new_entry(self) -> None:
        problems = _problems(_follow([Change("A", "lib/fresh/x.py")], [K_LIB_README],
                                     after=K_TREE | {"lib/fresh/x.py"}))
        self.assertEqual(len(problems), 1)
        self.assertIn("adds lib/fresh", problems[0])

    def test_a_file_inside_an_existing_entry_is_free(self) -> None:
        problems = _problems(_follow([Change("A", "lib/sub/more.py")], [K_LIB_README],
                                     after=K_TREE | {"lib/sub/more.py"}))
        self.assertEqual(problems, [])

    def test_an_edit_of_an_existing_file_is_free(self) -> None:
        self.assertEqual(_problems(_follow([Change("M", "lib/old.py")], [K_LIB_README])), [])

    def test_removing_an_entry_needs_the_readme(self) -> None:
        problems = _problems(_follow([Change("D", "lib/old.py")], [K_LIB_README],
                                     after=K_TREE - {"lib/old.py"}))
        self.assertEqual(len(problems), 1)
        self.assertIn("removes lib/old.py", problems[0])

    def test_removing_one_file_of_a_folder_that_stays_is_free(self) -> None:
        problems = _problems(_follow([Change("D", "lib/sub/two.py")], [K_LIB_README],
                                     before=K_TREE | {"lib/sub/two.py"}))
        self.assertEqual(problems, [])

    def test_the_readme_in_the_same_commit_satisfies_the_rule(self) -> None:
        changes = [Change("A", "lib/new.py"), Change("M", "lib/README.md")]
        problems = _problems(_follow(changes, [K_LIB_README], after=K_TREE | {"lib/new.py"}))
        self.assertEqual(problems, [])

    def test_a_work_folder_is_not_followed(self) -> None:
        spec = "docs/agent/2026_01_01_WORK/SPEC.md"
        problems = _problems(_follow([Change("A", spec)], [K_DOCS_README], after=K_TREE | {spec}))
        self.assertEqual(problems, [])

    def test_the_backlog_is_not_followed(self) -> None:
        entry = "docs/backlog/OPEN.md"
        problems = _problems(_follow([Change("A", entry)], [K_DOCS_README],
                                     after=K_TREE | {entry}))
        self.assertEqual(problems, [])


class TestFactRule(unittest.TestCase):
    """Covers a page that follows a fact by its id, judged on the fact's text before and after."""
    K_TEST_SIZE = "small"

    def test_a_page_following_a_changed_fact_is_a_problem(self) -> None:
        versions = [Versions(K_FACT_OLD, K_FACT_NEW)]
        problems = _problems(_follow([Change("M", K_FACTS_FILE)], [K_FOLLOWER], versions=versions))
        self.assertEqual(len(problems), 1)
        self.assertIn("docs/GUIDE.md follows the fact ship-date, which this commit changes",
                      problems[0])

    def test_a_page_following_an_unchanged_fact_is_free(self) -> None:
        versions = [Versions(K_FACT_OLD, K_FACT_OLD)]
        problems = _problems(_follow([Change("M", K_FACTS_FILE)], [K_FOLLOWER], versions=versions))
        self.assertEqual(problems, [])


class TestPages(unittest.TestCase):
    """Covers which paths the checker follows as pages."""
    K_TEST_SIZE = "small"

    def test_only_markdown_outside_the_ignored_folders_is_followed(self) -> None:
        self.assertTrue(is_followed_page("docs/GUIDE.md", K_IGNORED))
        self.assertFalse(is_followed_page("docs/agent/2026_01_01_WORK/SPEC.md", K_IGNORED))
        self.assertFalse(is_followed_page("third_party/lib/README.md", K_IGNORED))
        self.assertFalse(is_followed_page("src/a.py", K_IGNORED))


class TestRepository(unittest.TestCase):
    """Runs the checker and the commit hook on a temporary git repository."""
    K_TEST_SIZE = "medium"

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
        identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "-c", "commit.gpgsign=false"]
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

    def _problems(self, base: str | None = None) -> list[str]:
        """Returns the problems of the commits made after the base, as the checker reads them."""
        with mock.patch.object(docs_follow, "K_SINCE", base or self.base):
            return [issue.message for commit in check_docs_follow.collect([self.root])
                    for issue in check_docs_follow.rule_pages_follow(commit)]

    def test_nothing_is_read_without_a_base(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._commit("fix: change a")
        with mock.patch.object(docs_follow, "K_SINCE", ""):
            self.assertEqual(check_docs_follow.collect([self.root]), [])

    def test_only_the_commits_after_the_base_are_read(self) -> None:
        self._write("src/a.py", "x = 2\n")
        self._commit("fix: change a")
        base = self._git("rev-parse", "HEAD").strip()
        self._write("src/a.py", "x = 3\n")
        self._commit("fix: change a again")
        self.assertEqual(len(self._problems(base)), 1)

    def _run_hook(self, message: str) -> tuple[int, str]:
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

    def test_a_new_entry_is_seen_in_the_index(self) -> None:
        self._write("lib/new.py", "n = 1\n")
        self._git("add", "-A")
        code, errors = self._run_hook("feat: add new\n")
        self.assertEqual(code, 1)
        self.assertIn("lib/README.md lists its folder", errors)


if __name__ == "__main__":
    unittest.main()
