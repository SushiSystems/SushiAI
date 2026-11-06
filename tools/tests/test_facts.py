# test_facts.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests the facts core on values in memory, and the command, checker and follow rule on a tree."""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.source_file import SourceFile  # noqa: E402
from documentation import (  # noqa: E402
    check_docs_facts,
    check_docs_follow,
    docs_follow,
    facts,
    facts_cli,
)

K_HOME = (
    "# Home\n\n<!-- fact:begin shipping -->\n"
    "A topic ships when its last phase is built.\n<!-- fact:end shipping -->\n"
)
K_QUOTE = (
    "# Page\n\n<!-- quote:begin shipping -->\n"
    "A topic ships when its last phase is built.\n<!-- quote:end shipping -->\n"
)
K_FOLLOW = "# Page\n\n<!-- follows: shipping -->\nThe topic is done after its last phase.\n"
K_ROOT = Path("repo")


def _manual(pages: dict[str, str]) -> check_docs_facts.Manual:
    """Returns the manual of a made-up repository holding the given pages, built from values."""
    return check_docs_facts.build_manual(
        K_ROOT, [SourceFile(K_ROOT / name, text) for name, text in pages.items()],
    )


def _messages(rule, pages: dict[str, str]) -> list[str]:
    """Returns the message of each issue the rule finds in the given pages."""
    return [issue.message for issue in rule(_manual(pages))]


class TestParsing(unittest.TestCase):
    """Covers how the markers of one page are read, from text in memory."""
    K_TEST_SIZE = "small"

    def test_a_fact_has_its_id_lines_and_body(self) -> None:
        block = facts.parse("a.md", K_HOME).facts[0]
        self.assertEqual((block.id, block.first, block.last), ("shipping", 3, 5))
        self.assertEqual(block.body, "A topic ships when its last phase is built.")

    def test_a_multi_line_fact_keeps_every_line_of_its_body(self) -> None:
        text = "<!-- fact:begin x -->\nfirst\nsecond\n<!-- fact:end x -->\n"
        self.assertEqual(facts.parse("a.md", text).facts[0].body, "first\nsecond")

    def test_an_empty_fact_has_an_empty_body(self) -> None:
        text = "<!-- fact:begin x -->\n<!-- fact:end x -->\n"
        self.assertEqual(facts.parse("a.md", text).facts[0].body, "")

    def test_a_quote_is_found(self) -> None:
        self.assertEqual(facts.parse("a.md", K_QUOTE).quotes[0].id, "shipping")

    def test_a_follow_mark_is_found_with_its_line(self) -> None:
        self.assertEqual(facts.parse("a.md", K_FOLLOW).follows, [("shipping", 3)])

    def test_a_follow_mark_inside_a_fact_is_body_text(self) -> None:
        text = "<!-- fact:begin x -->\n<!-- follows: y -->\n<!-- fact:end x -->\n"
        page = facts.parse("a.md", text)
        self.assertEqual((page.follows, page.facts[0].body), ([], "<!-- follows: y -->"))

    def test_a_well_formed_page_has_no_faults(self) -> None:
        self.assertEqual(facts.parse("a.md", K_HOME + K_QUOTE + K_FOLLOW).faults, [])

    def test_a_fact_never_closed_is_a_fault(self) -> None:
        found = facts.parse("a.md", "<!-- fact:begin x -->\ntext\n").faults
        self.assertIn("never closed", found[0][1])

    def test_an_end_without_its_beginning_is_a_fault(self) -> None:
        found = facts.parse("a.md", "<!-- fact:end x -->\n").faults
        self.assertIn("without its beginning", found[0][1])

    def test_an_end_with_another_id_is_a_fault(self) -> None:
        self.assertTrue(facts.parse("a.md", "<!-- fact:begin x -->\n<!-- fact:end y -->\n").faults)

    def test_a_quote_end_that_closes_a_fact_is_a_fault(self) -> None:
        self.assertTrue(facts.parse("a.md", "<!-- fact:begin x -->\n<!-- quote:end x -->\n").faults)

    def test_a_marker_inside_another_block_is_a_fault(self) -> None:
        nested = ("<!-- fact:begin x -->\n<!-- quote:begin y -->\n"
                  "<!-- quote:end y -->\n<!-- fact:end x -->\n")
        self.assertIn("begins inside", facts.parse("a.md", nested).faults[0][1])

    def test_crlf_text_is_read_the_same_as_lf(self) -> None:
        self.assertEqual(
            facts.parse("a.md", K_HOME.replace("\n", "\r\n")).facts,
            facts.parse("a.md", K_HOME).facts,
        )


class TestRegeneration(unittest.TestCase):
    """Covers how a quote is rewritten from the body of its home."""
    K_TEST_SIZE = "small"

    def test_regenerate_replaces_only_the_quote_bodies(self) -> None:
        stale = K_QUOTE.replace("last phase is built", "old text")
        home = {"shipping": "A topic ships when its last phase is built."}
        self.assertEqual(facts.regenerate(stale, home), K_QUOTE)

    def test_regenerate_keeps_crlf_and_unknown_quotes(self) -> None:
        crlf = K_QUOTE.replace("\n", "\r\n")
        self.assertEqual(facts.regenerate(crlf, {}), crlf)

    def test_regenerate_rewrites_every_stale_quote_of_a_page(self) -> None:
        stale = K_QUOTE.replace("last phase is built", "old text")
        home = {"shipping": "A topic ships when its last phase is built."}
        self.assertEqual(facts.regenerate(stale + stale, home), K_QUOTE + K_QUOTE)

    def test_an_empty_home_empties_the_quote(self) -> None:
        self.assertEqual(
            facts.regenerate(K_QUOTE, {"shipping": ""}),
            "# Page\n\n<!-- quote:begin shipping -->\n<!-- quote:end shipping -->\n",
        )

    def test_a_multi_line_home_replaces_the_body_with_all_its_lines(self) -> None:
        self.assertEqual(
            facts.regenerate(K_QUOTE, {"shipping": "one\ntwo"}),
            "# Page\n\n<!-- quote:begin shipping -->\none\ntwo\n<!-- quote:end shipping -->\n",
        )


class TestRules(unittest.TestCase):
    """Covers the four rules of the facts checker, on manuals built in memory."""
    K_TEST_SIZE = "small"

    def test_a_current_quote_passes_every_rule(self) -> None:
        pages = {"home.md": K_HOME, "page.md": K_QUOTE + K_FOLLOW}
        for rule in check_docs_facts.K_CHECKER.rules.values():
            self.assertEqual(_messages(rule, pages), [])

    def test_a_stale_quote_names_its_home_and_the_command(self) -> None:
        pages = {"home.md": K_HOME, "page.md": K_QUOTE.replace("last phase", "final phase")}
        message = _messages(check_docs_facts.rule_quote_current, pages)[0]
        self.assertIn("home.md", message)
        self.assertIn("facts_cli.py apply", message)

    def test_a_quote_whose_fact_has_no_home_is_left_to_the_existence_rule(self) -> None:
        self.assertEqual(_messages(check_docs_facts.rule_quote_current, {"page.md": K_QUOTE}), [])

    def test_two_homes_for_one_id_are_refused(self) -> None:
        found = _messages(check_docs_facts.rule_unique_ids, {"a.md": K_HOME, "b.md": K_HOME})
        self.assertEqual(len(found), 1)

    def test_one_id_twice_on_one_page_is_refused(self) -> None:
        found = _messages(check_docs_facts.rule_unique_ids, {"a.md": K_HOME + K_HOME})
        self.assertEqual(len(found), 1)

    def test_a_quote_or_follow_of_a_missing_fact_is_refused(self) -> None:
        found = _messages(check_docs_facts.rule_cited_fact_exists, {"page.md": K_QUOTE + K_FOLLOW})
        self.assertEqual(len(found), 2)

    def test_a_follow_of_a_missing_fact_names_the_fact(self) -> None:
        pages = {"page.md": "<!-- follows: ghost -->\ntext\n"}
        found = _messages(check_docs_facts.rule_cited_fact_exists, pages)
        self.assertEqual(found, ["follows ghost, which has no home"])

    def test_a_home_on_the_quoting_page_satisfies_the_quote(self) -> None:
        pages = {"page.md": K_HOME + K_QUOTE}
        self.assertEqual(_messages(check_docs_facts.rule_cited_fact_exists, pages), [])

    def test_faulty_markers_are_reported(self) -> None:
        found = _messages(check_docs_facts.rule_markers, {"page.md": "<!-- fact:begin x -->\n"})
        self.assertEqual(len(found), 1)

    def test_a_marker_fault_is_reported_on_its_line(self) -> None:
        manual = _manual({"page.md": "text\n<!-- fact:end x -->\n"})
        issues = list(check_docs_facts.rule_markers(manual))
        self.assertEqual(issues[0].line, 2)

    def test_the_first_home_of_an_id_is_the_one_kept(self) -> None:
        manual = _manual({"a.md": K_HOME, "b.md": K_HOME.replace("built", "shipped")})
        self.assertEqual(manual.homes()["shipping"][0].path, "a.md")

    def test_a_quote_is_not_a_home(self) -> None:
        self.assertEqual(_manual({"page.md": K_QUOTE}).homes(), {})


class TestCitations(unittest.TestCase):
    """Covers which places the command lists for a fact, from manuals built in memory."""
    K_TEST_SIZE = "small"

    def test_citations_list_quotes_and_follows_in_page_order(self) -> None:
        manual = _manual({"quote.md": K_QUOTE, "follow.md": K_FOLLOW})
        self.assertEqual(
            facts_cli.citations(manual, "shipping"),
            ["quote.md:3: quote", "follow.md:3: follows"],
        )

    def test_every_quote_of_a_fact_is_listed(self) -> None:
        manual = _manual({"a.md": K_QUOTE, "b.md": K_QUOTE})
        self.assertEqual(len(facts_cli.citations(manual, "shipping")), 2)

    def test_citations_skip_the_quotes_of_other_facts(self) -> None:
        manual = _manual({"quote.md": K_QUOTE, "other.md": K_QUOTE.replace("shipping", "other")})
        self.assertEqual(facts_cli.citations(manual, "shipping"), ["quote.md:3: quote"])

    def test_an_uncited_fact_has_no_citations(self) -> None:
        self.assertEqual(facts_cli.citations(_manual({"home.md": K_HOME}), "shipping"), [])


class TestPlanApply(unittest.TestCase):
    """Covers which pages the rewrite changes, and with what text, before anything is written."""
    K_TEST_SIZE = "small"

    def test_a_stale_quote_is_planned_with_its_new_text(self) -> None:
        stale = K_QUOTE.replace("last phase", "final phase")
        manual = _manual({"home.md": K_HOME, "page.md": stale})
        self.assertEqual(facts_cli.plan_apply(manual), [("page.md", K_QUOTE)])

    def test_pages_with_current_quotes_are_not_planned(self) -> None:
        manual = _manual({"home.md": K_HOME, "page.md": K_QUOTE, "other.md": "# Other\n"})
        self.assertEqual(facts_cli.plan_apply(manual), [])

    def test_planning_a_planned_text_again_plans_nothing(self) -> None:
        stale = K_QUOTE.replace("last phase", "final phase")
        first = facts_cli.plan_apply(_manual({"home.md": K_HOME, "page.md": stale}))
        again = _manual({"home.md": K_HOME, "page.md": first[0][1]})
        self.assertEqual(facts_cli.plan_apply(again), [])

    def test_planned_pages_keep_their_order(self) -> None:
        stale = K_QUOTE.replace("last phase", "final phase")
        manual = _manual({"home.md": K_HOME, "b.md": stale, "a.md": stale})
        self.assertEqual([page for page, _ in facts_cli.plan_apply(manual)], ["b.md", "a.md"])


class TestPageSelection(unittest.TestCase):
    """Covers which pages the manual reads and the names it gives them, from paths in memory."""
    K_TEST_SIZE = "small"

    def test_build_folders_are_skipped(self) -> None:
        self.assertFalse(check_docs_facts.is_checked_page(K_ROOT, K_ROOT / "build" / "a.md"))

    def test_third_party_and_dependency_folders_are_skipped(self) -> None:
        third_party = K_ROOT / "docs" / "third_party" / "b.md"
        self.assertFalse(check_docs_facts.is_checked_page(K_ROOT, third_party))
        self.assertFalse(check_docs_facts.is_checked_page(K_ROOT, K_ROOT / "node_modules" / "c.md"))

    def test_ordinary_folders_are_checked(self) -> None:
        guide = K_ROOT / "docs" / "guides" / "d.md"
        self.assertTrue(check_docs_facts.is_checked_page(K_ROOT, guide))

    def test_page_paths_are_relative_and_posix(self) -> None:
        self.assertEqual(_manual({"docs/guides/a.md": K_HOME}).pages[0].path, "docs/guides/a.md")


class TestTreeOnDisk(unittest.TestCase):
    """Runs the command, the checker and the follow rule on tiny trees in temporary folders."""
    K_TEST_SIZE = "medium"

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def write(self, name: str, text: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")

    def _run(self, *arguments: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = facts_cli.main(["--root", str(self.root), *arguments])
        return code, out.getvalue()

    def _git(self, *arguments: str) -> str:
        identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "-c", "commit.gpgsign=false"]
        return subprocess.run(["git", "-C", str(self.root), *identity, *arguments],
                              check=True, capture_output=True, text=True).stdout

    def _commit(self, message: str) -> None:
        self._git("add", "-A")
        self._git("commit", "-q", "-m", message)

    def _start_repository(self) -> None:
        """Makes a git repository of a home and a page, as the base of the follow tests."""
        self._git("init", "-q")
        self.write("home.md", K_HOME)
        self.write("follow.md", K_FOLLOW)
        self._commit("start")
        self.base = self._git("rev-parse", "HEAD").strip()

    def _follow_problems(self) -> list[str]:
        with mock.patch.object(docs_follow, "K_SINCE", self.base):
            return [issue.message for commit in check_docs_follow.collect([self.root])
                    for issue in check_docs_follow.rule_pages_follow(commit)]

    def test_cited_by_lists_every_quote_and_follow(self) -> None:
        self.write("home.md", K_HOME)
        self.write("quote.md", K_QUOTE)
        self.write("follow.md", K_FOLLOW)
        code, output = self._run("cited-by", "shipping")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(output.splitlines()), ["follow.md:3: follows", "quote.md:3: quote"])

    def test_where_and_list_name_the_home(self) -> None:
        self.write("home.md", K_HOME)
        self.write("quote.md", K_QUOTE)
        self.assertEqual(self._run("where", "shipping")[1].strip(), "home.md:3")
        self.assertEqual(self._run("list")[1].strip(), "shipping: home.md:3, cited 1")

    def test_an_unknown_id_is_an_error(self) -> None:
        self.assertEqual(self._run("cited-by", "nothing")[0], 1)

    def test_apply_rewrites_a_stale_quote_and_a_second_run_writes_nothing(self) -> None:
        self.write("home.md", K_HOME)
        self.write("page.md", K_QUOTE.replace("last phase", "final phase"))
        self.assertEqual(self._run("apply")[1].strip(), "page.md: written")
        self.assertEqual((self.root / "page.md").read_text(encoding="utf-8"), K_QUOTE)
        self.assertEqual(self._run("apply")[1], "")

    def test_apply_refuses_a_stale_page_that_is_not_utf8_and_writes_nothing(self) -> None:
        self.write("home.md", K_HOME)
        bad = b"# Bad\n\xff\n<!-- quote:begin shipping -->\nold\n<!-- quote:end shipping -->\n"
        (self.root / "bad.md").write_bytes(bad)
        with self.assertRaises(UnicodeDecodeError):
            self._run("apply")
        self.assertEqual((self.root / "bad.md").read_bytes(), bad)

    def test_the_checker_passes_a_clean_tree_and_names_a_stale_quote(self) -> None:
        self.write("home.md", K_HOME)
        self.write("page.md", K_QUOTE)
        self.assertEqual(self._check(), (0, ""))
        self.write("page.md", K_QUOTE.replace("last phase", "final phase"))
        code, output = self._check()
        self.assertEqual(code, 1)
        self.assertIn("rule_quote_current", output)
        self.assertIn("facts_cli.py apply", output)

    def _check(self) -> tuple[int, str]:
        """Returns the exit code and the printed findings of the checker run on the tree."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = check_docs_facts.run(check_docs_facts.K_CHECKER, [str(self.root)])
        return code, out.getvalue()

    def test_changing_a_fact_lists_the_page_that_follows_it(self) -> None:
        self._start_repository()
        self.write("home.md", K_HOME.replace("last phase", "final phase"))
        self._commit("change the fact")
        problems = self._follow_problems()
        self.assertEqual(len(problems), 1)
        self.assertIn("follow.md follows the fact shipping", problems[0])

    def test_the_followers_page_in_the_commit_satisfies_the_rule(self) -> None:
        self._start_repository()
        self.write("home.md", K_HOME.replace("last phase", "final phase"))
        self.write("follow.md", K_FOLLOW + "Checked.\n")
        self._commit("change the fact")
        self.assertEqual(self._follow_problems(), [])

    def test_a_docs_line_satisfies_the_rule(self) -> None:
        self._start_repository()
        self.write("home.md", K_HOME.replace("last phase", "final phase"))
        self._commit("change the fact\n\nDocs: follow.md unchanged (the paraphrase still holds)")
        self.assertEqual(self._follow_problems(), [])

    def test_editing_text_outside_the_fact_is_free(self) -> None:
        self._start_repository()
        self.write("home.md", K_HOME + "\nMore text outside the fact.\n")
        self._commit("edit around the fact")
        self.assertEqual(self._follow_problems(), [])


if __name__ == "__main__":
    unittest.main()
