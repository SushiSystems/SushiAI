# test_facts.py
# SushiSkills - https://github.com/SushiSystems/SushiSkills
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests reading facts, quotes and follow marks, the checker of them, the command and the follow rule."""

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

from documentation import check_docs_facts, docs_follow, facts, facts_cli  # noqa: E402

K_HOME = "# Home\n\n<!-- fact:begin shipping -->\nA topic ships when its last phase is built.\n<!-- fact:end shipping -->\n"
K_QUOTE = "# Page\n\n<!-- quote:begin shipping -->\nA topic ships when its last phase is built.\n<!-- quote:end shipping -->\n"
K_FOLLOW = "# Page\n\n<!-- follows: shipping -->\nThe topic is done after its last phase.\n"


class ParseTest(unittest.TestCase):
    """Covers how the markers of a page are read."""

    def test_a_fact_has_its_id_lines_and_body(self) -> None:
        block = facts.parse("a.md", K_HOME).facts[0]
        self.assertEqual((block.id, block.first, block.last), ("shipping", 3, 5))
        self.assertEqual(block.body, "A topic ships when its last phase is built.")

    def test_a_quote_and_a_follow_mark_are_found(self) -> None:
        self.assertEqual(facts.parse("a.md", K_QUOTE).quotes[0].id, "shipping")
        self.assertEqual(facts.parse("a.md", K_FOLLOW).follows, [("shipping", 3)])

    def test_unpaired_misnamed_and_nested_markers_are_faults(self) -> None:
        self.assertIn("never closed", facts.parse("a.md", "<!-- fact:begin x -->\ntext\n").faults[0][1])
        self.assertIn("without its beginning", facts.parse("a.md", "<!-- fact:end x -->\n").faults[0][1])
        mismatch = "<!-- fact:begin x -->\n<!-- fact:end y -->\n"
        self.assertTrue(facts.parse("a.md", mismatch).faults)
        nested = "<!-- fact:begin x -->\n<!-- quote:begin y -->\n<!-- quote:end y -->\n<!-- fact:end x -->\n"
        self.assertIn("begins inside", facts.parse("a.md", nested).faults[0][1])

    def test_regenerate_replaces_only_the_quote_bodies(self) -> None:
        stale = K_QUOTE.replace("last phase is built", "old text")
        self.assertEqual(facts.regenerate(stale, {"shipping": "A topic ships when its last phase is built."}), K_QUOTE)

    def test_regenerate_keeps_crlf_and_unknown_quotes(self) -> None:
        crlf = K_QUOTE.replace("\n", "\r\n")
        self.assertEqual(facts.regenerate(crlf, {}), crlf)


class RepositoryTest(unittest.TestCase):
    """Builds a folder of pages for each test."""

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def write(self, name: str, text: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")

    def issues(self, rule) -> list[str]:
        return [issue.message for manual in check_docs_facts.collect([self.root]) for issue in rule(manual)]


class CheckerTest(RepositoryTest):
    """Covers the four rules of the facts checker."""

    def test_a_current_quote_passes_every_rule(self) -> None:
        self.write("home.md", K_HOME)
        self.write("page.md", K_QUOTE + K_FOLLOW)
        for rule in check_docs_facts.K_CHECKER.rules.values():
            self.assertEqual(self.issues(rule), [])

    def test_a_stale_quote_names_its_home_and_the_command(self) -> None:
        self.write("home.md", K_HOME)
        self.write("page.md", K_QUOTE.replace("last phase", "final phase"))
        message = self.issues(check_docs_facts.rule_quote_current)[0]
        self.assertIn("home.md", message)
        self.assertIn("facts_cli.py apply", message)

    def test_two_homes_for_one_id_are_refused(self) -> None:
        self.write("a.md", K_HOME)
        self.write("b.md", K_HOME)
        self.assertEqual(len(self.issues(check_docs_facts.rule_unique_ids)), 1)

    def test_a_quote_or_follow_of_a_missing_fact_is_refused(self) -> None:
        self.write("page.md", K_QUOTE + K_FOLLOW)
        self.assertEqual(len(self.issues(check_docs_facts.rule_cited_fact_exists)), 2)

    def test_faulty_markers_are_reported(self) -> None:
        self.write("page.md", "<!-- fact:begin x -->\n")
        self.assertEqual(len(self.issues(check_docs_facts.rule_markers)), 1)


class CommandTest(RepositoryTest):
    """Covers the questions the command answers and the rewrite."""

    def _run(self, *arguments: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = facts_cli.main(["--root", str(self.root), *arguments])
        return code, out.getvalue()

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


class FollowRuleTest(RepositoryTest):
    """Covers a commit that changes a fact a page follows."""

    def setUp(self) -> None:
        super().setUp()
        self._git("init", "-q")
        self.write("home.md", K_HOME)
        self.write("follow.md", K_FOLLOW)
        self._commit("start")
        self.base = self._git("rev-parse", "HEAD").strip()

    def _git(self, *arguments: str) -> str:
        identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]
        return subprocess.run(["git", "-C", str(self.root), *identity, *arguments],
                              check=True, capture_output=True, text=True).stdout

    def _commit(self, message: str) -> None:
        self._git("add", "-A")
        self._git("commit", "-q", "-m", message)

    def _problems(self) -> list[str]:
        from documentation import check_docs_follow
        with mock.patch.object(docs_follow, "K_SINCE", self.base):
            return [issue.message for commit in check_docs_follow.collect([self.root])
                    for issue in check_docs_follow.rule_pages_follow(commit)]

    def test_changing_a_fact_lists_the_page_that_follows_it(self) -> None:
        self.write("home.md", K_HOME.replace("last phase", "final phase"))
        self._commit("change the fact")
        problems = self._problems()
        self.assertEqual(len(problems), 1)
        self.assertIn("follow.md follows the fact shipping", problems[0])

    def test_the_followers_page_in_the_commit_satisfies_the_rule(self) -> None:
        self.write("home.md", K_HOME.replace("last phase", "final phase"))
        self.write("follow.md", K_FOLLOW + "Checked.\n")
        self._commit("change the fact")
        self.assertEqual(self._problems(), [])

    def test_a_docs_line_satisfies_the_rule(self) -> None:
        self.write("home.md", K_HOME.replace("last phase", "final phase"))
        self._commit("change the fact\n\nDocs: follow.md unchanged (the paraphrase still holds)")
        self.assertEqual(self._problems(), [])

    def test_editing_text_outside_the_fact_is_free(self) -> None:
        self.write("home.md", K_HOME + "\nMore text outside the fact.\n")
        self._commit("edit around the fact")
        self.assertEqual(self._problems(), [])


if __name__ == "__main__":
    unittest.main()
