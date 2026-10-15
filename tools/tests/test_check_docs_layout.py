# test_check_docs_layout.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests the required-entry and reachability rules of the docs layout checker."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checking.checker import Issue  # noqa: E402
from documentation.check_docs_layout import (  # noqa: E402
    K_REQUIRED_ENTRIES,
    K_CHECKER,
    Repository,
    rule_docs_entries,
    rule_link_anchors,
    rule_links,
    rule_reachable_from_index,
    rule_required_entries,
)


def _repository(folder: str, files: dict[str, str]) -> Repository:
    """Returns a repository whose docs/ tree holds the given files."""
    root = Path(folder)
    for name, text in files.items():
        path = root / "docs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Repository(root)


def _names(issues: Iterable[Issue]) -> list[str]:
    """Returns the docs-relative path of each issue."""
    return [issue.path.as_posix().split("/docs/", 1)[1] for issue in issues]


class RequiredEntriesTest(unittest.TestCase):
    """Checks that each required document is demanded."""

    def test_is_registered(self) -> None:
        """Runs both rules as part of the checker's rule table."""
        self.assertIs(K_CHECKER.rules["rule_required_entries"], rule_required_entries)
        self.assertIs(K_CHECKER.rules["rule_reachable_from_index"], rule_reachable_from_index)

    def test_reports_every_missing_document(self) -> None:
        """Reports all required documents for a docs/ tree that holds none."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {"guides/X.md": "# X\n"})
            self.assertEqual(_names(rule_required_entries(repository)), list(K_REQUIRED_ENTRIES))

    def test_accepts_a_complete_tree(self) -> None:
        """Reports nothing when every required document exists."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {name: "# T\n" for name in K_REQUIRED_ENTRIES})
            self.assertEqual(list(rule_required_entries(repository)), [])


class ReachabilityTest(unittest.TestCase):
    """Checks which documents count as reached from the index."""

    def test_follows_links_through_an_index(self) -> None:
        """Reaches a document linked from an index the root index links, anchor included."""
        files = {
            "README.md": "# Manual\n[design](design/README.md)\n[guide](guides/GUIDE.md#install)\n",
            "design/README.md": "# Design\n[topic](TOPIC.md)\n",
            "design/TOPIC.md": "# Topic\n",
            "guides/GUIDE.md": "# Guide\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_reachable_from_index(_repository(folder, files))), [])

    def test_reports_an_orphan(self) -> None:
        """Reports a document no index links."""
        files = {"README.md": "# Manual\n", "guides/ORPHAN.md": "# Orphan\n"}
        with tempfile.TemporaryDirectory() as folder:
            issues = rule_reachable_from_index(_repository(folder, files))
            self.assertEqual(_names(issues), ["guides/ORPHAN.md"])

    def test_ignores_links_in_code(self) -> None:
        """Does not count a link inside a fence or a code span as reaching its target."""
        files = {
            "README.md": "# Manual\n`[a](guides/A.md)`\n```\n[b](guides/B.md)\n```\n",
            "guides/A.md": "# A\n",
            "guides/B.md": "# B\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = rule_reachable_from_index(_repository(folder, files))
            self.assertEqual(_names(issues), ["guides/A.md", "guides/B.md"])

    def test_exempts_work_folders_and_archive(self) -> None:
        """Reports nothing for unlinked documents under agent/ and archive/."""
        files = {
            "README.md": "# Manual\n",
            "agent/2026_10_05_WORK/SPEC.md": "# Spec\n",
            "archive/OLD.md": "# Old\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_reachable_from_index(_repository(folder, files))), [])

    def test_reads_titled_wrapped_bracketed_and_queried_links(self) -> None:
        """Reaches targets written with a title, a wrapped text, angle brackets or a query."""
        files = {
            "README.md": (
                '# Manual\n[a](guides/A.md "the title")\n[a link whose text\nwraps](guides/B.md)\n'
                "[c](<guides/C.md>)\n[d](guides/D.md?plain=1)\n[e](guides/MY%20E.md)\n"
            ),
            "guides/A.md": "# A\n",
            "guides/B.md": "# B\n",
            "guides/C.md": "# C\n",
            "guides/D.md": "# D\n",
            "guides/MY E.md": "# E\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, files)
            self.assertEqual(_names(rule_reachable_from_index(repository)), [])
            self.assertEqual(_names(rule_links(repository)), [])

    def test_reports_a_broken_titled_link_on_its_line(self) -> None:
        """Reports a titled link whose target is missing, with the line it starts on."""
        files = {"README.md": '# Manual\n\n[a](guides/GONE.md "the title")\n'}
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_links(_repository(folder, files)))
            self.assertEqual([(issue.line, issue.message) for issue in issues],
                             [(3, "link target does not exist: guides/GONE.md")])

    def test_does_not_reach_through_a_file_outside_docs(self) -> None:
        """Reports a guide the index reaches only by way of the root README."""
        files = {"README.md": "# Manual\n[front](../README.md)\n", "guides/A.md": "# A\n"}
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, files)
            (Path(folder) / "README.md").write_text("[a](docs/guides/A.md)\n", encoding="utf-8")
            self.assertEqual(_names(rule_reachable_from_index(repository)), ["guides/A.md"])

    def test_does_not_reach_through_a_work_folder(self) -> None:
        """Reports a guide linked only from an agent work folder the index links."""
        files = {
            "README.md": "# Manual\n[spec](agent/2026_10_05_WORK/SPEC.md)\n",
            "agent/2026_10_05_WORK/SPEC.md": "# Spec\n[a](../../guides/A.md)\n",
            "guides/A.md": "# A\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = rule_reachable_from_index(_repository(folder, files))
            self.assertEqual(_names(issues), ["guides/A.md"])

    def test_is_silent_without_an_index(self) -> None:
        """Reports nothing when docs/README.md is absent; the required-entry rule owns that."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {"guides/X.md": "# X\n"})
            self.assertEqual(list(rule_reachable_from_index(repository)), [])


class DocsEntriesTest(unittest.TestCase):
    """Checks which entries may sit directly under docs/."""

    def test_accepts_the_publish_list(self) -> None:
        """Accepts docs/publish.toml and still reports any other loose file."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(
                folder, {"README.md": "# R\n", "publish.toml": "", "notes.txt": ""},
            )
            self.assertEqual(_names(rule_docs_entries(repository)), ["notes.txt"])

    def test_accepts_the_docs_truth_allow_list(self) -> None:
        """Accepts docs/.docs_truth_allow, the one configuration file a checker reads."""
        with tempfile.TemporaryDirectory() as folder:
            repository = _repository(folder, {"README.md": "# R\n", ".docs_truth_allow": ""})
            self.assertEqual(_names(rule_docs_entries(repository)), [])


class LinkAnchorsTest(unittest.TestCase):
    """Covers the rule that a link's anchor names a heading of the document it reaches."""

    def test_accepts_a_heading_slug_and_a_label(self) -> None:
        """Checks that a slugged heading, a repeated heading and a {#label} all resolve."""
        files = {
            "README.md": "[a](guides/A.md#the-first-step) [b](guides/A.md#the-first-step-1) "
                         "[c](guides/A.md#module-a) [d](#index)\n\n## Index\n",
            "guides/A.md": "# A {#module-a}\n\n## The `first` step\n\n## The first step\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(list(rule_link_anchors(_repository(folder, files))), [])

    def test_reports_an_anchor_no_heading_has(self) -> None:
        """Checks that an anchor of a renamed section is a finding on the line of the link."""
        files = {
            "README.md": "# Index\n\n[a](guides/A.md#the-old-name)\n",
            "guides/A.md": "# A\n\n## The new name\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            issues = list(rule_link_anchors(_repository(folder, files)))
        self.assertEqual([issue.line for issue in issues], [3])

    def test_ignores_a_heading_inside_a_fence(self) -> None:
        """Checks that a heading line inside a fenced block gives no anchor."""
        files = {
            "README.md": "# Index\n\n[a](guides/A.md#fenced)\n",
            "guides/A.md": "# A\n\n```\n## Fenced\n```\n",
        }
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(len(list(rule_link_anchors(_repository(folder, files)))), 1)


if __name__ == "__main__":
    unittest.main()
